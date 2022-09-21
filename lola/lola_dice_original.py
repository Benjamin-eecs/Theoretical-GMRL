# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from copy import deepcopy
import hashlib
from envs import IPD
from argument import get_args
import copy
import json
import os
import pickle as pkl

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)

hp = get_args()
ipd_inner = IPD(hp.len_rollout, hp.inner_batch_size)
ipd_outer = IPD(hp.len_rollout, hp.outer_batch_size)

def magic_box(x):
    return torch.exp(x - x.detach())
def phi(x1,x2):
    return [x1*x2, x1*(1-x2), (1-x1)*x2,(1-x1)*(1-x2)]

def true_objective(theta1, theta2):
    p1 = torch.sigmoid(theta1)
    p2 = torch.sigmoid(theta2[[0,1,3,2,4]])
    p0 = (p1[0], p2[0])
    p = (p1[1:], p2[1:])
    # create initial laws, transition matrix and rewards:
    P0 = torch.stack(phi(*p0), dim=0).view(1,-1)
    P = torch.stack(phi(*p), dim=1)
    R = torch.from_numpy(ipd_inner.payout_mat).view(-1,1).float()
    # the true value to optimize:
    objective = (P0.mm(torch.inverse(torch.eye(4) - hp.gamma*P))).mm(R)
    return -objective

class Memory():
    def __init__(self):
        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []

    def add(self, lp, other_lp, v, r):
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

    def dice_objective(self, use_baseline=True):
        self_logprobs = torch.stack(self.self_logprobs, dim=1)
        other_logprobs = torch.stack(self.other_logprobs, dim=1)
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(hp.gamma * torch.ones(*rewards.size()), dim=1)/hp.gamma
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards, dim=1))

        if use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective # want to minimize -objective

    def value_loss(self):
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)
        return torch.mean((rewards - values)**2)

def act(batch_states, theta, values):
    batch_states = torch.from_numpy(batch_states).long()
    probs = torch.sigmoid(theta)[batch_states]
    m = Bernoulli(1-probs)
    actions = m.sample()
    log_probs_actions = m.log_prob(actions)
    return actions.numpy().astype(int), log_probs_actions, values[batch_states]

def get_gradient(objective, theta):
    # create differentiable gradient for 2nd orders:
    grad_objective = torch.autograd.grad(objective, (theta), create_graph=True)[0]
    return grad_objective

def step(theta1, theta2, values1, values2, ipd):
    # just to evaluate progress:
    (s1, s2), _ = ipd.reset()
    score1 = 0
    score2 = 0
    for t in range(hp.len_rollout):
        a1, lp1, v1 = act(s1, theta1, values1)
        a2, lp2, v2 = act(s2, theta2, values2)
        (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
        # cumulate scores
        score1 += np.mean(r1)/float(hp.len_rollout)
        score2 += np.mean(r2)/float(hp.len_rollout)
    return (score1, score2)

class Agent():
    def __init__(self):
        # init theta and its optimizer
        self.theta = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.theta_optimizer = torch.optim.Adam((self.theta,),lr=hp.lr_out)
        # init values and its optimizer
        self.values = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.value_optimizer = torch.optim.Adam((self.values,),lr=hp.lr_v)

    def theta_update(self, objective):
        self.theta_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        grad = self.theta.grad.detach()
        self.theta_optimizer.step()
        return grad

    def value_update(self, loss):
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def in_lookahead_exact(self, other_theta):
        other_objective = true_objective(other_theta, self.theta)
        grad = get_gradient(other_objective, other_theta)
        return grad

    def out_lookahead_exact_value(self, other_theta, other_values, ipd):
        (s1, s2), _ = ipd.reset()
        memory = Memory()
        for t in range(hp.len_rollout):
            a1, lp1, v1 = act(s1, self.theta, self.values)
            a2, lp2, v2 = act(s2, other_theta, other_values)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            memory.add(lp1, lp2, v1, torch.from_numpy(r1).float())
        
        objective = true_objective(self.theta, other_theta)
        grad = self.theta_update(objective)
        
        v_loss = memory.value_loss()
        self.value_update(v_loss)
        return grad, grad
    
    def out_lookahead_exact(self, other_theta):  
        objective = true_objective(self.theta, other_theta)
        grad = self.theta_update(objective)
        return grad

    def in_lookahead(self, other_theta, other_values, ipd):
        (s1, s2), _ = ipd.reset()
        other_memory = Memory()
        for t in range(hp.len_rollout):
            a1, lp1, v1 = act(s1, self.theta, self.values)
            a2, lp2, v2 = act(s2, other_theta, other_values)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            other_memory.add(lp2, lp1, v2, torch.from_numpy(r2).float())
        
        other_objective = other_memory.dice_objective(use_baseline=not hp.no_inner_baseline)
        grad = get_gradient(other_objective, other_theta)
        return grad

    def out_lookahead(self, other_theta, other_values, ipd):
        (s1, s2), _ = ipd.reset()
        memory = Memory()
        for t in range(hp.len_rollout):
            a1, lp1, v1 = act(s1, self.theta, self.values)
            a2, lp2, v2 = act(s2, other_theta, other_values)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            memory.add(lp1, lp2, v1, torch.from_numpy(r1).float())

        # For visualising graidient correlation, we won't use this gradient
        objective = true_objective(self.theta, other_theta)
        grad_outer_exact = torch.autograd.grad(objective, self.theta, retain_graph=True)[0].detach()      
        
        # update self theta
        objective = memory.dice_objective(use_baseline=not hp.no_outer_baseline)
        grad = self.theta_update(objective)
        # update self value:
        v_loss = memory.value_loss()
        self.value_update(v_loss)
        
        return grad, grad_outer_exact
        
def Corr(x, y):
        """
        Angular accuracy measure between tensors
        between -1 and 1, the higher the better

        Args:
            two tensors x and y
        Returns:
            Angular accuracy measure
        """
        x = x.flatten()
        y = y.flatten()
        x -= np.mean(x)
        y -= np.mean(y)
        return x.dot(y) / np.sqrt(x.dot(x) * y.dot(y))
    
def play(agent1, agent2, n_lookaheads):
    joint_scores = []
    joint_corr = []
    joint_corr_outer_exact = []
    print("start iterations with", n_lookaheads, "lookaheads:")
    for update in range(hp.n_update):
        agent1_exact = copy.deepcopy(agent1)
        agent2_exact = copy.deepcopy(agent2)
        
        # copy other's parameters:
        theta1_ = torch.tensor(agent1.theta.detach(), requires_grad=True)
        values1_ = torch.tensor(agent1.values.detach(), requires_grad=True)
        theta2_ = torch.tensor(agent2.theta.detach(), requires_grad=True)
        values2_ = torch.tensor(agent2.values.detach(), requires_grad=True)

        for k in range(n_lookaheads):
            # estimate other's gradients from in_lookahead:
            if hp.inner_exact:
                grad2 = agent1.in_lookahead_exact(theta2_)
                grad1 = agent2.in_lookahead_exact(theta1_)
            else:
                grad2 = agent1.in_lookahead(theta2_, values2_, ipd_inner)
                grad1 = agent2.in_lookahead(theta1_, values1_, ipd_inner)
            # update other's theta
            theta2_ = theta2_ - hp.lr_in * grad2
            theta1_ = theta1_ - hp.lr_in * grad1

        # update own parameters from out_lookahead:
        if hp.outer_exact:
            metagrad1, metagrad1_outer_exact = agent1.out_lookahead_exact_value(theta2_, values2_, ipd_outer)# ipd_outer for learning value function
            metagrad2, metagrad2_outer_exact = agent2.out_lookahead_exact_value(theta1_, values1_, ipd_outer)   
        else:
            metagrad1, metagrad1_outer_exact = agent1.out_lookahead(theta2_, values2_, ipd_outer)
            metagrad2, metagrad2_outer_exact = agent2.out_lookahead(theta1_, values1_, ipd_outer)

        # esimtate exact gradient
        theta1_ = torch.tensor(agent1_exact.theta.detach(), requires_grad=True)
        theta2_ = torch.tensor(agent2_exact.theta.detach(), requires_grad=True)
        
        for k in range(n_lookaheads):
            # estimate other's gradients from in_lookahead:
            grad2 = agent1_exact.in_lookahead_exact(theta2_)
            grad1 = agent2_exact.in_lookahead_exact(theta1_)
            # update other's theta
            theta2_ = theta2_ - hp.lr_in * grad2
            theta1_ = theta1_ - hp.lr_in * grad1
        
        # update own parameters from out_lookahead:
        metagrad1_exact = agent1_exact.out_lookahead_exact(theta2_)
        metagrad2_exact = agent2_exact.out_lookahead_exact(theta1_)
     
        corr = [0, 0]
        corr[0] = Corr(metagrad1.numpy(), metagrad1_exact.numpy())
        corr[1] = Corr(metagrad2.numpy(), metagrad2_exact.numpy())
        
        corr_outer_exact = [0,0]
        corr_outer_exact[0] = Corr(metagrad1_outer_exact.numpy(), metagrad1_exact.numpy())
        corr_outer_exact[1] = Corr(metagrad2_outer_exact.numpy(), metagrad2_exact.numpy())
        
        # evaluate progress:
        score = step(agent1.theta, agent2.theta, agent1.values, agent2.values, ipd_inner)
        joint_scores.append(0.5*(score[0] + score[1]))
        joint_corr.append(0.5*(corr[0] + corr[1]))
        joint_corr_outer_exact.append(0.5*(corr_outer_exact[0] + corr_outer_exact[1]))

        # print
        if update%10==0 :
            p1 = [p.item() for p in torch.sigmoid(agent1.theta)]
            p2 = [p.item() for p in torch.sigmoid(agent2.theta)]
            print(update, 'score (%.3f,%.3f)' % (score[0], score[1]), 'corr (%.3f,%.3f)' % (0.5*(corr[0] + corr[1]), 0.5*(corr_outer_exact[0] + corr_outer_exact[1])), 'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p1[0], p1[1], p1[2], p1[3], p1[4]),' (agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))

    return joint_scores, joint_corr, joint_corr_outer_exact

def get_path_from_args(args):
    """ Returns a unique hash for an argparse object. """
    args_str = str(args)
    path = hashlib.md5(args_str.encode()).hexdigest()
    return path   

class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        if callable(o):
            return {'function': o.__name__}
        return json.JSONEncoder.default(self, o)

# plot progress:
if __name__=="__main__":
    
    log_dir = os.path.expanduser(hp.logdir)
    log_dir = log_dir + '/test_%s' % get_path_from_args(hp)
    os.makedirs(log_dir)
    json.dump(vars(hp), open(log_dir + '/params.json', 'w'), cls=ClassEncoder)
    scores_all = []
    corr_all = []
    corr_all_outer_exact = []
    print(hp)

    for seed in range(10):
        print('seed', seed)
        set_seed(seed)
        scores, corr, corr_outer_exact = play(Agent(), Agent(), hp.inner_step)
        scores_all.append(scores)
        corr_all.append(corr)
        corr_all_outer_exact.append(corr_outer_exact)
    file_name = 'score_result.pkl'
    with open(os.path.join(log_dir, file_name), 'wb') as file:
        pkl.dump(scores_all, file)
    file_name = 'corr_result.pkl'
    with open(os.path.join(log_dir, file_name), 'wb') as file:
        pkl.dump(corr_all, file)
    file_name = 'corr_outer_exact_result.pkl'
    with open(os.path.join(log_dir, file_name), 'wb') as file:
        pkl.dump(corr_all_outer_exact, file)
        