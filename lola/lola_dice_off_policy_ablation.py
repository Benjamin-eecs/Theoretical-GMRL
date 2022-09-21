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

def off_policy_magic_box(x, y):
    return torch.exp(x - y.detach())

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

class off_policy_Buffer():
    def __init__(self, max_traj_size = 1024):
        self.ptr = 0
        self.max_size = max_traj_size
        self.self_state = np.zeros([self.max_size, hp.len_rollout])
        self.self_action = np.zeros([self.max_size, hp.len_rollout])
        self.other_state = np.zeros([self.max_size, hp.len_rollout])
        self.other_action = np.zeros([self.max_size, hp.len_rollout])
        self.self_logprobs = np.zeros([self.max_size, hp.len_rollout])
        self.other_logprobs = np.zeros([self.max_size, hp.len_rollout])
        self.reward = np.zeros([self.max_size, hp.len_rollout])
        self.reward2 = np.zeros([self.max_size, hp.len_rollout])
        
    def add(self, ss, sa, os, oa, slp, olp, r1, r2):#bs * hp_len_rollout
        length = ss.shape[0]
        assert self.max_size % length == 0
        ptr = self.ptr 
        ptr_final = ptr + length
        
        self.self_state[ptr:ptr_final] = ss
        self.self_action[ptr:ptr_final] =sa
        self.other_state[ptr:ptr_final] = os
        self.other_action[ptr:ptr_final] = oa
        self.self_logprobs[ptr:ptr_final] = slp
        self.other_logprobs[ptr:ptr_final] = olp
        self.reward[ptr:ptr_final] = r1
        self.reward2[ptr:ptr_final] = r2
        
        self.ptr += length
        self.ptr = self.ptr % self.max_size
        
    def off_policy_dice_objective(self, theta1, value1, theta2, value2, agent_idx=0, use_baseline=True):
        if agent_idx == 0:
            rewards = torch.from_numpy(self.reward)
            values = value1[torch.from_numpy(self.self_state).long()]
        else:
            rewards = torch.from_numpy(self.reward2)
            values = value2[torch.from_numpy(self.other_state).long()]
            
        cum_discount = torch.cumprod(hp.gamma * torch.ones(*rewards.size()), dim=1)/hp.gamma
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        previous_self_logprobs = torch.from_numpy(self.self_logprobs)
        previous_other_logprobs = torch.from_numpy(self.other_logprobs)
        
        batch_states = torch.from_numpy(self.self_state).long()
        probs = torch.sigmoid(theta1)[batch_states]# bs * 100
        probs_all = torch.stack([probs, 1-probs], dim=-1)#bs * 100 * 2
        
        probs_sa = torch.sum(probs_all * torch.eye(2)[torch.from_numpy(self.self_action).long()], dim=-1)
        
        current_self_logprobs = torch.log(probs_sa + 1e-6) 
        
        batch_states = torch.from_numpy(self.other_state).long()
        probs = torch.sigmoid(theta2)[batch_states]
        probs_all = torch.stack([probs, 1-probs], dim=-1)#bs * 100 * 2
        probs_sa = torch.sum(probs_all * torch.eye(2)[torch.from_numpy(self.other_action).long()], dim=-1)
        
        current_other_logprobs =  torch.log(probs_sa + 1e-6)
        
        previous_dependencies = torch.cumsum(previous_self_logprobs + previous_other_logprobs, dim=1)
        current_dependencies = torch.cumsum(current_self_logprobs + current_other_logprobs, dim=1)

        # logprob of each stochastic nodes:
        previous_stochastic_nodes = previous_self_logprobs + previous_other_logprobs 
        current_stochastic_nodes = current_self_logprobs + current_other_logprobs
        
        # dice objective:
        dice_objective = torch.mean(torch.sum(off_policy_magic_box(current_dependencies, previous_dependencies) * discounted_rewards, dim=1))

        if use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - off_policy_magic_box(current_stochastic_nodes, previous_stochastic_nodes)) * discounted_values, dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective # want to minimize -objective
    
buffer = off_policy_Buffer(max_traj_size=hp.buffer_size)
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
        self_logprobs = torch.stack(self.self_logprobs, dim=1)#128*100
        #print(self_logprobs.shape)
        other_logprobs = torch.stack(self.other_logprobs, dim=1)
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(hp.gamma * torch.ones(*rewards.size()), dim=1)/hp.gamma
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)
        #print(111,dependencies.shape)

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

def act(batch_states, theta):
    batch_states = torch.from_numpy(batch_states).long()
    probs = torch.sigmoid(theta)[batch_states]
    m = Bernoulli(1-probs)
    actions = m.sample()
    log_probs_actions = m.log_prob(actions)
    return actions.numpy().astype(int), log_probs_actions.detach().numpy()

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
        a1, lp1, v1 = act_original(s1, theta1, values1)
        a2, lp2, v2 = act_original(s2, theta2, values2)
        (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
        # cumulate scores
        score1 += np.mean(r1)/float(hp.len_rollout)
        score2 += np.mean(r2)/float(hp.len_rollout)
    return (score1, score2)

def fill_replay(theta1, theta2, ipd):#ss, sa, os, oa, slp, olp, r1, r2
    for i in range(int(buffer.max_size/hp.inner_batch_size)):
        run_len_rollout(theta1,theta2, ipd)

def run_len_rollout(theta1, theta2, ipd):
    s1_all = []
    s2_all = []
    lp1_all = []
    lp2_all = []
    a1_all = []
    a2_all = []
    r1_all = []
    r2_all = []
    (s1, s2), _ = ipd.reset()
    for t in range(hp.len_rollout):
        a1, lp1 = act(s1, theta1)
        a2, lp2 = act(s2, theta2)
        s1_all.append(s1)
        s2_all.append(s2)
        a1_all.append(a1)
        a2_all.append(a2)
        lp1_all.append(lp1)
        lp2_all.append(lp2)
        (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
        r1_all.append(r1)
        r2_all.append(r2)
    s1_all = np.stack(s1_all, axis=-1)
    s2_all = np.stack(s2_all, axis=-1)
    r1_all = np.stack(r1_all, axis=-1)
    r2_all = np.stack(r2_all, axis=-1)
    a1_all = np.stack(a1_all, axis=-1)
    a2_all = np.stack(a2_all, axis=-1)
    lp1_all = np.stack(lp1_all, axis=-1)
    lp2_all = np.stack(lp2_all, axis=-1)
    buffer.add(s1_all, a1_all, s2_all, a2_all, lp1_all, lp2_all, r1_all, r2_all)
    
def in_lookahead(theta1, values1, theta2, values2, agent_idx, ipd):
    run_len_rollout(theta1, theta2, ipd)
    objective = buffer.off_policy_dice_objective(theta1, values1, theta2, values2, agent_idx=agent_idx, use_baseline=True)
    return objective  

def act_original(batch_states, theta, values):
    batch_states = torch.from_numpy(batch_states).long()
    probs = torch.sigmoid(theta)[batch_states]
    m = Bernoulli(1-probs)
    actions = m.sample()
    log_probs_actions = m.log_prob(actions)
    return actions.numpy().astype(int), log_probs_actions, values[batch_states]

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

    def in_lookahead(self, other_theta, other_values, ipd):
        (s1, s2), _ = ipd.reset()
        other_memory = Memory()
        for t in range(hp.len_rollout):
            a1, lp1, v1 = act_original(s1, self.theta, self.values)
            a2, lp2, v2 = act_original(s2, other_theta, other_values)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            other_memory.add(lp2, lp1, v2, torch.from_numpy(r2).float())

        other_objective = other_memory.dice_objective(use_baseline=False)
        grad = get_gradient(other_objective, other_theta)
        return grad
    
    def out_lookahead(self, other_theta, other_values, ipd):
        (s1, s2), _ = ipd.reset()
        memory = Memory()
        for t in range(hp.len_rollout):
            a1, lp1, v1 = act_original(s1, self.theta, self.values)
            a2, lp2, v2 = act_original(s2, other_theta, other_values)
            (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
            memory.add(lp1, lp2, v1, torch.from_numpy(r1).float())
        
        # For visualising graidient correlation, we won't use this gradient
        objective = true_objective(self.theta, other_theta)
        grad_outer_exact = torch.autograd.grad(objective, self.theta, retain_graph=True)[0].detach()      

        # update self theta
        objective = memory.dice_objective()
        grad = self.theta_update(objective)
        # update self value:
        v_loss = memory.value_loss()
        self.value_update(v_loss)
        
        return grad, grad_outer_exact
    
    def in_lookahead_exact(self, other_theta):
        other_objective = true_objective(other_theta, self.theta)
        grad = get_gradient(other_objective, other_theta)
        return grad
    
    def out_lookahead_exact(self, other_theta):  
        objective = true_objective(self.theta, other_theta)
        grad = self.theta_update(objective)
        return grad

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
    fill_replay(agent1.theta, agent2.theta, ipd_inner)
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
            if hp.comp_on_policy:
                grad2 = agent1.in_lookahead(theta2_, values2_, ipd_inner)
                grad1 = agent2.in_lookahead(theta1_, values1_, ipd_inner)
            else:
                obj2 = in_lookahead(agent1.theta, values1_, theta2_, values2_, agent_idx=1, ipd=ipd_inner)
                obj1 = in_lookahead(theta1_, values1_, agent2.theta, values2_, agent_idx=0, ipd=ipd_inner)
                grad2 = torch.autograd.grad(obj2, theta2_, create_graph=True)[0]
                grad1 = torch.autograd.grad(obj1, theta1_, create_graph=True)[0]
            theta2_comp = theta2_ - hp.lr_in * grad2
            theta1_comp = theta1_ - hp.lr_in * grad1
            
            if hp.hessian_on_policy:
                grad2 = agent1.in_lookahead(theta2_, values2_, ipd_inner)
                grad1 = agent2.in_lookahead(theta1_, values1_, ipd_inner)
            else:
                obj2 = in_lookahead(agent1.theta, values1_, theta2_, values2_, agent_idx=1, ipd=ipd_inner)
                obj1 = in_lookahead(theta1_, values1_, agent2.theta, values2_, agent_idx=0, ipd=ipd_inner)
                grad2 = torch.autograd.grad(obj2, theta2_, create_graph=True)[0]
                grad1 = torch.autograd.grad(obj1, theta1_, create_graph=True)[0]
                
            theta2_hessian = theta2_ - hp.lr_in * grad2
            theta1_hessian = theta1_ - hp.lr_in * grad1
                               
            theta2_ = theta2_comp.detach() + theta2_hessian - theta2_hessian.detach()
            theta1_ = theta1_comp.detach() + theta1_hessian - theta1_hessian.detach()

        # update own parameters from out_lookahead:
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
