import torch
import torch.nn as nn
import a2c_ppo_acktr.algo.meta_optim as optim

from a2c_ppo_acktr.algo.kfac import KFACOptimizer
from torch.distributions.kl import kl_divergence
import copy

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val
        
class A2C_ACKTR_meta():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 device,
                 lr=None,
                 meta_lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 outer_policy_coef=1.0,
                 outer_critic_coef=0.25,
                 outer_entropy_coef=0.01,
                 outer_kl_coef=1.0):
        
        self.outer_policy_coef = outer_policy_coef
        self.outer_critic_coef = outer_critic_coef
        self.outer_entropy_coef = outer_entropy_coef
        self.outer_kl_coef = outer_kl_coef

        self.actor_critic = actor_critic
        self.acktr = acktr
        self.device = device

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.MetaAdam([actor_critic], lr)
        self.optimizer.reset_grad_normalizer(max_norm=max_grad_norm, norm_type=2.0)    # set normalizer
        
        self.meta_gamma = torch.tensor([4.6], requires_grad=True, device=self.device)#sigmoid 4.6 = 0.99
        self.meta_gae = torch.tensor([2.944], requires_grad=True, device=self.device)
        self.meta_policy_coef = torch.tensor([1.0], requires_grad=False, device=self.device)
        self.meta_value_coef = torch.tensor([0.0], requires_grad=True, device=self.device)
        self.meta_entropy_coef = torch.tensor([-4.6], requires_grad=True, device=self.device)
        self.meta_optimizer = optim.Adam([self.meta_gamma, self.meta_gae, self.meta_value_coef, self.meta_entropy_coef], lr=meta_lr)
    
    def meta_update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape).detach().clone(),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size).detach().clone(),
            rollouts.masks[:-1].view(-1, 1).detach().clone(),
            rollouts.actions.view(-1, action_shape).detach().clone())

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        
        assert rollouts.returns.requires_grad == False
        advantages = rollouts.returns.detach() - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()
        
        loss = action_loss
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
     
        return None
        
        
    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape).detach().clone(),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size).detach().clone(),
            rollouts.masks[:-1].view(-1, 1).detach().clone(),
            rollouts.actions.view(-1, action_shape).detach().clone())

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        
        advantages = rollouts.returns - values
        value_loss = advantages.pow(2).mean()
        
        advantages_policy = rollouts.returns - values.detach()

        action_loss = -(advantages_policy * action_log_probs).mean()

        #self.optimizer.zero_grad()
        loss = (value_loss * self.meta_value_coef + action_loss - dist_entropy * torch.sigmoid(self.meta_entropy_coef))

        self.optimizer.step(loss)

        return value_loss.item(), action_loss.item(), dist_entropy.item()
    
    def meta_update_condition(self, rollouts, gamma, prev_agent=None):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        
        if prev_agent is None:
            return_dist = False
        else:
            return_dist = True

        values, action_log_probs, dist_entropy, _, action_dist = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape).detach().clone(),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size).detach().clone(),
            rollouts.masks[:-1].view(-1, 1).detach().clone(),
            rollouts.actions.view(-1, action_shape).detach().clone(),
            gamma=gamma, return_dist=return_dist)

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        
        assert rollouts.returns.requires_grad == False
        advantages = rollouts.returns.detach() - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()
        
        if prev_agent is None:
            kl_div = 0
        else:
            prev_action_dist = prev_agent.get_action_dist(rollouts.obs[:-1].view(-1, *obs_shape).detach().clone(),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size).detach().clone(),
            rollouts.masks[:-1].view(-1, 1).detach().clone(),
            rollouts.actions.view(-1, action_shape).detach().clone(),
            gamma=gamma)
            kl_div = kl_divergence(action_dist, prev_action_dist)
            kl_div = kl_div.mean()
        
        loss = self.outer_policy_coef * action_loss + self.outer_critic_coef * value_loss - self.outer_entropy_coef * dist_entropy + self.outer_kl_coef * kl_div
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

        return action_loss.item(), value_loss.item(), dist_entropy.item(), kl_div.item()
    
    def update_condition(self, rollouts, gamma, lvc=False):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        
        bs = num_steps * num_processes

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape).detach().clone(),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size).detach().clone(),
            rollouts.masks[:-1].view(-1, 1).detach().clone(),
            rollouts.actions.view(-1, action_shape).detach().clone(),
            gamma=gamma, lvc=lvc)

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        
        advantages = rollouts.returns - values
        value_loss = advantages.pow(2).mean()
        
        advantages_policy = rollouts.returns - values.detach()

        action_loss = -(advantages_policy * action_log_probs).mean()

        #self.optimizer.zero_grad()
        loss = (value_loss * torch.sigmoid(self.meta_value_coef) + self.meta_policy_coef * action_loss - dist_entropy * torch.sigmoid(self.meta_entropy_coef))
        
        self.optimizer.step(loss)

        return value_loss.item(), action_loss.item(), dist_entropy.item()
    
    def store_old(self):
        with torch.no_grad():
            self.old_policy = copy.deepcopy(self.actor_critic.state_dict())
            self.old_optimizer = copy.deepcopy(self.optimizer.state_dict())
    
    def reset_policy(self):
        with torch.no_grad():
            self.actor_critic = copy.deepcopy(self.old_policy)
            self.optimizer = copy.deepcopy(self.old_optimizer)
        
