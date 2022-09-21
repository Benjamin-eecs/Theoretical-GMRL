import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from a2c_ppo_acktr import logger
import time
import json

class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        if callable(o):
            return {'function': o.__name__}
        return json.JSONEncoder.default(self, o)

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    idx = int(time.time())
    args = get_args()

    set_seed(args.seed)
    log_dir = os.path.expanduser(args.log_dir)
    log_dir = os.path.join(log_dir, args.env_name) + '/test_%d' % idx
    args.save_dir = log_dir + '/train_models/'
    #eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    #utils.cleanup_log_dir(eval_log_dir)
    
    logger.configure(dir=log_dir, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='none')
    json.dump(vars(args), open(log_dir + '/params.json', 'w'), cls=ClassEncoder)

    torch.set_num_threads(16)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        device=device,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
        
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, device=device)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    epinfobuf = deque(maxlen=64)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    time_step = 0
    value_loss, action_loss, dist_entropy = 0, 0, 0
    for update in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, update, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            epinfos = []
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
        
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            
        epinfobuf.extend(epinfos)
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        
        rollouts.after_update()
    
        if update % args.log_interval == 0 and update > 0:
            nseconds = time.time() - start
            env_step = update * args.num_steps * args.num_processes
            logger.log('** n_timesteps ' + str(env_step) + ' **')
            logger.logkv('Updates', update)
            logger.logkv('n_timesteps', env_step)
            logger.logkv('fps', env_step/nseconds)
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('policy_loss', action_loss)
            logger.logkv('value_loss', value_loss)
            logger.logkv('entropy_loss', dist_entropy)
            logger.dumpkvs()
            
if __name__ == "__main__":
    main()
