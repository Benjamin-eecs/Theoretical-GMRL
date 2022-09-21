import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--lr',
        type=float,
        default=10.0,
        help='learning rate')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.8,
        help='gamma')
    parser.add_argument(
        '--T',
        type=int,
        default=20,
        help='trajectory length')
    parser.add_argument(
        '--step_num',
        type=int,
        default=1,
        help='step_num')
    parser.add_argument(
        '--outer_traj',
        type=int,
        default=5,
        help='num of trajectories used for averaging estimates for outer gradient')
    parser.add_argument(
        '--inner_traj',
        type=int,
        default=5,
        help='num of trajectories used for averaging estimates for inner gradient')
    parser.add_argument(
        '--hessian_traj',
        type=int,
        default=5,
        help='num of trajectories used for averaging estimates for hessian')
    parser.add_argument(
        '--independent_trials',
        type=int,
        default=10,
        help='num of independent trials on envs')
    parser.add_argument(
        '--same_trials',
        type=int,
        default=20,
        help='num of independent trials on the same point')
    parser.add_argument(
        '--ns',
        type=int,
        default=20,
        help='dim of state')
    parser.add_argument(
        '--na',
        type=int,
        default=5,
        help='dim of action')
    parser.add_argument(
        '--density',
        type=float,
        default=0.001,
        help='parameter for generating MDP')
    parser.add_argument(
        '--outer_est',
        default='exact')
    parser.add_argument(
        '--inner_est',
        default='exact')
    parser.add_argument(
        '--hessian_est',
        default='exact')
    parser.add_argument(
        '--noise',
        type=float,
        default=1.0)
    
    parser.add_argument(
        '--hessian_error_coef',
        type=float,
        default=0.0)
    
    parser.add_argument(
        '--logdir',
        default='./results',
        help='seed')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='seed')
  
    args = parser.parse_args()

    assert args.outer_est in ['exact', 'dice', 'lvc', 'loaded', 'maml']
    assert args.inner_est in ['exact', 'dice', 'lvc', 'loaded', 'maml']
    assert args.hessian_est in ['exact', 'dice', 'lvc', 'loaded', 'maml']

    return args