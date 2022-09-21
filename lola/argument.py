import argparse
        
def get_args():
    parser = argparse.ArgumentParser(description='LOLA-DICE')
    parser.add_argument(
        '--lr_out',
        type=float,
        default=0.1,
        help='outer learning rate')
    parser.add_argument(
        '--lr_in',
        type=float,
        default=0.3,
        help='inner learning rate')
    parser.add_argument(
        '--lr_v',
        type=float,
        default=0.1,
        help='inner learning rate')
    parser.add_argument(
        '--gamma',
        type=int,
        default=0.96,
        help='dicsount factor')
    parser.add_argument(
        '--n_update',
        type=int,
        default=500,
        help='amount of meta update')
    parser.add_argument(
        '--len_rollout',
        type=int,
        default=100,
        help='length of rollout')
    parser.add_argument(
        '--inner_batch_size',
        type=int,
        default=128,
        help='num of trajectories used for averaging estimates for inner gradient')
    parser.add_argument(
        '--outer_batch_size',
        type=int,
        default=128,
        help='num of trajectories used for averaging estimates for outer gradient')
    parser.add_argument(
        '--inner_step',
        type=int,
        default=1,
        help='num of trajectories used for averaging estimates for outer gradient')
    parser.add_argument(
        '--inner_exact',
        action='store_true', 
        help='num of trajectories used for averaging estimates for outer gradient')
    parser.add_argument(
        '--outer_exact',
        action='store_true', 
        help='num of trajectories used for averaging estimates for outer gradient')
    parser.add_argument(
        '--no_inner_baseline',
        action='store_true', 
        help='num of trajectories used for averaging estimates for outer gradient')
    parser.add_argument(
        '--no_outer_baseline',
        action='store_true', 
        help='num of trajectories used for averaging estimates for outer gradient')
    
    # For ablation study
    parser.add_argument(
        '--comp_batch_size',
        type=int,
        default=128,
        help='num of trajectories used for averaging estimates for outer gradient')
    parser.add_argument(
        '--hessian_batch_size',
        type=int,
        default=128,
        help='num of trajectories used for averaging estimates for outer gradient')
    parser.add_argument(
        '--comp_exact',
        action='store_true', 
        help='use exact inner-loop pg')
    parser.add_argument(
        '--hessian_exact',
        action='store_true', 
        help='use exact hessian gradient')
    
    parser.add_argument(
        '--logdir',
        default='./results',
        help='seed')
    
    # For off-policy DiCE
    parser.add_argument(
        '--buffer_size',
        type=int,
        default=1024,
        help='replay buffer size')
    parser.add_argument(
        '--comp_on_policy',
        action='store_true', 
        help='on-policy first order')
    parser.add_argument(
        '--hessian_on_policy',
        action='store_true', 
        help='on-policy second order')
  
    args = parser.parse_args()

    return args