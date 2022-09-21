import numpy as np
import os
import tabular_mdp
from evaluation_utils import get_PR, oracle_value, evaluations_vtrace, evaluations_importancesampling, evaluations_firstorder, evaluations_MAML
import torch
import torch.nn.functional as F
from argument import get_args
import json
import time
import hashlib

rho = np.inf  # truncation hyper-parameter -- default to no truncation
c = np.inf  # truncation hyper-parameter -- default to no truncation

class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        if callable(o):
            return {'function': o.__name__}
        return json.JSONEncoder.default(self, o)
    
def generate_trajectories(mdp, mu, num_simulations, T):
    """
    Generate trajectories from the MDP using policy mu
    Args:
    mdp: the mdp object
    mu: the policy to be executed
    num_simulations: num of trajectories
    T: truncated horizon of the trajectory
    Returns:
    A list of trajectories
    """
    trajs = []
    na = mu.shape[1]
    for i in range(num_simulations):
        rsum = []
        states = []
        actions = []
        s = mdp.reset()
        for t in range(T):
            a = np.random.choice(np.arange(na), p=mu[s])
            s_next, r, _, _ = mdp.step(a)
            rsum.append(r)
            actions.append(a)
            states.append(s)
            s = s_next
        states.append(s_next)
        trajs.append({'states': states, 'actions': actions, 'rewards': rsum})
    return trajs


def corr(x, y):
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
    
def get_variance(gradient):
    if isinstance(gradient, list):
        gradient = torch.stack(gradient, dim=0)
    return torch.sum(torch.std(gradient, unbiased=True, dim=0))

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_estimator(estimator):
    if estimator == 'exact':
        return oracle_value
    elif estimator == 'lvc':
        return evaluations_firstorder
    elif estimator == 'loaded':
        return evaluations_importancesampling
    elif estimator == 'dice':
        return evaluations_vtrace
    elif estimator == 'maml':
        return evaluations_MAML
    else:
        return None

def get_value(est, estimator, params_fast, P, R, mdp, T, num_traj, gamma, noise=1.0):
    if est == 'exact':
        value_all = oracle_value(params_fast, P, R, gamma)
    else:
        mu = F.softmax(params_fast, dim=-1).detach().numpy()
        trajs_all = [generate_trajectories(mdp, mu, 1, T) for _ in range(num_traj)]

        V_exact = mdp.evaluate(gamma, F.softmax(params_fast, dim=-1).detach().numpy())['v']
        noise_level = noise
        V_exact += np.random.randn(*np.shape(V_exact)) * noise_level * np.linalg.norm(V_exact) / V_exact.size
        V_bootstrapped = V_exact.copy()

        if est == 'dice':
            V_bootstrapped = V_bootstrapped * 0.0
        value_all = 0
        for i in range(num_traj):
            trajs = trajs_all[i]
            value = estimator(params_fast, F.softmax(params_fast, dim=-1).detach(), T, gamma, V_bootstrapped, trajs, rho, c)
            value_all = value_all + value / num_traj
    return value_all

def get_path_from_args(args):
    """ Returns a unique hash for an argparse object. """
    args_str = str(args)
    path = hashlib.md5(args_str.encode()).hexdigest()
    return path   

def main():
    args = get_args()
    set_seed(args.seed)
    
    log_dir = os.path.expanduser(args.logdir)
    log_dir = log_dir + '/test_%s' % get_path_from_args(args)
    os.makedirs(log_dir)
    json.dump(vars(args), open(log_dir + '/params.json', 'w'), cls=ClassEncoder)
    
    ns = args.ns
    na = args.na
    independent_trials = args.independent_trials
    T = args.T
    same_trials = args.same_trials
    gamma = args.gamma
    density = args.density
    step_num = args.step_num
    lr = args.lr
    outer_traj = args.outer_traj
    inner_traj = args.inner_traj
    hessian_traj = args.hessian_traj
    outer_est = args.outer_est
    inner_est = args.inner_est
    hessian_est = args.hessian_est
    noise = args.noise
    
    # Compute noise corrupted Vtrace values as bootstrapped values
    corr_list = []
    var_list = []
    grad_mse = []
    
    outer_estimator = get_estimator(outer_est)
    inner_estimator = get_estimator(inner_est)
    hessian_estimator = get_estimator(hessian_est)
    
    for i in range(independent_trials):
        mdp = tabular_mdp.TabularMDP(ns, na, r_std=0.0, dirichlet_intensity=density)
        params = torch.zeros([ns, na], requires_grad=True)

        # get P, R from the MDP and compute oracle gradient and Hessian
        P, R = get_PR(mdp)
        P, R = torch.tensor(P).float(), torch.tensor(R).float()

        #get oralce meta gradient
        params_fast = params.clone()
        for step in range(step_num):     
            value = oracle_value(params_fast, P, R, gamma)
            oracle_gradient = torch.autograd.grad(value, params_fast, create_graph=True)[0]
            params_fast = params_fast + lr * oracle_gradient
        value = oracle_value(params_fast, P, R, gamma)
        oracle_meta_gradient = torch.autograd.grad(value, params)[0]
        
        meta_gradient_list = []
        for _ in range(same_trials):
            params_fast = params.clone()
            for step in range(step_num):
                # inner
                value = get_value(est=inner_est, estimator=inner_estimator, params_fast=params_fast, P=P, R=R, mdp=mdp, T=T, num_traj=inner_traj, gamma=gamma, noise=noise)

                prarams_inner = params_fast + lr * torch.autograd.grad(value, params_fast)[0]
                
                # hessian
                value = get_value(est=hessian_est, estimator=hessian_estimator, params_fast=params_fast, P=P, R=R, mdp=mdp, T=T, num_traj=hessian_traj, gamma=gamma, noise=noise)
                
                
                hessian_error_matrix = args.hessian_error_coef * torch.ones_like(params_fast) + 0.3 * torch.randn_like(params_fast) # add some noise to simulate the Hessian estimation
                value = value + torch.mean(params_fast**2 * hessian_error_matrix) # manually add hessian estimation bias
                
                prarams_hessian = params_fast + lr * torch.autograd.grad(value, params_fast, create_graph=True)[0]

                params_fast = prarams_inner.detach() + prarams_hessian - prarams_hessian.detach()
                             
            # outer
            value = get_value(est=outer_est, estimator=outer_estimator, params_fast=params_fast, P=P, R=R, mdp=mdp, T=T, num_traj=outer_traj, gamma=gamma, noise=noise)
            meta_gradient = torch.autograd.grad(value, params)[0]
            meta_gradient_list.append(meta_gradient.detach().numpy().reshape(-1))
        meta_gradient_mean = np.mean(np.stack(meta_gradient_list, axis=0), axis=0)
        # identify the bias of estimator
        bias = np.mean((meta_gradient_mean - oracle_meta_gradient.detach().numpy().reshape(-1))**2)
        #grad_corr = sum(grad_corr)/len(grad_corr)
        grad_mse.append(bias)
    grad_mse = np.array(grad_mse)
    np.save(log_dir+'/grad_mse', grad_mse)

if __name__ == "__main__":
    main()
