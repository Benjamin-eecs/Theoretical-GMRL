import numpy as np
import torch
import torch.nn.functional as F

def _safe_ratio(pi, mu):
	return pi / (mu + 1e-8)

def policy_evaluation(P, R, discount, policy):
	"""
	Policy evaluation solver. Compute the exact values for a target policy.
	Args:
		P: transition matrix
		R: reward vector
		discount: discount factor
		policy: target policy
	Returns:
		Exact value function (vf) and Q-function (qf)
	"""
	nstates = P.shape[-1]
	ppi = torch.einsum('ast,sa->st', P, policy)
	rpi = torch.einsum('sa,sa->s', R, policy)
	vf = torch.linalg.solve(torch.eye(nstates) - discount*ppi, rpi)
	qf = R + discount*torch.einsum('ast,t->sa', P, vf)
	return vf, qf

def get_PR(mdp):
	"""
	Extract transition matrix P and reward vector R from a mdp object
	Args:
		mdp: the MDP object
	Returns:
		The matrix P and vector R
	"""
	ns, na = mdp.ns, mdp.na
	P = np.zeros([na, ns, ns])
	R = np.zeros([ns, na])
	for i in range(na):
		for j in range(ns):
			P[i, j] = mdp.P[j * na + i]
	R = np.reshape(mdp.R_matrix, [ns, na])
	return P, R

def oracle_value(params, P, R, gamma):
	"""
	Compute the exact values for a target policy, at the initial state.
	The value is a differentiable function of the target policy parameters.
	Args:
		params: target policy parameters
		P: transition matrix
		R: reward vector
		gamma discount factor
	Returns:
		Exact value function at the initial state
	"""
	pi = F.softmax(params, -1)
	vf, _ = policy_evaluation(P, R, gamma, pi)
	return vf[0]

def Vtrace_evaluation(pi, mu, T, gamma, V, trajs, rho, c):
	"""
	This evaluation subroutine is based on V-trace (Espeholt et al, 2018).
	Args:
		pi: target policy
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		Average Vtrace value estimates at the initial state of the trajectories
	"""
	evaluations = 0
	num_simulations = len(trajs)
	for i in range(num_simulations):
		traj = trajs[i]
		states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
		v_estimate = V[states[-1]]
		for s,a,r,s_next in zip(states[:-1][::-1], actions[::-1], rewards[::-1], states[1:][::-1]):
			ratio = _safe_ratio(pi[s, a], mu[s, a])
			rho_bar = ratio
			c_bar = ratio
			v_estimate = V[s] + rho_bar * (r + gamma * V[s_next]- V[s]) + gamma * c_bar * (v_estimate - V[s_next])
		evaluations = evaluations + (v_estimate)/num_simulations
	return evaluations

def Importancesampling_evaluation(pi, mu, T, gamma, V, trajs, rho, c):
	"""
	This evaluation subroutine is based on step-wise importance sampling  (Precup et al, 2001)
	and doubly-robust estimate (Jiang et al, 2016). 
	It subsumes DiCE (Foerster et al, 2018) and its variants as special cases.
	Args:
		pi: target policy
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		Average importance sampling estimates at the initial state of the trajectories
	"""
	evaluations = 0
	num_simulations = len(trajs)
	for i in range(num_simulations):
		traj = trajs[i]
		states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
		all_estimates = []
		v_estimate = V[states[-1]]
		for s,a,r,s_next in zip(states[:-1][::-1], actions[::-1], rewards[::-1], states[1:][::-1]):
			ratio = _safe_ratio(pi[s, a], mu[s, a])
			rho_bar = ratio
			c_bar = ratio
			v_estimate = V[s] + 1.0 * (r + gamma * V[s_next]- V[s]) + gamma * 1.0 * (v_estimate - V[s_next])
			new_estimate = v_estimate - V[s]
			all_estimates.append(new_estimate)
		all_estimates = all_estimates[::-1]
		init_estimate = 0.0
		product_IS = 1.0
		old_product_IS = 1.0
		for step,estimate in enumerate(all_estimates):
			s, a = states[step], actions[step]
			ratio = _safe_ratio(pi[s, a], mu[s, a])
			product_IS =  ratio * product_IS
			init_estimate = init_estimate + gamma**step * estimate * (product_IS - old_product_IS)
			old_product_IS = product_IS
		evaluations = evaluations + init_estimate/num_simulations#.append(init_estimate)
	return evaluations

def MAML_evaluation(pi, mu, T, gamma, V, trajs, rho, c, intrinsic_reward=None):
	"""
	This evaluation subroutine is based on original MAML biased estimation
	Args:
		pi: target policy
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		Average importance sampling estimates at the initial state of the trajectories
	"""
	evaluations = 0
	num_simulations = len(trajs)
	for i in range(num_simulations):
		traj = trajs[i]
		states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
		all_estimates = []
		v_estimate = V[states[-1]]
		for s,a,r,s_next in zip(states[:-1][::-1], actions[::-1], rewards[::-1], states[1:][::-1]):
			ratio = _safe_ratio(pi[s, a], mu[s, a])
			rho_bar = ratio
			c_bar = ratio
			v_estimate = V[s] + 1.0 * (r + gamma * V[s_next]- V[s]) + gamma * 1.0 * (v_estimate - V[s_next])
			new_estimate = v_estimate - V[s]
			all_estimates.append(new_estimate)
		all_estimates = all_estimates[::-1]
		init_estimate = 0.0
		for step,estimate in enumerate(all_estimates):
			s, a = states[step], actions[step]
			init_estimate = init_estimate + gamma**step * estimate * torch.log(pi[s, a]+1e-8)
		evaluations = evaluations + init_estimate/num_simulations
	return evaluations

def Firstorder_evaluation(pi, mu, T, gamma, V, trajs, rho, c):
	"""
	This evaluation subroutine is based on first-order Taylor expansion of value function  (Kakade et al, 2002; Tang et al, 2020).
	First-order Taylor expansion is commonly used in policy optimization algorithms such as TRPO and PPO.
	Args:
		pi: target policy
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		Average first-order vaue estimates at the initial state of the trajectories
	"""
	evaluations = 0
	num_simulations = len(trajs)
	for i in range(num_simulations):
		traj = trajs[i]
		states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
		all_estimates = []
		v_estimate = V[states[-1]]
		for s,a,r,s_next in zip(states[:-1][::-1], actions[::-1], rewards[::-1], states[1:][::-1]):
			ratio = _safe_ratio(pi[s, a], mu[s, a])
			rho_bar = ratio
			c_bar = ratio
			v_estimate = V[s] + 1.0 * (r + gamma * V[s_next]- V[s]) + gamma * 1.0 * (v_estimate - V[s_next])
			new_estimate = (v_estimate - V[s]) * (ratio - 1.0)
			all_estimates.append(new_estimate)
		all_estimates = all_estimates[::-1]
		init_estimate = 0.0
		for step,estimate in enumerate(all_estimates):
			init_estimate = init_estimate + gamma**step * estimate
		evaluations = evaluations + init_estimate/num_simulations
	return evaluations

def evaluations_vtrace(params, mu, T, gamma, V, trajs, rho, c):
	"""
	Compute evaluation output as a differentiable function of target policy parameter
	for Vtrace evaluation
	Args:
		params: target policy parameters
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		estimates as a differentiable function of parameters
	"""
	pi = F.softmax(params, dim=-1)
	evals = Vtrace_evaluation(pi, mu, T, gamma, V, trajs, rho, c)
	return evals

def evaluations_importancesampling(params, mu, T, gamma, V, trajs, rho, c):
	"""
	Compute evaluation output as a differentiable function of target policy parameter
	for importance sampling evaluation
	Args:
		params: target policy parameters
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		estimates as a differentiable function of parameters
	"""
	pi = F.softmax(params, dim=-1)
	evals = Importancesampling_evaluation(pi, mu, T, gamma, V, trajs, rho, c)
	return evals

def evaluations_MAML(params, mu, T, gamma, V, trajs, rho, c):
	"""
	Compute evaluation output as a differentiable function of target policy parameter
	for importance sampling evaluation
	Args:
		params: target policy parameters
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		estimates as a differentiable function of parameters
	"""
	pi = F.softmax(params, dim=-1)
	evals = MAML_evaluation(pi, mu, T, gamma, V, trajs, rho, c)
	return evals

def evaluations_firstorder(params, mu, T, gamma, V, trajs, rho, c):
	"""
	Compute evaluation output as a differentiable function of target policy parameter
	for first-order evaluation
	Args:
		params: target policy parameters
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		estimates as a differentiable function of parameters
	"""
	pi = F.softmax(params, dim=-1)
	evals = Firstorder_evaluation(pi, mu, T, gamma, V, trajs, rho, c)
	return evals

def evaluations_firstorder_mg_gae(params, mu, T, gamma, V, trajs, rho, c, gae_lambda):
	"""
	Compute evaluation output as a differentiable function of target policy parameter
	for importance sampling evaluation
	Args:
		params: target policy parameters
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		estimates as a differentiable function of parameters
	"""
	pi = F.softmax(params, dim=-1)
	evals = First_order_evaluation_gae_value(pi, mu, T, gamma, V, trajs, rho, c, gae_lambda)
	return evals

def evaluations_MAML_mg_gae(params, mu, T, gamma, V, trajs, rho, c, gae_lambda):
	"""
	Compute evaluation output as a differentiable function of target policy parameter
	for importance sampling evaluation
	Args:
		params: target policy parameters
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		estimates as a differentiable function of parameters
	"""
	pi = F.softmax(params, dim=-1)
	evals = MAML_evaluation_gae_value(pi, mu, T, gamma, V, trajs, rho, c, gae_lambda)
	return evals


def get_gae(traj, V, gamma, gae_lambda):
    # GAE
	states, actions, rewards, ir = traj['states'], traj['actions'], traj['rewards'], traj['ir']
	all_estimates = []
	new_estimate = 0
	for s,a,r,s_next,inr in zip(states[:-1][::-1], actions[::-1], rewards[::-1], states[1:][::-1], ir[::-1]):
		td_error = r + inr + gamma * V[s_next]- V[s]
		new_estimate = gamma * gae_lambda * new_estimate + td_error
		all_estimates.append(new_estimate)
	all_estimates = all_estimates[::-1]
	return all_estimates
    
def MAML_evaluation_gae_value(pi, mu, T, gamma, V, trajs, rho, c, gae_lambda):
	"""
	This evaluation subroutine is based on original MAML biased estimation
	Args:
		pi: target policy
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		Average importance sampling estimates at the initial state of the trajectories
	"""
	evaluations = 0
	num_simulations = len(trajs)
	for i in range(num_simulations):
		traj = trajs[i]
		states, actions, rewards, ir = traj['states'], traj['actions'], traj['rewards'], traj['ir']
		all_estimates = get_gae(traj, V, gamma, gae_lambda)
		init_estimate = 0.0
		for step,estimate in enumerate(all_estimates):
			s, a = states[step], actions[step]
			init_estimate = init_estimate + gamma**step * estimate * torch.log(pi[s, a]+1e-8)
		evaluations = evaluations + init_estimate/num_simulations
	return evaluations

def First_order_evaluation_gae_value(pi, mu, T, gamma, V, trajs, rho, c, gae_lambda):
	"""
	This evaluation subroutine is based on original MAML biased estimation
	Args:
		pi: target policy
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		Average importance sampling estimates at the initial state of the trajectories
	"""
	evaluations = 0
	num_simulations = len(trajs)
	for i in range(num_simulations):
		traj = trajs[i]
		states, actions, rewards, ir = traj['states'], traj['actions'], traj['rewards'], traj['ir']
		all_estimates = get_gae(traj, V, gamma, gae_lambda)
		init_estimate = 0.0
		for step,estimate in enumerate(all_estimates):
			s, a = states[step], actions[step]
			ratio = _safe_ratio(pi[s, a], mu[s, a])
			init_estimate = init_estimate + gamma**step * estimate * ratio
		evaluations = evaluations + init_estimate/num_simulations
	return evaluations