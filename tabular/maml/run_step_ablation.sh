#!/bin/bash
traj_num=32
for step_num in {1,2,3,4,5,6}
do
    
    nohup python3 decompose.py --inner_est exact --hessian_est exact --outer_est maml --outer_traj $traj_num --logdir ./results_step_ablation/maml/eel --step_num $step_num &
    nohup python3 decompose.py --inner_est maml --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results_step_ablation/maml/lee --step_num $step_num &
    nohup python3 decompose.py --inner_est exact --hessian_est maml --outer_est exact --hessian_traj $traj_num --logdir ./results_step_ablation/maml/ele --step_num $step_num &
    nohup python3 decompose.py --inner_est exact --hessian_est maml --outer_est maml --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results_step_ablation/maml/ell --step_num $step_num &
    nohup python3 decompose.py --inner_est maml --hessian_est maml --outer_est exact --hessian_traj $traj_num --inner_traj $traj_num --logdir ./results_step_ablation/maml/lle --step_num $step_num &
    nohup python3 decompose.py --inner_est maml --hessian_est exact --outer_est maml --inner_traj $traj_num --outer_traj $traj_num --logdir ./results_step_ablation/maml/lel --step_num $step_num &
    nohup python3 decompose.py --inner_est maml --hessian_est maml --outer_est maml --inner_traj $traj_num --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results_step_ablation/maml/lll --step_num $step_num &
    
done
