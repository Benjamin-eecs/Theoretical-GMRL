#!/bin/bash
traj_num=32
step_num=5
for lr in {1,3,5,7,9,11,13}
do
    
    nohup python3 decompose.py --inner_est exact --hessian_est exact --outer_est maml --outer_traj $traj_num --logdir ./results_lr_ablation/maml/eel --step_num $step_num --lr $lr &
    nohup python3 decompose.py --inner_est maml --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results_lr_ablation/maml/lee --step_num $step_num --lr $lr &
    nohup python3 decompose.py --inner_est exact --hessian_est maml --outer_est exact --hessian_traj $traj_num --logdir ./results_lr_ablation/maml/ele --step_num $step_num --lr $lr &
    nohup python3 decompose.py --inner_est exact --hessian_est maml --outer_est maml --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results_lr_ablation/maml/ell --step_num $step_num --lr $lr &
    nohup python3 decompose.py --inner_est maml --hessian_est maml --outer_est exact --hessian_traj $traj_num --inner_traj $traj_num --logdir ./results_lr_ablation/maml/lle --step_num $step_num --lr $lr &
    nohup python3 decompose.py --inner_est maml --hessian_est exact --outer_est maml --inner_traj $traj_num --outer_traj $traj_num --logdir ./results_lr_ablation/maml/lel --step_num $step_num --lr $lr &
    nohup python3 decompose.py --inner_est maml --hessian_est maml --outer_est maml --inner_traj $traj_num --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results_lr_ablation/maml/lll --step_num $step_num --lr $lr &
    
done
