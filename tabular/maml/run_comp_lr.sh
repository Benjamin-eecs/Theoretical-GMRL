#!/bin/bash
traj_num=32
step_num=6
for lr in 1 3 5 7 9 11 13
do
    nohup python3 decompose_comp_theory.py --lr $lr --step_num $step_num --inner_est maml --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results_comp_lr/maml/lee --same_trials 10 &
    nohup python3 decompose_comp_theory.py --lr $lr --step_num $step_num --inner_est dice --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results_comp_lr/dice/lee --same_trials 10 &

done
