#!/bin/bash
lr=3
step_num=2
for traj_num in 2 4 8 16 32 64 128 256
do
    nohup python3 decompose_comp_theory.py --lr $lr --step_num $step_num --inner_est maml --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results_comp_sample/maml/lee --same_trials 10 &
    nohup python3 decompose_comp_theory.py --lr $lr --step_num $step_num --inner_est dice --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results_comp_sample/dice/lee --same_trials 10 &

done
