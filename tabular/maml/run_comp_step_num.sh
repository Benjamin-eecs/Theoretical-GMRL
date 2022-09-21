#!/bin/bash
traj_num=32
init=3
inc=0.5
count=1
for step_num in 1 2 3 4 5 6
do  
    for (( i=0; i < $count; i++ ))
    do
        lr=$(echo "$init + ( $inc * $i )" | bc)
        nohup python3 decompose_comp_theory.py --lr $lr --step_num $step_num --inner_est maml --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results_comp_step_num/maml/lee --same_trials 10 &
        nohup python3 decompose_comp_theory.py --lr $lr --step_num $step_num --inner_est lvc --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results_comp_step_num/lvc/lee --same_trials 10 &
        nohup python3 decompose_comp_theory.py --lr $lr --step_num $step_num --inner_est loaded --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results_comp_step_num/loaded/lee --same_trials 10 &
        nohup python3 decompose_comp_theory.py --lr $lr --step_num $step_num --inner_est dice --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results_comp_step_num/dice/lee --same_trials 10 &
    done
done
