#!/bin/bash
traj_num=3
step_num=2
init=3
inc=0.5
count=4
for hec in 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0
do
    for (( i=0; i < $count; i++ ))
    do
        lr=$(echo "$init + ( $inc * $i )" | bc)
        nohup python3 decompose_hess_theory.py --lr $lr --inner_est exact --hessian_est exact --outer_est exact --hessian_traj $traj_num --logdir ./results_hessian_bias/exact/eee --step_num $step_num --hessian_error_coef $hec &
        
    done
done
