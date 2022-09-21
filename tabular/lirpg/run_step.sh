#!/bin/bash
traj_num=64
gae_lambda=0.9
for traj_num in {16,32,64,128,256}
do
    nohup python3 decompose_mg.py --inner_est exact --hessian_est maml --outer_est exact --hessian_traj $traj_num --logdir ./results_gae_ratio8/maml/ele --step_num 1 --gae_lambda $gae_lambda --gamma 0.8 &  
    nohup python3 decompose_mg.py --inner_est exact --hessian_est maml --outer_est exact --hessian_traj $traj_num --logdir ./results_gae_ratio8/maml/ele --step_num 2 --gae_lambda $gae_lambda --gamma 0.8 &  
    nohup python3 decompose_mg.py --inner_est exact --hessian_est maml --outer_est exact --hessian_traj $traj_num --logdir ./results_gae_ratio8/maml/ele --step_num 3 --gae_lambda $gae_lambda --gamma 0.8 &  
    nohup python3 decompose_mg.py --inner_est exact --hessian_est maml --outer_est exact --hessian_traj $traj_num --logdir ./results_gae_ratio8/maml/ele --step_num 4 --gae_lambda $gae_lambda --gamma 0.8 &  
    nohup python3 decompose_mg.py --inner_est exact --hessian_est maml --outer_est exact --hessian_traj $traj_num --logdir ./results_gae_ratio8/maml/ele --step_num 5 --gae_lambda $gae_lambda --gamma 0.8 &  
    
    nohup python3 decompose_mg.py --inner_est exact --hessian_est lvc --outer_est exact --hessian_traj $traj_num --logdir ./results_gae_ratio8/lvc/ele --step_num 1 --gae_lambda $gae_lambda --gamma 0.8 &  
    nohup python3 decompose_mg.py --inner_est exact --hessian_est lvc --outer_est exact --hessian_traj $traj_num --logdir ./results_gae_ratio8/lvc/ele --step_num 2 --gae_lambda $gae_lambda --gamma 0.8 &  
    nohup python3 decompose_mg.py --inner_est exact --hessian_est lvc --outer_est exact --hessian_traj $traj_num --logdir ./results_gae_ratio8/lvc/ele --step_num 3 --gae_lambda $gae_lambda --gamma 0.8 &  
    nohup python3 decompose_mg.py --inner_est exact --hessian_est lvc --outer_est exact --hessian_traj $traj_num --logdir ./results_gae_ratio8/lvc/ele --step_num 4 --gae_lambda $gae_lambda --gamma 0.8 &  
    nohup python3 decompose_mg.py --inner_est exact --hessian_est lvc --outer_est exact --hessian_traj $traj_num --logdir ./results_gae_ratio8/lvc/ele --step_num 5 --gae_lambda $gae_lambda --gamma 0.8 & 
done
