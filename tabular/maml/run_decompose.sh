#!/bin/bash
nohup python3 decompose.py --inner_est exact --hessian_est exact --outer_est exact --logdir ./results/exact_all &
for traj_num in {2,4,8,16,32,64,128,256}
do
    nohup python3 decompose.py --inner_est exact --hessian_est exact --outer_est loaded --outer_traj $traj_num --logdir ./results/loaded/eel &
    nohup python3 decompose.py --inner_est loaded --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results/loaded/lee &
    nohup python3 decompose.py --inner_est exact --hessian_est loaded --outer_est exact --hessian_traj $traj_num --logdir ./results/loaded/ele &
    nohup python3 decompose.py --inner_est exact --hessian_est loaded --outer_est loaded --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results/loaded/ell &
    nohup python3 decompose.py --inner_est loaded --hessian_est loaded --outer_est exact --hessian_traj $traj_num --inner_traj $traj_num --logdir ./results/loaded/lle &
    nohup python3 decompose.py --inner_est loaded --hessian_est exact --outer_est loaded --inner_traj $traj_num --outer_traj $traj_num --logdir ./results/loaded/lel &
    nohup python3 decompose.py --inner_est loaded --hessian_est loaded --outer_est loaded --inner_traj $traj_num --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results/loaded/lll &

    nohup python3 decompose.py --inner_est exact --hessian_est exact --outer_est dice --outer_traj $traj_num --logdir ./results/dice/eel &
    nohup python3 decompose.py --inner_est dice --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results/dice/lee &
    nohup python3 decompose.py --inner_est exact --hessian_est dice --outer_est exact --hessian_traj $traj_num --logdir ./results/dice/ele &
    nohup python3 decompose.py --inner_est exact --hessian_est dice --outer_est dice --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results/dice/ell &
    nohup python3 decompose.py --inner_est dice --hessian_est dice --outer_est exact --hessian_traj $traj_num --inner_traj $traj_num --logdir ./results/dice/lle &
    nohup python3 decompose.py --inner_est dice --hessian_est exact --outer_est dice --inner_traj $traj_num --outer_traj $traj_num --logdir ./results/dice/lel &
    nohup python3 decompose.py --inner_est dice --hessian_est dice --outer_est dice --inner_traj $traj_num --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results/dice/lll &
    
    nohup python3 decompose.py --inner_est exact --hessian_est exact --outer_est lvc --outer_traj $traj_num --logdir ./results/lvc/eel &
    nohup python3 decompose.py --inner_est lvc --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results/lvc/lee &
    nohup python3 decompose.py --inner_est exact --hessian_est lvc --outer_est exact --hessian_traj $traj_num --logdir ./results/lvc/ele &
    nohup python3 decompose.py --inner_est exact --hessian_est lvc --outer_est lvc --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results/lvc/ell &
    nohup python3 decompose.py --inner_est lvc --hessian_est lvc --outer_est exact --hessian_traj $traj_num --inner_traj $traj_num --logdir ./results/lvc/lle &
    nohup python3 decompose.py --inner_est lvc --hessian_est exact --outer_est lvc --inner_traj $traj_num --outer_traj $traj_num --logdir ./results/lvc/lel &
    nohup python3 decompose.py --inner_est lvc --hessian_est lvc --outer_est lvc --inner_traj $traj_num --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results/lvc/lll &
    
    nohup python3 decompose.py --inner_est exact --hessian_est exact --outer_est maml --outer_traj $traj_num --logdir ./results/maml/eel &
    nohup python3 decompose.py --inner_est maml --hessian_est exact --outer_est exact --inner_traj $traj_num --logdir ./results/maml/lee &
    nohup python3 decompose.py --inner_est exact --hessian_est maml --outer_est exact --hessian_traj $traj_num --logdir ./results/maml/ele &
    nohup python3 decompose.py --inner_est exact --hessian_est maml --outer_est maml --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results/maml/ell &
    nohup python3 decompose.py --inner_est maml --hessian_est maml --outer_est exact --hessian_traj $traj_num --inner_traj $traj_num --logdir ./results/maml/lle &
    nohup python3 decompose.py --inner_est maml --hessian_est exact --outer_est maml --inner_traj $traj_num --outer_traj $traj_num --logdir ./results/maml/lel &
    nohup python3 decompose.py --inner_est maml --hessian_est maml --outer_est maml --inner_traj $traj_num --hessian_traj $traj_num --outer_traj $traj_num --logdir ./results/maml/lll &
done
