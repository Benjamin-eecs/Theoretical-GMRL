# Theoretical-GMRL
This repo is the official implementation for **A Theoretical Understanding of Gradient Bias in Meta-Reinforcement Learning (NeurIPS 2022).** This code is developed based on the source code of [TorchOpt](https://github.com/metaopt/TorchOpt) and [OpTree](https://github.com/metaopt/optree).

```Bibtex
@inproceedings{liu2022a,
  title={A Theoretical Understanding of Gradient Bias in Meta-Reinforcement Learning},
  author={Bo Liu and Xidong Feng and Jie Ren and Luo Mai and Rui Zhu and Haifeng Zhang and Jun Wang and Yaodong Yang},
  booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=p9zeOtKQXKs}
}
```

## Requirements

```bash
conda create --name GMRL python=3.7.11
pip install -r requirements.txt
```

## Usages

### Tabular MDP

Experiment on tabular MDP using MAML-RL and LIRPG.

1. Go into directory:

   For MAML-RL,

   ```bash
   cd tabular/maml
   ```

   For LIRPG,

   ```bash
   cd tabular/lirpg
   ```

2. Start training:

   For MAML-RL meta-gradient decomposition and correlation ablation study,

   ```bash
   sh run_decompose.sh

   sh run_lr.sh 

   sh run_step_ablation.sh 
   ```

   For MAML-RL compositional bias ablation study,

   ```bash
   sh run_comp_lr.sh

   sh run_comp_sample.sh

   sh run_comp_step_num.sh
   ```

   For MAML-RL multi-step hessian bias ablation study,

   ```bash
   sh run_hessian_bias.sh
   ```

   For LIRPG ablation study,

   ```bash
   sh run_step.sh
   ```

   

### Iterated Prisoner's Dilemma (IPD)

Experiment on Iterated Prisoner's Dilemma (IPD) using LOLA-DiCE.

1. Go into directory:

    ```bash
    cd lola
    ```

2. Start training:

    For original LOLA-DICE,

    ```bash
    python3 lola_dice_original.py --logdir ./results/inner_128_outer128_baseline

    python3 lola_dice_original.py --inner_exact --logdir ./results/inner_exact_outer128_baseline

    python3 lola_dice_original.py --inner_batch_size 1024 --logdir ./results/inner1024_outer128_baseline

    python3 lola_dice_original.py --inner_exact --outer_exact --logdir ./results/inner_exact_outer_exact
    ```

    For LOLA-DICE ablation study,
    
    ```bash
    python3 lola_dice_ablation.py --hessian_batch_size 1024 --logdir ./result_ablation/comp_128_hessian_1024

    python3 lola_dice_ablation.py --hessian_exact --logdir --logdir ./result_ablation/comp_128_hessian_exact

    python3 lola_dice_ablation.py --comp_exact --logdir ./result_ablation/comp_exact_hessian_128

    python3 lola_dice_ablation.py --comp_batch_size 1024 --logdir ./result_ablation/comp1024_hessian_128
    ```

    For LOLA-DICE off-policy and ablation study,
    
    ```bash
    python3 lola_dice_off_policy.py --logdir ./result_offpolicy/off_policy

    python3 lola_dice_off_policy_ablation.py --comp_on_policy --logdir ./result_offpolicy/off_comp_on_hessian

    python3 lola_dice_off_policy_ablation.py --hessian_on_policy --logdir ./result_offpolicy/on_comp_off_hessian
    ```


### Atari Games

Experiment on eight atari games using MGRL. 

1. Go into directory:

   ```
   cd mgrl
   ```

2. Start training:

    For running baseline A2C,

    ```bash
    python3 main_baseline.py --env-name "QbertNoFrameskip-v4" --algo a2c --use-gae --log-interval 100 --num-steps 5 --num-processes 64 --lr 7e-4 --entropy-coef 0.01 --value-loss-coef 0.5 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 40000000 --log-dir ./baseline/ --seed 0 --use-linear-lr-decay
    ```

    For running 3-step MGRL,
    
    ```bash
    python3 main_meta_condition_kl.py --env-name "QbertNoFrameskip-v4" --algo a2c_meta --use-gae --log-interval 100 --num-steps 5 --num-processes 64 --lr 7e-4 --entropy-coef 0.01 --value-loss-coef 0.5 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 40000000 --log-dir ./meta/ --use-linear-lr-decay --comment all4_sigmoid_v --meta-lr 1e-3 --meta-update 3 --outer_kl_coef 0.0 --outer_entropy_coef 0.0 --outer_critic_coef 0.0 --seed 0
    ```

    For running 3-step MGRL-LVC,
    
    ```bash
    python3 main_meta_condition_kl.py --env-name "QbertNoFrameskip-v4" --algo a2c_meta --use-gae --log-interval 100 --num-steps 5 --num-processes 64 --lr 7e-4 --entropy-coef 0.01 --value-loss-coef 0.5 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 40000000 --log-dir ./meta/ --use-linear-lr-decay --comment all4_sigmoid_v_lvc --meta-lr 1e-3 --meta-update 3 --outer_kl_coef 0.0 --outer_entropy_coef 0.0 --outer_critic_coef 0.0 --seed 0 --lvc
    ```



## Acknowledgements

- [Pytorch implementation of LOLA](https://github.com/alexis-jacq/LOLA_DiCE)
- [Pytorch implementation of A2C](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
- [Tabular MDP](https://github.com/robintyh1/neurips2021-meta-gradient-offpolicy-evaluation)

## Licence

The MIT License
