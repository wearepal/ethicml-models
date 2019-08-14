# Fair GP model based on GPyTorch

## Requirements

* Python >= 3.6
* PyTorch == 1.1.0
* GPyTorch == 0.3.4
  (as the code goes deep into the internals of GPyTorch, no other version is guaranteed to work)

## How to use
To quickly run with toy data (no fairness), do

```
python3 run.py --data toy_data_1d --inf Variational --lik GaussianLikelihood \
               --batch_size 50 --epochs 400 --plot simple_1d --metrics RMSE,MAE
```

To run with the supplied ProPublica data, do

```
python3 run.py --dataset_path ./propublica-recidivism_race_0.npz --batch_size 10000 \
               --epochs 80 --num_inducing 500 --lr 0.05 --length_scale 1.1 --lik BaselineLikelihood
```

This will does not enforce any fairness, however.
In order to enforce, for example, Equality of Opportunity,
the correct likelihood function has to be chosen: `TuneTprLikelihood`.
Additionally, the target TPRs have to be specified with the flags
`p_ybary1_s0`, `p_ybary1_s1`, `p_ybary0_s0` and `p_ybary0_s1`.
They correspond to the target TPR for group 0 and group 1
and to the the target TNR for group 0 and group 1.
Finally, the acceptance rate in the training set has to be specified.
For the supplied data, these are 0.3521 for `s=0` and 0.4797 for `s=1`.

Putting everything together, we get

```
python3 run.py --dataset_path ./propublica-recidivism_race_0.npz --batch_size 10000 \
               --epochs 80 --num_inducing 500 --lr 0.05 --length_scale 1.1 --lik TuneTprLikelihood \
               --p_ybary1_s0 1.0 --p_ybary1_s1 0.7 --p_ybary0_s0 1.0 --p_ybary0_s1 0.7 \
               --biased_acceptance1 0.3521 --biased_acceptance2 0.4797
```

To see all possible flags, do

```
python3 run.py --help
```

## Structure of the code

* `run.py` simply calls `main_loop()` in `fairgp/training.py`
* `tests/` contains tests that can be run with `python3 -m pytest tests/`
* `fairgp/` contains the main code
  - `dataset.py`: loads data from a numpy file and does some pre-processing;
    also defines some toy datasets
  - `fair_likelihood.py`: defines a likelihood function that can enforce fairness
  - `flags.py`: defines the commandline flags
  - `gp_model.py`: defines the structure of the GP model:
    the variational distribution and the covariance
  - `training.py`: defines the main training loop with evaluation and checkpointing
  - `utils/`: useful functions

