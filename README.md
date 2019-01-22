# Fair GP model based on GPyTorch

## Requirements

* Python >= 3.6
* PyTorch >= 1.0
* GPyTorch

## How to use
To quickly run with toy data (no fairness), do

```
python3 run.py --data toy_data_1d --inf Variational --likelihood GaussianLikelihood \
               --batch_size 50 --epochs 400 --plot simple_1d
```

To run with the supplied ProPublica data, do

```
python3 run.py --dataset_path ./propublica-recidivism_race_0.npz --batch_size 10000 \
               --epochs 80 --num_inducing 500 --lr 0.05 --length_scale 1.1
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

