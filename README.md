# Fair GP model based on GPyTorch

## Requirements

* Python >= 3.6
* PyTorch >= 1.0
* GPyTorch

## How to use
To run with default parameters, do

```
python3 run.py
```

However, this will fail unless there is a numpy file with training data in the same directory.

To run with toy data, do

```
python3 run.py --data toy_data_1d --inf Variational --likelihood GaussianLikelihood \
               --batch_size 50 --epochs 400 --plot simple_1d
```

To see all possible flags, do

```
python3 run.py --help
```

## Structure of the code

* `run.py` simply calls `main_loop()` in `fairgp/training.py`
* `tests` contains tests that can be run with `python3 -m pytest tests/`
* `fairgp` contains the main code
  - `dataset.py`: loads data from a numpy file and does some pre-processing;
    also defines some toy datasets
  - `fair_likelihood.py`: defines a likelihood function that can enforce fairness
  - `flags.py`: defines the commandline flags
  - `gp_model.py`: defines the structure of the GP model:
    the variational distribution and the covariance
  - `training.py`: defines the main training loop with evaluation and checkpointing
  - `utils/`: useful functions

