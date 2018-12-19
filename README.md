# Fair GP model based on GPyTorch

Contents of the files:

* `dataset.py`: loads data from a numpy file and does some pre-processing
* `fair_likelihood.py`: defines a likelihood function that can enforce fairness
* `flags.py`: defines the commandline flags
* `gp_model.py`: defines the structure of the GP model: the variational distribution and the covariance
* `metrics.py`: several functions that compute metrics for the predictions of the model
* `run.py`: the main training loop
* `utils.py`: useful functions

To run, do:

```
python run.py
```
