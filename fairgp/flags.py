import argparse
import random


def parse_arguments(raw_args=None):
    bool_type = dict(type=str2bool, choices=[True, False])
    float_type = dict(type=float, metavar='<float>')
    int_type = dict(type=int, metavar='<int>')
    path_type = dict(metavar='PATH')
    default = '(default: %(default)s)'
    default_str = '(default: "%(default)s")'
    parser = argparse.ArgumentParser()
    # Training flags
    parser.add_argument('--data', default='sensitive_from_numpy', help=f'Dataset {default_str}',
                        choices=['sensitive_from_numpy', 'toy_data_1d', 'toy_data_1d_multitask'])
    parser.add_argument('--lr', default=0.01,
                        help=f'Learning rate {default}', **float_type)
    parser.add_argument('--metrics', default='binary_accuracy', help='List of metrics to log')
    parser.add_argument('--loo_steps', default=0, **int_type,
                        help=f'Number of steps for optimizing LOO loss; 0 disables {default}')
    parser.add_argument('--nelbo_steps', default=0, **int_type,
                        help='Number of steps for optimizing NELBO loss; 0 means same as loo_steps')
    parser.add_argument('--num_all', default=200, **int_type,
                        help='Suggested total number of examples (datasets don\'t have to use it)')
    parser.add_argument('--num_train', default=50, **int_type,
                        help='Suggested number of train examples (datasets don\'t have to use it)')
    parser.add_argument('--num_inducing', default=1000, **int_type,
                        help='Suggested number of inducing inputs (datasets don\'t have to use it)')
    parser.add_argument('--optimizer', default='Adam',
                        help=f'Optimizer to use for gradient descent {default_str}',
                        choices=['Adam', 'RMSprop', 'SGD', 'LBFGS'])
    parser.add_argument('--model_name', default='local',
                        help=f'Name of model (used for name of checkpoints) {default_str}')
    parser.add_argument('--batch_size', default=1000, **int_type, help='Batch size')
    parser.add_argument('--epochs', default=30, **int_type, help='Number of epochs for training')
    parser.add_argument('--eval_epochs', default=1, **int_type,
                        help=f'Number of epochs between evaluations {default}')
    parser.add_argument('--summary_steps', default=100, **int_type,
                        help=f'How many steps between saving summary {default}')
    parser.add_argument('--chkpt_epochs', default=5, **int_type,
                        help=f'How many epochs between saving checkpoints {default}')
    parser.add_argument('--save_dir', default='', **path_type,
                        help='Directory where the checkpoints and summaries are saved (or \'\')')
    parser.add_argument('--plot', default='',
                        help=f'Which function to use for plotting {default_str}',
                        choices=['', 'simple_1d'])
    parser.add_argument('--logging_steps', default=1, **int_type,
                        help=f'How many steps between logging the loss {default}')
    parser.add_argument('--gpus', default='0',
                        help=f'Which GPUs to use (should normally only be one) {default}')
    parser.add_argument('--preds_path', default='', **path_type,
                        help='Path where the predictions for the test data will be save (or "")')
    parser.add_argument('--lr_drop_steps', default=0, **int_type,
                        help=f'Number of steps before doing a learning rate drop {default}')
    parser.add_argument('--lr_drop_factor', default=0.2, **float_type,
                        help=f'For learning rate drop multiply by this factor {default}')
    parser.add_argument('--manual_seed', **int_type,
                        help='manual seed, if not given resorts to random seed.')
    parser.add_argument('--save_best', default=False, **bool_type,
                        help=f'if True, the best model is saved in a separate file {default}')

    # Gaussian Process model
    parser.add_argument('--inf', default='Variational', help=f'Inference method {default_str}',
                        choices=['Variational', 'Exact', 'ExactMultitask'])
    parser.add_argument('--cov', default='RBFKernel', help=f'Covariance function {default_str}',
                        choices=['RBFKernel', 'LinearKernel', 'MaternKernel'])
    parser.add_argument('--mean', default='ZeroMean',
                        help=f'Mean for the Gaussian Process {default_str}',
                        choices=['ZeroMean', 'ConstantMean'])
    parser.add_argument('--lik', default='TunePrLikelihood',
                        help=f'Likelihood function {default_str}',
                        choices=['BaselineLikelihood', 'TunePrLikelihood', 'TuneTprLikelihood',
                                 'GaussianLikelihood', 'CalibrationLikelihood'])

    # Variational inference
    parser.add_argument('--num_components', default=1, **int_type,
                        help=f'Number of mixture of Gaussians components {default}')
    parser.add_argument('--num_samples', default=100, **int_type,
                        help='Number of samples for mean and variance estimate of likelihood')
    parser.add_argument('--diag_post', default=False, **bool_type,
                        help=f'Whether the posterior is diagonal or not {default}')
    parser.add_argument('--optimize_inducing', default=True, **bool_type,
                        help=f'Whether to optimize the inducing inputs in training {default}')
    parser.add_argument('--use_loo', default=False, **bool_type,
                        help='Whether to use the LOO (leave one out) loss (for hyper parameters)')

    # Likelihood
    parser.add_argument('--num_samples_pred', default=2000, **int_type,
                        help='Number of samples for mean and variance estimate for prediction')
    parser.add_argument('--sn', default=1.0, **float_type,
                        help=f'Initial standard dev for the Gaussian likelihood {default}')

    # Kernel
    parser.add_argument('--length_scale', default=1.0, **float_type,
                        help=f'Initial length scale for the kernel {default}')
    parser.add_argument('--sf', default=1.0, **float_type,
                        help=f'Initial standard dev for the kernel {default}')
    parser.add_argument('--iso', default=False, **bool_type,
                        help='True to use isotropic kernel otherwise use automatic relevance det')

    # Fairness
    parser.add_argument('--biased_acceptance1', default=0.5, help='', **float_type)
    parser.add_argument('--biased_acceptance2', default=0.5, help='', **float_type)
    parser.add_argument('--s_as_input', default=True, **bool_type,
                        help='Whether the sensitive attribute is treated as part of the input')
    parser.add_argument('--p_s0', default=0.5, **float_type, help='Expected probability of s=0')
    parser.add_argument('--p_s1', default=0.5, **float_type, help='Expected probability of s=1')
    # Demographic parity
    parser.add_argument('--target_rate1', default=0.5, help='', **float_type)
    parser.add_argument('--target_rate2', default=0.5, help='', **float_type)
    parser.add_argument('--probs_from_flipped', default=False, **bool_type,
                        help=f'Whether to take the target rates from the flipping probs {default}')
    parser.add_argument('--average_prediction', default=False, **bool_type,
                        help=f'Whether to take the average of both sensitive attributes {default}')
    parser.add_argument('--p_ybary0_or_ybary1_s0', default=1.0, **float_type,
                        help=('Determine how similar the target labels'
                              'are to the true labels for s=0'))
    parser.add_argument('--p_ybary0_or_ybary1_s1', default=1.0, **float_type,
                        help=('Determine how similar the target labels'
                              'are to the true labels for s=1'))
    # Equalized Odds
    parser.add_argument('--p_ybary0_s0', default=1.0, **float_type, help='Target TNR for s=0')
    parser.add_argument('--p_ybary1_s0', default=1.0, **float_type, help='Target TPR for s=0')
    parser.add_argument('--p_ybary0_s1', default=1.0, **float_type, help='Target TNR for s=1')
    parser.add_argument('--p_ybary1_s1', default=1.0, **float_type, help='Target TPR for s=1')

    # Calibration
    parser.add_argument('--p_yybar0_s0', default=1.0, **float_type, help='Target NPV for s=0')
    parser.add_argument('--p_yybar1_s0', default=1.0, **float_type, help='Target PPV for s=0')
    parser.add_argument('--p_yybar0_s1', default=1.0, **float_type, help='Target NPV for s=1')
    parser.add_argument('--p_yybar1_s1', default=1.0, **float_type, help='Target PPV for s=1')

    # Dataset
    parser.add_argument('--dataset_path', default='', **path_type,
                        help='Path to the numpy file that contains the data')
    parser.add_argument('--dataset_standardize', default=False, **bool_type,
                        help=f'If True, the inputs of the dataset are standardized {default}')

    flags = parser.parse_args(raw_args)

    # Random seeding
    if flags.manual_seed is None:
        flags.manual_seed = random.randint(1, 100000)
    return flags


def str2bool(bool_str):
    """Convert a string to a boolean"""
    if bool_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif bool_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'"{bool_str}" is not a boolean value.')
