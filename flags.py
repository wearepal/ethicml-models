import argparse
import random


def parse_arguments(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--inf', default='Variational', help='Inference method')
    parser.add_argument('--cov', default='RBFKernel', help='Covariance function')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--loo_steps', default=0, type=int,
                        help='Number of steps for optimizing LOO loss; 0 disables')
    parser.add_argument('--nelbo_steps', default=0, type=int,
                        help='Number of steps for optimizing NELBO loss; 0 means same as loo_steps')
    parser.add_argument('--num_all', default=200, type=int,
                        help='Suggested total number of examples (datasets don\'t have to use it)')
    parser.add_argument('--num_train', default=50, type=int,
                        help='Suggested number of train examples (datasets don\'t have to use it)')
    parser.add_argument('--num_inducing', default=10, type=int,
                        help='Suggested number of inducing inputs (datasets don\'t have to use it)')
    parser.add_argument('--optimizer', default='RMSPropOptimizer', help='Optimizer to use for SGD')
    parser.add_argument('--model_name', default='local',
                        help='Name of model (used for name of checkpoints)')
    parser.add_argument('--batch_size', default=50, type=int, help='Batch size')
    parser.add_argument('--train_steps', default=50, type=int, help='Number of training steps')
    parser.add_argument('--eval_epochs', default=10000, help='Number of epochs between evaluations')
    parser.add_argument('--summary_steps', default=100, type=int,
                        help='How many steps between saving summary')
    parser.add_argument('--chkpnt_steps', default=5000, type=int,
                        help='How many steps between saving checkpoints')
    parser.add_argument('--save_dir', default='',
                        help='Directory where the checkpoints and summaries are saved (or \'\')')
    parser.add_argument('--plot', default='', help='Which function to use for plotting (or \'\')')
    parser.add_argument('--logging_steps', default=1, type=int,
                        help='How many steps between logging the loss')
    parser.add_argument('--gpus', default='0',
                        help='Which GPUs to use (should normally only be one)')
    parser.add_argument('--preds_path', default='',
                        help='Path where the predictions for the test data will be save (or "")')
    parser.add_argument('--eval_throttle', default=600, type=int,
                        help='How long to wait before evaluating in seconds')
    parser.add_argument('--lr_drop_steps', default=0, type=int,
                        help='Number of steps before doing a learning rate drop')
    parser.add_argument('--lr_drop_factor', default=0.2, type=float,
                        help='For learning rate drop multiply by this factor')
    parser.add_argument('--manual_seed', type=int,
                        help='manual seed, if not given resorts to random seed.')

    # Variational inference
    parser.add_argument('--num_components', default=1, type=int,
                        help='Number of mixture of Gaussians components')
    parser.add_argument('--num_samples', default=100, type=int,
                        help='Number of samples for mean and variance estimate of likelihood')
    parser.add_argument('--diag_post', default=False, type=str2bool,
                        help='Whether the posterior is diagonal or not')
    parser.add_argument('--optimize_inducing', default=True, type=str2bool,
                        help='Whether to optimize the inducing inputs in training')
    parser.add_argument('--use_loo', default=False, type=str2bool,
                        help='Whether to use the LOO (leave one out) loss (for hyper parameters)')

    # Likelihood
    parser.add_argument('--num_samples_pred', default=2000, type=int,
                        help='Number of samples for mean and variance estimate for prediction')

    # Fairness
    parser.add_argument('--biased_acceptance1', default=0.5, type=float, help='')
    parser.add_argument('--biased_acceptance2', default=0.5, type=float, help='')
    parser.add_argument('--s_as_input', default=True, type=str2bool,
                        help='Whether the sensitive attribute is treated as part of the input')
    parser.add_argument('--p_s0', default=0.5, type=float, help='')
    parser.add_argument('--p_s1', default=0.5, type=float, help='')
    # Demographic parity
    parser.add_argument('--target_rate1', default=0.5, type=float, help='')
    parser.add_argument('--target_rate2', default=0.5, type=float, help='')
    parser.add_argument('--probs_from_flipped', default=False, type=str2bool,
                        help='Whether to take the target rates from the flipping probs')
    parser.add_argument('--average_prediction', default=False, type=str2bool,
                        help='Whether to take the average of both sensitive attributes')
    parser.add_argument('--p_ybary0_or_ybary1_s0', default=1.0, type=float,
                        help=('Determine how similar the target labels'
                              'are to the true labels for s=0'))
    parser.add_argument('--p_ybary0_or_ybary1_s1', default=1.0, type=float,
                        help=('Determine how similar the target labels'
                              'are to the true labels for s=1'))
    # Equalized Odds
    parser.add_argument('--p_ybary0_s0', default=1.0, type=float, help='')
    parser.add_argument('--p_ybary1_s0', default=1.0, type=float, help='')
    parser.add_argument('--p_ybary0_s1', default=1.0, type=float, help='')
    parser.add_argument('--p_ybary1_s1', default=1.0, type=float, help='')

    # Dataset
    parser.add_argument('--dataset_path', default='',
                        help='Path to the numpy file that contains the data')
    parser.add_argument('--dataset_standardize', default=False, type=str2bool,
                        help='If True, the inputs of the dataset are standardized')

    args = parser.parse_args(raw_args)

    # Random seeding
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 100000)
    return args


def str2bool(bool_str):
    """Convert a string to a boolean"""
    if bool_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif bool_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'"{bool_str}" is not a boolean value.')
