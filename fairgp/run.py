"""Training loop for GP training"""
import time
from pathlib import Path
from tempfile import mkdtemp
import shutil

import torch
from torchnet.meter import AverageValueMeter
import numpy as np

import gpytorch
from gpytorch import settings

import fair_likelihood
from flags import parse_arguments
import gp_model
import datasets
import utils
import plot


def train(model, optimizer, dataset, mll, previous_steps, flags):
    """Train for one epoch"""
    start = time.time()
    model.train()
    mll.likelihood.train()
    for (step, (inputs, labels)) in enumerate(dataset, start=previous_steps + 1):
        if flags.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(inputs)
        # Calc loss and backprop gradients
        loss = -mll(output, labels)
        loss.backward()
        optimizer.step()

        if flags.logging_steps != 0 and (step - 1) % flags.logging_steps == 0:
            print(f"Step #{step} ({time.time() - start:.4f} sec)\t", end=' ')
            print(
                f"loss: {loss.item():.3f}"
                # f" log_lengthscale:"
                # f" {model.covar_module.base_kernel.log_lengthscale.detach().cpu().numpy()}"
                # f"log_noise: {model.likelihood.log_noise.item()}"
            )
            # for loss_name, loss_value in obj_func.items():
            #     print(f"{loss_name}: {loss_value:.2f}", end=' ')
            start = time.time()


def evaluate(model, likelihood, dataset, mll, step_counter, flags):
    """Evaluate on test set"""
    # Go into eval mode
    model.eval()
    likelihood.eval()

    loss_meter = AverageValueMeter()
    metrics = utils.init_metrics(flags.metrics)

    with torch.no_grad():
        for inputs, labels in dataset:
            if flags.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            loss = -mll(output, labels)
            loss_meter.add(loss.item())
            # Get classification predictions
            observed_pred = likelihood(output)
            utils.update_metrics(metrics, inputs, labels, observed_pred.mean)
    average_loss = loss_meter.mean
    print(f"Average loss: {average_loss}")
    utils.record_metrics(metrics, step_counter=step_counter)
    return average_loss


def predict(model, likelihood, dataset, use_cuda):
    """Make predictions"""
    # Go into eval mode
    model.eval()
    likelihood.eval()

    pred_mean = []
    pred_var = []
    with torch.no_grad():
        for inputs, _ in dataset:
            if use_cuda:
                inputs = inputs.cuda()
            # Get classification predictions
            observed_pred = likelihood(model(inputs))
            # Get mean and variance
            pred_mean.append(observed_pred.mean.cpu().numpy())
            pred_var.append(observed_pred.variance.cpu().numpy())
    pred_mean, pred_var = np.concatenate(pred_mean, axis=0), np.concatenate(pred_var, axis=0)
    if len(pred_mean.shape) == 1:
        pred_mean, pred_var = pred_mean[:, np.newaxis], pred_var[:, np.newaxis]
    return pred_mean, pred_var


def construct(flags):
    """Construct GP object, likelihood and marginal log likelihood function from the flags

    Args:
        flags: settings object
    """
    # We use the built-in function "getattr" to turn the flags into Python references
    data_func = getattr(datasets, flags.data)
    Likelihood = getattr(fair_likelihood, flags.lik)
    Kernel = getattr(gpytorch.kernels, flags.cov)
    GPModel = getattr(gp_model, flags.inf)
    Optimizer = getattr(torch.optim, flags.optimizer)

    # Get dataset objects
    train_ds, test_ds, inducing_inputs = data_func(flags)
    input_dim = inducing_inputs.shape[1]

    if flags.use_cuda:
        inducing_inputs = inducing_inputs.cuda()

    # Initialize likelihood, kernel and model
    likelihood: gpytorch.likelihoods.Likelihood = Likelihood(flags)
    # TODO: figure out how to specify initial values for the kernel parameters
    kernel_unscaled: gpytorch.kernels.Kernel = Kernel(ard_num_dims=None if flags.iso else input_dim)
    kernel = gpytorch.kernels.ScaleKernel(kernel_unscaled)
    model: gpytorch.models.GP = GPModel(train_ds=train_ds, inducing_inputs=inducing_inputs,
                                        likelihood=likelihood, kernel=kernel, flags=flags)
    if flags.use_cuda:
        model, likelihood = model.cuda(), likelihood.cuda()

    # "Loss" for the GP model: the marginal log likelihood
    mll = model.get_marginal_log_likelihood(likelihood, len(train_ds))

    # Initialize optimizer
    optimizer: torch.optim.Optimizer = Optimizer(
        list(model.parameters()) + list(likelihood.parameters()), lr=flags.lr)
    return model, likelihood, mll, optimizer, train_ds, test_ds


def main_loop(flags):
    # Check if CUDA is available
    flags.use_cuda = torch.cuda.is_available()
    # Construct model and all other necessary objects
    model, likelihood, mll, optimizer, train_ds, test_ds = construct(flags)
    print(f"Number of training samples: {len(train_ds)}")

    shuff = (len(train_ds) != flags.batch_size)  # don't shuffle if each batch equals the dataset
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=flags.batch_size, shuffle=shuff)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=flags.batch_size)

    # Set checkpoint path
    if flags.save_dir:
        save_dir = Path(flags.save_dir) / flags.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(mkdtemp())  # Create temporary directory

    best_loss = np.inf
    start_epoch = 1

    # Restore from checkpoint if one exists
    best_checkpoint = save_dir / 'model_best.pth.tar'
    if best_checkpoint.is_file():
        checkpoint = torch.load(str(best_checkpoint))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        likelihood.load_state_dict(checkpoint['likelihood'])
        mll.load_state_dict(checkpoint['mll'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

    # Main training loop
    for epoch in range(start_epoch, start_epoch + flags.epochs):
        print(f"Training on epoch {epoch}")
        start = time.time()
        step_counter = (epoch - 1) * len(train_loader)
        with settings.use_toeplitz(not flags.use_cuda),\
                settings.fast_computations(covar_root_decomposition=False),\
                settings.num_likelihood_samples(flags.num_samples):
            # settings.lazily_evaluate_kernels(state=False),\
            # settings.max_cholesky_numel(4096),\
            # settings.max_preconditioner_size(10),\
            train(model, optimizer, train_loader, mll, step_counter, flags)
        end = time.time()
        print(f"Train time for epochs {epoch} (global step {step_counter}):"
              f" {end - start:0.2f}s")
        if epoch % flags.eval_epochs == 0:
            # do evaluation and update the best loss
            val_loss = evaluate(model, likelihood, test_loader, mll, step_counter, flags)
            is_best_loss_yet = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            # Save checkpoint
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'likelihood': likelihood.state_dict(),
                'mll': mll.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
            }
            save_checkpoint(checkpoint, f'checkpoint_{epoch:04d}.pth.tar', is_best_loss_yet,
                            save_dir)

    # if predictions are to be save or to be plotted, then make predictions on the test set
    if flags.preds_path or flags.plot:
        print("Making predictions...")
        pred_mean, pred_var = predict(model, likelihood, test_loader, flags.use_cuda)
        utils.save_predictions(pred_mean, pred_var, save_dir, flags)
        if flags.plot:
            getattr(plot, flags.plot)(pred_mean, pred_var, train_ds, test_ds)


def save_checkpoint(checkpoint, filename, is_best_loss_yet, save_dir):
    print(f"===> Saving checkpoint '{filename}'")
    model_filename = save_dir / filename
    torch.save(checkpoint, model_filename)
    best_filename = save_dir / 'model_best.pth.tar'
    print(f"===> Saved checkpoint '{model_filename}'")
    if is_best_loss_yet:
        shutil.copyfile(model_filename, best_filename)
        print(f"Best loss yet. Saved in '{best_filename}'")


if __name__ == "__main__":
    main_loop(parse_arguments())
