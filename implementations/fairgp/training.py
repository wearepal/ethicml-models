"""Training loop for GP training"""
import time
from pathlib import Path
from tempfile import mkdtemp
import random

import torch
import numpy as np

from gpytorch import settings

from .utils import utils, plot
from .construction import construct_model


def main_loop(flags):
    random.seed(flags.manual_seed)
    np.random.seed(flags.manual_seed)
    torch.manual_seed(flags.manual_seed)
    torch.cuda.manual_seed_all(flags.manual_seed)
    # Check if CUDA is available
    flags.use_cuda = torch.cuda.is_available()
    print(flags)  # print the configuration
    # Construct model and all other necessary objects
    model, likelihood, mll, optimizer, train_ds, test_ds = construct_model(flags)
    print(f"Number of training samples: {len(train_ds)}")

    shuff = len(train_ds) != flags.batch_size  # don't shuffle if each batch equals the dataset
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
    previous_checkpoints = list(save_dir.glob('checkpoint_*.pth.tar'))
    if previous_checkpoints:
        latest_chkpt = max(previous_checkpoints)  # `max()` is here equivalent to `sorted(...)[-1]`
        print(f"===> Restoring from '{latest_chkpt}'")
        start_epoch, best_loss = utils.load_checkpoint(
            latest_chkpt, model, likelihood, mll, optimizer
        )

    print(f"Training for {flags.epochs} epochs")
    # Main training loop
    for epoch in range(start_epoch, start_epoch + flags.epochs):
        print(f"Training on epoch {epoch}")
        start = time.time()
        step_counter = (epoch - 1) * len(train_loader)
        with settings.use_toeplitz(not flags.use_cuda):
            # settings.fast_computations(covar_root_decomposition=False),\
            # settings.lazily_evaluate_kernels(state=False),\
            # settings.tridiagonal_jitter(1e-2),\
            # settings.max_cholesky_numel(4096),\
            # settings.max_preconditioner_size(10),\
            train(model, optimizer, train_loader, mll, step_counter, flags)
        end = time.time()
        print(f"Train time for epoch {epoch}: {end - start:0.2f}s")
        if epoch % flags.eval_epochs == 0:
            # do evaluation and update the best loss
            val_loss = evaluate(model, likelihood, test_loader, mll, step_counter, flags)
            if flags.save_best and val_loss < best_loss:
                best_loss = val_loss
                print(f"Best loss yet. Saving in '{best_checkpoint}'")
                utils.save_checkpoint(
                    best_checkpoint, model, likelihood, mll, optimizer, epoch, best_loss
                )

        if epoch % flags.chkpt_epochs == 0:
            # Save checkpoint
            chkpt_path = save_dir / f'checkpoint_{epoch:04d}.pth.tar'
            print(f"===> Saving checkpoint in '{chkpt_path}'")
            utils.save_checkpoint(chkpt_path, model, likelihood, mll, optimizer, epoch, best_loss)

    # if predictions are to be save or to be plotted, then make predictions on the test set
    if flags.preds_path or flags.plot:
        # print("Loading best model...")
        # utils.load_checkpoint(best_checkpoint, model, likelihood)
        print("Making predictions...")
        pred_mean, pred_var = predict(model, likelihood, test_loader, flags.use_cuda)
        utils.save_predictions(pred_mean, pred_var, save_dir, flags)
        if flags.plot:
            getattr(plot, flags.plot)(pred_mean, pred_var, train_ds, test_ds)


def train(model, optimizer, dataset, mll, previous_steps, flags):
    """Train for one epoch"""
    start = time.time()
    model.train()
    mll.likelihood.train()
    for (step, (inputs, labels)) in enumerate(dataset, start=previous_steps + 1):
        if flags.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        # print(
        #     f" lengthscale:"
        #     f" {model.covar_module.base_kernel.log_lengthscale.detach().exp().cpu().numpy()}"
        # )
        optimizer.zero_grad()  # Zero backpropped gradients from previous iteration
        try:
            # Get predictive output
            output = model(inputs)
        except RuntimeError as e:
            error = str(e)
            if error == "NaNs encounterd when trying to perform matrix-vector multiplication":
                raise RuntimeError("Consider changing the lengthscale")
            raise RuntimeError(error)
        # Calc loss and backprop gradients
        loss = -mll(output, labels)
        loss.backward()  # back-propagation
        optimizer.step()  # update parameters

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

    loss_sum = 0.0
    num_samples = 0
    metrics = utils.init_metrics(flags.metrics)

    with torch.no_grad():
        for inputs, labels in dataset:
            if flags.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            loss = -mll(output, labels)
            loss_sum += loss.item() * inputs.size()[0]  # needed because of different batch sizes
            num_samples += inputs.size()[0]
            # Get classification predictions
            observed_pred = likelihood(output)
            utils.update_metrics(metrics, inputs, labels, observed_pred.mean)
    average_loss = loss_sum / num_samples
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
