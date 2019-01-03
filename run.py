"""Training loop for GP training"""
import time
from pathlib import Path
from tempfile import mkdtemp
import shutil

import torch
from torchnet.meter import AverageValueMeter
import numpy as np
# from matplotlib import pyplot as plt

import gpytorch
from gpytorch.mlls.variational_elbo import VariationalELBO

import fair_likelihood
from flags import parse_arguments
from gp_model import GPClassificationModel
from dataset import from_numpy
import utils


def train(model, optimizer, dataset, loss_func, step_counter, args):
    """Train for one epoch"""
    start = time.time()
    model.train()
    loss_func.likelihood.train()
    for (batch_num, (inputs, labels)) in enumerate(dataset):
        if args.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(inputs)
        # Calc loss and backprop gradients
        with gpytorch.settings.num_likelihood_samples(args.num_samples):
            loss = -loss_func(output, labels)
        loss.backward()
        optimizer.step()

        if args.logging_steps != 0 and batch_num % args.logging_steps == 0:
            print(f"Step #{step_counter + batch_num} ({time.time() - start:.4f} sec)\t", end=' ')
            print(f"loss: {loss.item():.3f}", end=' ')
            # for loss_name, loss_value in obj_func.items():
            #     print(f"{loss_name}: {loss_value:.2f}", end=' ')
            print("")  # newline
            start = time.time()


def evaluate(model, likelihood, dataset, loss_func, step_counter, args):
    """Evaluate on test set"""
    # Go into eval mode
    model.eval()
    likelihood.eval()

    loss_meter = AverageValueMeter()
    metrics = utils.init_metrics(args.metrics)

    with torch.no_grad():
        for inputs, labels in dataset:
            if args.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            loss = -loss_func(output, labels)
            loss_meter.add(loss.item())
            # Get classification predictions
            observed_pred = likelihood(output)
            utils.update_metrics(metrics, inputs, labels, observed_pred.mean)
    average_loss = loss_meter.mean
    print(f"Average loss: {average_loss}")
    utils.record_metrics(metrics, step_counter=step_counter)
    return average_loss


def predict(model, likelihood, dataset):
    """Make predictions"""
    # Go into eval mode
    model.eval()
    likelihood.eval()

    predictions = []
    with torch.no_grad():
        for sample in dataset:
            inputs, _ = sample
            # Get classification predictions
            observed_pred = likelihood(model(inputs))
            # Get the predicted labels (probabilites of belonging to the positive class)
            # Transform these probabilities to be 0/1 labels
            predictions.append(observed_pred.mean.ge(0.5).float().mul(2).sub(1).numpy())
    return np.concatenate(predictions, axis=0)


def main(args):
    # Toy data:
    # train_x = torch.linspace(0, 1, 10)
    # train_y = torch.sign(torch.cos(train_x * (4 * np.pi)))
    # train_s = 0.5 * (torch.sign(torch.linspace(-1, 1, 10)) + 1)
    # train_labels = torch.stack((train_y, train_s), dim=-1)
    # Test x are regularly spaced by 0.01 0,1 inclusive
    # test_x = torch.linspace(0, 1, 101)
    # test_y = torch.sign(torch.cos(test_x * (4 * np.pi)))

    # Load data
    train_ds, test_ds, inducing_inputs = from_numpy(args)
    print(f"Number of training samples: {len(train_ds)}")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)

    # Set checkpoint path
    if args.save_dir:
        save_dir = Path(args.save_dir) / args.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(mkdtemp())  # Create temporary directory

    # Check if CUDA is available
    args.use_cuda = torch.cuda.is_available()

    # Initialize model and likelihood
    if args.use_cuda:
        inducing_inputs = inducing_inputs.cuda()
    model = GPClassificationModel(inducing_inputs, args)
    likelihood = getattr(fair_likelihood, args.lik)(args)
    if args.use_cuda:
        model, likelihood = model.cuda(), likelihood.cuda()

    # Initialize optimizer
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the amount of training data
    mll = VariationalELBO(likelihood, model, len(train_ds))

    best_loss = np.inf
    start_epoch = 1

    # Restore from existing checkpoint
    best_checkpoint = save_dir / 'model_best.pth.tar'
    if best_checkpoint.is_file():
        checkpoint = torch.load(str(best_checkpoint))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        likelihood.load_state_dict(checkpoint['likelihood'])
        mll.load_state_dict(checkpoint['mll'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

    # Find optimal model hyperparameters
    for i in range(args.epochs):
        epoch = start_epoch + i
        print(f"Training on epoch {epoch}")
        start = time.time()
        step_counter = (epoch - 1) * len(train_loader)
        train(model, optimizer, train_loader, mll, step_counter, args)
        end = time.time()
        print(f"Train time for epochs {epoch} (global step {step_counter}):"
              f" {end - start:0.2f}s")
        if epoch % args.eval_epochs == 0:
            # do evaluation and update the best loss
            val_loss = evaluate(model, likelihood, test_loader, mll, step_counter, args)
            is_best_loss_yet = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
        else:
            # we don't know if the loss is better so we'll assume it's not to be on the safe side
            is_best_loss_yet = False

        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'likelihood': likelihood.state_dict(),
            'mll': mll.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
        }
        save_checkpoint(checkpoint, f'checkpoint_{epoch:04d}.pth.tar', is_best_loss_yet, save_dir)

    # # Initialize fig and axes for plot
    # fig, plot = plt.subplots(figsize=(4, 3))
    # plot.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # plot.plot(test_x.numpy(), pred_labels.numpy(), 'b')
    # plot.set_ylim([-3, 3])
    # plot.legend(['Observed Data', 'Mean', 'Confidence'])


def save_checkpoint(checkpoint, filename, is_best_loss_yet, save_dir):
    print(f"===> Saving checkpoint '{filename}'")
    model_filename = save_dir / filename
    torch.save(checkpoint, model_filename)
    best_filename = save_dir / 'model_best.pth.tar'
    if is_best_loss_yet:
        shutil.copyfile(model_filename, best_filename)
    print(f"===> Saved checkpoint '{model_filename}'")


if __name__ == "__main__":
    main(parse_arguments())
