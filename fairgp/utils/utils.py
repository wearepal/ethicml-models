from pathlib import Path
import shutil
import numpy as np
import torch

from . import metrics as metrics_module


def init_metrics(metric_flag):
    """Initialize metrics

    Args:
        metric_flag: a string that contains the names of the metrics separated with commas
        is_eager: True if in eager execution
    Returns:
        a dictionary with the initialized metrics
    """
    metrics = {}
    if metric_flag == "":
        return metrics  # No metric names given -> return empty dictionary

    # First, find all metrics
    dict_of_metrics = {}
    for class_name in dir(metrics_module):  # Loop over everything that is defined in `metrics`
        class_ = getattr(metrics_module, class_name)
        # Here, we filter out all functions and other classes which are not metrics
        if isinstance(class_, type(metrics_module.Metric)) and issubclass(
            class_, metrics_module.Metric
        ):
            dict_of_metrics[class_.name] = class_

    if isinstance(metric_flag, list):
        metric_list = metric_flag  # `metric_flag` is already a list
    else:
        metric_list = metric_flag.split(',')  # Split `metric_flag` into a list
    for name in metric_list:
        try:
            # Now we can just look up the metrics in the dictionary we created
            metric = dict_of_metrics[name]
        except KeyError:  # No metric found with the name `name`
            raise ValueError(f"Unknown metric \"{name}\"")
        metrics[name] = metric()
    return metrics


def update_metrics(metrics, inputs, labels, pred_mean):
    """Update metrics

    Args:
        metrics: a dictionary with the initialized metrics
        features: the input
        labels: the correct labels
        pred_mean: the predicted mean
    """
    for metric in metrics.values():
        metric.update(inputs, labels, pred_mean)


def record_metrics(metrics, summary_writer=None, step_counter=None):
    """Print the result and record it in the summary if a summary writer is given

    Args:
        metrics: a dictionary with the updated metrics
    """
    for metric in metrics.values():
        result = metric.result()
        print(f"{metric.name}: {result}")
        if summary_writer is not None and step_counter is not None:
            summary_writer.add_scalar(metric.name, result, step_counter)


def save_predictions(pred_mean, pred_var, save_dir, flags):
    """Save predictions into a numpy file"""
    if flags.preds_path:
        working_dir = save_dir if flags.save_dir else Path(".")
        np.savez_compressed(working_dir / flags.preds_path, pred_mean=pred_mean, pred_var=pred_var)
        print(f"Saved in \"{str(working_dir / flags.preds_path)}\"")


def dataset2numpy(dataset):
    """Convert PyTorch dataset to numpy arrays"""
    # features = []
    # labels = []
    # for feature, label in dataset:
    #     features.append(np.atleast_1d(feature.numpy()))
    #     labels.append(np.atleast_1d(label.numpy()))
    # features, labels = np.concatenate(features, axis=0), np.concatenate(labels, axis=0)
    # if len(features.shape) == 1:
    #     features = features[:, np.newaxis]
    # if len(labels.shape) == 1:
    #     labels = labels[:, np.newaxis]
    features, labels = dataset2tensor(dataset)
    return features.numpy(), labels.numpy()


def dataset2tensor(dataset):
    """Convert PyTorch dataset to tensors"""
    # create loader with enormous batch size
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1_000_000, shuffle=False)
    return next(iter(dataloader))


def save_checkpoint(chkpt_path, model, likelihood, mll, optimizer, epoch, best_loss):
    """Save checkpoint"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'likelihood': likelihood.state_dict(),
        'mll': mll.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
    }
    torch.save(checkpoint, chkpt_path)


def load_checkpoint(checkpoint_path, model, likelihood, mll=None, optimizer=None):
    """Load checkpoint"""
    checkpoint = torch.load(str(checkpoint_path))
    model.load_state_dict(checkpoint['model'])
    likelihood.load_state_dict(checkpoint['likelihood'])
    if mll is not None:
        mll.load_state_dict(checkpoint['mll'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    return start_epoch, best_loss
