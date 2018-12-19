import metrics as metrics_module


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
                class_, metrics_module.Metric):
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
    """Print the result or record it in the summary

    Args:
        metrics: a dictionary with the updated metrics
    """
    for metric in metrics.values():
        result = metric.result()
        print(f"{metric.name}: {result}")
        if summary_writer is not None and step_counter is not None:
            summary_writer.add_scalar(metric.name, result, step_counter)
