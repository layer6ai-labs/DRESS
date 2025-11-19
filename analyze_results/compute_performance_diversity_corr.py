import numpy as np

def compute_performance_metric_correlation(performances, metrics):
    """
    Computes the correlation between few-shot learning performances and metrics.

    Args:
        performances (list of float): A list of few-shot performance values.
        metrics (list of float): A list of metric values.
    """

    if len(performances) != len(metrics):
        raise ValueError("The length of performances and metrics must be the same.")

    correlation_matrix = np.corrcoef(performances, metrics)
    correlation = correlation_matrix[0, 1]

    print(f"Correlation between performances and metrics: {correlation:.2f}")
    return correlation