import numpy as np

def compute_performance_diversity_correlation(performance_metrics, diversity_metrics):
    """
    Computes the correlation between performance metrics and diversity metrics.

    Args:
        performance_metrics (list of float): A list of performance metric values.
        diversity_metrics (list of float): A list of diversity metric values.
    """

    if len(performance_metrics) != len(diversity_metrics):
        raise ValueError("The length of performance_metrics and diversity_metrics must be the same.")

    correlation_matrix = np.corrcoef(performance_metrics, diversity_metrics)
    correlation = correlation_matrix[0, 1]

    print(f"Correlation between performance and diversity metrics: {correlation:.2f}")
    return correlation