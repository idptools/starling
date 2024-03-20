import numpy as np
import pkg_resources
from IPython import embed


def load_matrices():
    mean_matrix_path = pkg_resources.resource_filename(
        "starling.data", "mean_matrix_192.npy"
    )
    std_matrix_path = pkg_resources.resource_filename(
        "starling.data", "std_matrix_192.npy"
    )
    max_standard_path = pkg_resources.resource_filename(
        "starling.data", "max_standard_matrix_192.npy"
    )
    min_standard_path = pkg_resources.resource_filename(
        "starling.data", "min_standard_matrix_192.npy"
    )
    max_expected_path = pkg_resources.resource_filename(
        "starling.data", "max_expected_distances.npy"
    )
    mean_matrix = np.load(mean_matrix_path)
    std_matrix = np.load(std_matrix_path)
    max_standard = np.load(max_standard_path)
    min_standard = np.load(min_standard_path)
    max_expected_distances = np.load(max_expected_path)

    return mean_matrix, std_matrix, max_standard, min_standard, max_expected_distances
    # return max_expected_distances
