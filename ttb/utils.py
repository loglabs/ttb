"""Utility functions for caching data."""

from datetime import datetime

import cvxpy as cp
import joblib
import numpy as np
import os
import typing


def generate_filename(
    start_date: datetime, end_date: datetime, backend: str, cache_dir: str
) -> str:
    """Generates a filename for the data.

    Args:
        start_date (datetime): Start date of the data (inclusive).
        end_date (datetime): End date of the data (exclusive).
        backend (str): Backend being used.
        cache_dir (str): Directory to store the data.

    Returns:
        str: Filename for the data.
    """
    filename = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y-%m-%d')}_{backend}.joblib"
    return os.path.join(cache_dir, filename)


def save_to_filename(data: object, filename: str):
    """Saves data to filename.

    Args:
        data (object): Data to save.
        filename (str): Filename to save data to.
    """
    joblib.dump(data, filename)


def hadamard_product(arrays: typing.List[np.ndarray]) -> np.ndarray:
    if len(arrays) == 1:
        return arrays[0]

    res = arrays[0] * arrays[1]
    if len(arrays) > 2:
        for i in range(2, len(arrays)):
            res = res * arrays[i]
    return res


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def create_probabilities(
    domain_matrices: typing.List[np.ndarray],
    n: int,
    T: int,
    gamma: float = 1000,
    alpha: float = 1e-3,
    log_step: int = 100,
):
    m = len(domain_matrices)
    probabilities = []

    prev_s_vectors = [
        np.random.uniform(size=(mat.shape[1])) for mat in domain_matrices
    ]
    prev_z = hadamard_product(
        [mat @ s for mat, s in zip(domain_matrices, prev_s_vectors)]
    )
    prev_p = softmax(prev_z)
    probabilities.append(prev_p)

    # Iterate
    for t in range(1, T):
        c = np.ones(n)  # TODO(shreyashankar): change this when we do groups
        s_vectors = [cp.Variable(mat.shape[1]) for mat in domain_matrices]

        z = hadamard_product(
            [mat @ s for mat, s in zip(domain_matrices, s_vectors)]
        )

        op_term = prev_p @ (np.log(c) + z - cp.log_sum_exp(z + np.log(c)))

        smoothness_constraints = [
            float(1 / s_vectors[i].shape[0])
            * (cp.norm(s_vectors[i] - prev_s_vectors[i], 2) ** 2)
            <= gamma
            for i in range(m)
        ]

        nonnegativity_constraints = [s_vectors[i] >= 0 for i in range(m)]

        nondominating_constraints = [
            cp.abs(cp.sum(s_vectors[i]) - cp.sum(s_vectors[j])) <= alpha
            for i in range(m)
            for j in range(m)
            if i != j
        ]

        all_constraints = (
            smoothness_constraints
            + nonnegativity_constraints
            + nondominating_constraints
        )

        # Solve the problem
        prob = cp.Problem(cp.Maximize(op_term), all_constraints)
        prob.solve()

        optimal_value = prob.value
        curr_z = hadamard_product(
            [mat @ s.value for mat, s in zip(domain_matrices, s_vectors)]
        )
        curr_p = softmax(curr_z)
        probabilities.append(curr_p)
        if t % log_step == 0:
            print(f"Iteration {t}: {optimal_value}")

        # Set new prev vectors
        prev_s_vectors = [s_vec.value for s_vec in s_vectors]
        prev_z = curr_z
        prev_p = curr_p

    return probabilities


def create_domain_matrices(
    datasets: list, dataset_names: list
) -> typing.List[np.ndarray]:
    num_examples = sum([len(dataset) for dataset in datasets])
    num_datasets = len(datasets)
    idx_to_group = {}

    A_matrix = np.zeros((num_examples, num_datasets))
    curr_idx = 0
    for i, dataset in enumerate(datasets):
        for j in range(curr_idx, curr_idx + len(dataset)):
            idx_to_group[j] = (dataset_names[i], j - curr_idx)
        A_matrix[curr_idx : curr_idx + len(dataset), i] = 1
        curr_idx += len(dataset)

    return idx_to_group, [A_matrix]


def create_ordering(
    datasets: list,
    dataset_names: list,
    T: int,
    gamma: float = 1000,
    alpha: float = 1e-3,
    log_step: int = 100,
):
    assert len(datasets) == len(dataset_names)

    idx_to_group, domain_matrices = create_domain_matrices(
        datasets, dataset_names
    )
    num_examples = sum([len(dataset) for dataset in datasets])

    probabilities = create_probabilities(
        domain_matrices, num_examples, T, gamma, alpha, log_step
    )

    samples = []
    grouped_probs = []
    for prob in probabilities:
        sample = np.random.choice(len(prob), p=prob)
        samples.append(idx_to_group[sample])

        curr_idx = 0
        group_probs = {}
        for i, dataset in enumerate(datasets):
            group_probs[dataset_names[i]] = sum(
                prob[curr_idx : curr_idx + len(dataset)]
            )
            curr_idx += len(dataset)

        grouped_probs.append(group_probs)

    return samples, grouped_probs
