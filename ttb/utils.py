"""Utility functions for caching data."""

from datetime import datetime

import cvxpy as cp
import joblib
import numpy as np
import os
import random
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


def aggregate_min(
    arrays: typing.List[np.ndarray], use_cvx: bool = True
) -> np.ndarray:
    res = arrays[0]

    for i in range(1, len(arrays)):
        res = (
            cp.minimum(res, arrays[i])
            if use_cvx
            else np.minimum(res, arrays[i])
        )

    return res


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def create_probabilities(
    domain_matrices: typing.List[np.ndarray],
    T: int,
    gamma: float = 0.5,
    num_peaks: int = 5,
    start_max: int = 10,  # highest value signal can take to start with
    duration: int = 1,    # how many timesteps each signal value should persist for 
    log_step: int = 10,
    seed: int = 42,
    periodicity_slack: float = 2,
    periodicity: typing.List[typing.Tuple[int, int]] = []  # tuple of domains and periods
):
    n = domain_matrices[0].shape[0]
    m = len(domain_matrices)
    probabilities = []
    signals = []

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)

    prev_s_vectors = [
        [start_max / mat.shape[1]] * mat.shape[1] # initialize to midpoint of range
        for mat in domain_matrices
    ]
    prev_z = aggregate_min(
        [mat @ s for mat, s in zip(domain_matrices, prev_s_vectors)],
        use_cvx=False,
    )
    prev_p = softmax(prev_z)

    signals.append(prev_s_vectors)
    probabilities.append(prev_p)

    peaks = np.random.choice(n, num_peaks).tolist()
    duration_counter = 0

    # Iterate
    for t in range(1, T+1, duration):
        c = np.ones(n)  # TODO(shreyashankar): change this when we do groups
        s_vectors = [cp.Variable(mat.shape[1]) for mat in domain_matrices]

        # z is concave in optimization variable s
        z = aggregate_min(
            [mat @ s for mat, s in zip(domain_matrices, s_vectors)]
        )

        # convex alternative to z (take mean instead of min over domain types)
        pseudo_z = (
            1
            / m
            * cp.sum(
                [mat @ s for mat, s in zip(domain_matrices, s_vectors)], axis=1
            )
        )

        # instead of maximizing KL divergence with prev_p (not convex)
        # we find some entropic distribution p* with which we can minimize KL divergence (convex)
        peaks.pop(0)
        peaks.append(np.random.randint(n))
        p_star = np.zeros(prev_p.shape)

        if len(peaks) == 1:
            p_star[peaks[0]] = 1
        else:
            p_star[peaks] = start_max
            p_star = softmax(p_star)

        obj = cp.Minimize(
            -1
            * (p_star @ (np.log(c) + z - cp.log_sum_exp(pseudo_z + np.log(c))))
        )

        # prevent rapid changes from one timestep to another using L2 norm
        smoothness_constraints = [
            (
                1
                / (s_vectors[i].shape[0] ** 0.5)
                * cp.norm(s_vectors[i] - prev_s_vectors[i], 2)
            )
            <= gamma
            for i in range(m)
        ]

        nonnegativity_constraints = [s_vectors[i] >= 0 for i in range(m)]

        # value of jth value of domain i should be the same as (j-1)th value
        # 'period' timesteps ago
        periodic_constraints = []

        for i,period in periodicity:
            if t > period:
                periodic_constraints.append(
                    cp.norm(s_vectors[i] - np.roll(signals[-period][i], 1), 1) <= periodicity_slack 
                )

        all_constraints = smoothness_constraints + nonnegativity_constraints + periodic_constraints

        # Solve the problem
        prob = cp.Problem(obj, all_constraints)
        prob.solve()

        optimal_value = prob.value

        curr_z = aggregate_min(
            [mat @ s.value for mat, s in zip(domain_matrices, s_vectors)],
            use_cvx=False,
        )
        curr_p = softmax(curr_z)

        # signal value should persist for entirety of duration
        for _ in range(duration):
            if len(signals) <= T:
                signals.append([s.value for s in s_vectors])
                probabilities.append(curr_p)

        if t % log_step == 0:
            print(f"Iteration {t}: {optimal_value}")

        # Set new prev vectors
        prev_s_vectors = [s_vec.value for s_vec in s_vectors]
        prev_z = curr_z
        prev_p = curr_p

    return probabilities, signals


def create_ordering(
    datasets: list,
    dataset_names: list,
    T: int,
    gamma: float = 0.5,
    num_peaks: int = 5,
    start_max: int = 10,
    log_step: int = 10,
):
    assert len(datasets) == len(dataset_names)

    idx_to_group, domain_matrices = create_domain_matrices(
        datasets, dataset_names
    )

    probabilities, _ = create_probabilities(
        domain_matrices,
        T,
        gamma=gamma,
        num_peaks=num_peaks,
        start_max=start_max,
        log_step=log_step,
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
