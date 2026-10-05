"""Gaussian-mixture solver for the self-consistent noisy-belief prior."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc


@dataclass
class GaussianMixture:
    """Store mixture weights, 2D means, and full covariances."""

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray


def default_basis():
    """Return an orthonormal basis of the zero-mean log-belief plane."""
    b1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    b2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
    return np.column_stack([b1, b2])


def signal_displacements(likelihood, basis=None):
    """Project signal log-likelihood displacements into 2D coordinates."""
    likelihood = np.asarray(likelihood, dtype=float)
    if likelihood.shape != (3, 3) or np.any(likelihood <= 0):
        raise ValueError("likelihood must be a positive 3 x 3 array")
    if not np.allclose(likelihood.sum(axis=0), 1.0):
        raise ValueError("each likelihood column must sum to one")
    basis = default_basis() if basis is None else np.asarray(basis, dtype=float)
    centered_log = np.log(likelihood) - np.log(likelihood).mean(axis=1, keepdims=True)
    return centered_log @ basis


def initial_gmm(n_components, scale=2.0):
    """Create a rotation-symmetric, broad initial mixture."""
    component_scales = scale * np.linspace(0.6, 1.4, n_components)
    return GaussianMixture(
        np.full(n_components, 1.0 / n_components),
        np.zeros((n_components, 2)),
        np.array([value**2 * np.eye(2) for value in component_scales]),
    )


def _regularize(covariance, min_variance):
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    return (eigenvectors * np.maximum(eigenvalues, min_variance)) @ eigenvectors.T


def _component_logpdf(values, means, covariances):
    values = np.atleast_2d(np.asarray(values, dtype=float))
    output = np.empty((len(values), len(means)))
    constant = values.shape[1] * np.log(2 * np.pi)
    for k, (mean, covariance) in enumerate(zip(means, covariances)):
        chol = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(chol, (values - mean).T).T
        output[:, k] = -0.5 * (constant + 2 * np.log(np.diag(chol)).sum()
                                + np.sum(whitened**2, axis=1))
    return output


def gmm_logpdf(values, mixture):
    """Evaluate the mixture log density."""
    component = _component_logpdf(values, mixture.means, mixture.covariances)
    return logsumexp(component + np.log(mixture.weights), axis=1)


def gmm_denoise(values, mixture, noise_sigma):
    """Evaluate the exact posterior-mean denoiser for a GMM prior."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    observation_covariances = mixture.covariances + noise_sigma**2 * np.eye(2)
    log_joint = _component_logpdf(values, mixture.means, observation_covariances)
    log_joint += np.log(mixture.weights)
    responsibilities = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
    output = np.zeros_like(values)
    for k in range(len(mixture.weights)):
        gain = np.linalg.solve(observation_covariances[k], mixture.covariances[k]).T
        posterior_mean = mixture.means[k] + (values - mixture.means[k]) @ gain.T
        output += responsibilities[:, k, None] * posterior_mean
    return output


def sample_gmm(mixture, n_samples, rng):
    """Draw samples from a Gaussian mixture."""
    components = rng.choice(len(mixture.weights), n_samples, p=mixture.weights)
    samples = np.empty((n_samples, 2))
    for k in range(len(mixture.weights)):
        indices = np.flatnonzero(components == k)
        if len(indices):
            samples[indices] = rng.multivariate_normal(
                mixture.means[k], mixture.covariances[k], len(indices)
            )
    return samples


def gmm_moments(mixture):
    """Return the mixture mean and covariance."""
    mean = mixture.weights @ mixture.means
    centered = mixture.means - mean
    covariance = np.sum(
        mixture.weights[:, None, None]
        * (mixture.covariances + centered[:, :, None] * centered[:, None, :]),
        axis=0,
    )
    return mean, covariance


def combine_gmms(mixtures, weights=None):
    """Combine several Gaussian mixtures into one mixture."""
    weights = np.full(len(mixtures), 1 / len(mixtures)) if weights is None else np.asarray(weights)
    return GaussianMixture(
        np.concatenate([weight * mixture.weights for weight, mixture in zip(weights, mixtures)]),
        np.vstack([mixture.means for mixture in mixtures]),
        np.vstack([mixture.covariances for mixture in mixtures]),
    )


def _stratified_uniform(rng, n_samples):
    values = (np.arange(n_samples) + rng.random()) / n_samples
    return values[rng.permutation(n_samples)]


def _sample_signals(rng, likelihood, state, n_samples, stratified):
    cdf = np.cumsum(likelihood[:, state])
    uniforms = _stratified_uniform(rng, n_samples) if stratified else rng.random(n_samples)
    return np.searchsorted(cdf, uniforms)


def _sample_ages(rng, n_samples, age_lambda, max_age, stratified):
    if age_lambda == 0:
        return np.ones(n_samples, dtype=int)
    uniforms = _stratified_uniform(rng, n_samples) if stratified else rng.random(n_samples)
    ages = np.ceil(np.log1p(-uniforms) / np.log(age_lambda)).astype(int)
    return np.minimum(np.maximum(ages, 1), max_age)


def _sample_noise(rng, n_samples, sigma, stratified):
    if not stratified:
        return rng.normal(0.0, sigma, size=(n_samples, 2))
    half = n_samples // 2
    base = rng.normal(0.0, sigma, size=(half, 2))
    noise = np.vstack([base, -base])
    if n_samples % 2:
        noise = np.vstack([noise, np.zeros((1, 2))])
    return noise[rng.permutation(n_samples)]


def simulate_map(mixture, likelihood, displacements, noise_sigma, age_lambda,
                 samples_per_state, max_age=80, seed=0, stratified=True):
    """Simulate the state-conditioned self-consistency map with continuous noise."""
    if not 0 <= age_lambda < 1:
        raise ValueError("age_lambda must be in [0, 1)")
    rng = np.random.default_rng(seed)
    mapped = []
    ages_by_state = []
    for state in range(3):
        ages = _sample_ages(rng, samples_per_state, age_lambda, max_age, stratified)
        beliefs = np.zeros((samples_per_state, 2))
        for step in range(1, int(ages.max()) + 1):
            active = np.flatnonzero(ages >= step)
            signals = _sample_signals(rng, likelihood, state, len(active), stratified)
            noise = _sample_noise(rng, len(active), noise_sigma, stratified)
            query = beliefs[active] + displacements[signals] + noise
            beliefs[active] = gmm_denoise(query, mixture, noise_sigma)
        mapped.append(beliefs)
        ages_by_state.append(ages)
    return np.stack(mapped), np.stack(ages_by_state)


def _initialize_from_samples(values, n_components, rng, min_variance):
    means = [values[rng.integers(len(values))]]
    while len(means) < n_components:
        distance = np.min(
            np.sum((values[:, None, :] - np.asarray(means)[None, :, :])**2, axis=2),
            axis=1,
        )
        total = distance.sum()
        means.append(values[rng.integers(len(values))] if total == 0
                     else values[rng.choice(len(values), p=distance / total)])
    covariance = _regularize(np.cov(values, rowvar=False), min_variance)
    return GaussianMixture(
        np.full(n_components, 1.0 / n_components),
        np.asarray(means),
        np.repeat(covariance[None, :, :], n_components, axis=0),
    )


def fit_gmm(values, n_components, initial=None, min_variance=0.01,
            max_iterations=100, tolerance=1e-5, seed=0):
    """Fit a full-covariance Gaussian mixture by EM."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    mixture = _initialize_from_samples(values, n_components, rng, min_variance) \
        if initial is None else GaussianMixture(
            initial.weights.copy(), initial.means.copy(), initial.covariances.copy()
        )
    previous = -np.inf
    global_covariance = _regularize(np.cov(values, rowvar=False), min_variance)

    for iteration in range(1, max_iterations + 1):
        log_joint = _component_logpdf(values, mixture.means, mixture.covariances)
        log_joint += np.log(np.maximum(mixture.weights, 1e-300))
        log_normalizer = logsumexp(log_joint, axis=1, keepdims=True)
        responsibilities = np.exp(log_joint - log_normalizer)
        lower_bound = float(log_normalizer.mean())
        mass = responsibilities.sum(axis=0)

        for k in range(n_components):
            if mass[k] < 1e-3 * len(values):
                mixture.means[k] = values[rng.integers(len(values))]
                mixture.covariances[k] = global_covariance
                mass[k] = 1e-3 * len(values)
                continue
            mixture.means[k] = responsibilities[:, k] @ values / mass[k]
            centered = values - mixture.means[k]
            covariance = (centered * responsibilities[:, k, None]).T @ centered / mass[k]
            mixture.covariances[k] = _regularize(covariance, min_variance)

        mixture.weights = mass / mass.sum()
        if np.isfinite(previous) and abs(lower_bound - previous) <= tolerance * (1 + abs(previous)):
            break
        previous = lower_bound

    return mixture, {"iterations": iteration, "log_likelihood": lower_bound}


def align_components(reference, target):
    """Match target components to reference components before damping."""
    scale = np.mean(np.trace(reference.covariances, axis1=1, axis2=2)) / 2
    mean_cost = np.sum((reference.means[:, None] - target.means[None, :])**2, axis=2) / scale
    covariance_cost = np.sum(
        (reference.covariances[:, None] - target.covariances[None, :])**2,
        axis=(2, 3),
    ) / scale**2
    rows, columns = linear_sum_assignment(mean_cost + 0.25 * covariance_cost)
    order = columns[np.argsort(rows)]
    return GaussianMixture(target.weights[order], target.means[order], target.covariances[order])


def damp_gmm(old, new, damping, min_variance):
    """Damp matched GMM parameters."""
    covariances = np.array([
        _regularize((1 - damping) * old.covariances[k] + damping * new.covariances[k], min_variance)
        for k in range(len(old.weights))
    ])
    weights = (1 - damping) * old.weights + damping * new.weights
    return GaussianMixture(
        weights / weights.sum(),
        (1 - damping) * old.means + damping * new.means,
        covariances,
    )


def empirical_moments(values):
    """Return an empirical mean and covariance."""
    return values.mean(axis=0), np.cov(values, rowvar=False)


def sliced_wasserstein(first, second, n_directions=32, seed=0):
    """Estimate a rotation-independent distributional gap."""
    rng = np.random.default_rng(seed)
    n = min(len(first), len(second))
    first = first[rng.choice(len(first), n, replace=False)]
    second = second[rng.choice(len(second), n, replace=False)]
    directions = rng.normal(size=(n_directions, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    first_projection = np.sort(first @ directions.T, axis=0)
    second_projection = np.sort(second @ directions.T, axis=0)
    return float(np.mean(np.sqrt(np.mean((first_projection - second_projection)**2, axis=0))))


def solve_gmm_fixed_point(likelihood, n_components=5, noise_sigma=0.15,
                          age_lambda=0.9, samples_per_state=5000, max_age=80,
                          initial_scale=2.0, min_std=0.1, damping=0.5,
                          max_iterations=30, em_max_iterations=100,
                          residual_tolerance=0.05, patience=3, seed=15,
                          stratified=True, verbose=True):
    """Run the projected GMM fixed-point iteration."""
    likelihood = np.asarray(likelihood, dtype=float)
    displacements = signal_displacements(likelihood)
    mixture = initial_gmm(n_components, initial_scale)
    min_variance = min_std**2
    history = []
    stable_iterations = 0

    for iteration in range(1, max_iterations + 1):
        mapped_by_state, ages = simulate_map(
            mixture, likelihood, displacements, noise_sigma, age_lambda,
            samples_per_state, max_age=max_age, seed=seed + 1, stratified=stratified,
        )
        mapped = mapped_by_state.reshape(-1, 2)
        current = sample_gmm(mixture, len(mapped), np.random.default_rng(seed + 2))
        current_reference = sample_gmm(mixture, len(mapped), np.random.default_rng(seed + 5))
        residual = sliced_wasserstein(current, mapped, seed=seed + 3)
        sampling_baseline = sliced_wasserstein(current, current_reference, seed=seed + 3)
        excess_residual = max(0.0, residual - sampling_baseline)
        current_mean, current_covariance = gmm_moments(mixture)
        mapped_mean, mapped_covariance = empirical_moments(mapped)
        row = {
            "iteration": iteration,
            "residual": residual,
            "sampling_baseline": sampling_baseline,
            "excess_residual": excess_residual,
            "mean_gap": float(np.linalg.norm(current_mean - mapped_mean)),
            "covariance_gap": float(np.linalg.norm(current_covariance - mapped_covariance)),
            "trace": float(np.trace(current_covariance)),
            "mapped_trace": float(np.trace(mapped_covariance)),
            "mean_age": float(ages.mean()),
        }
        history.append(row)

        if verbose:
            print(
                f"iter {iteration:02d} | sliced W={residual:.4f} "
                f"(baseline={sampling_baseline:.4f}) "
                f"| trace={row['trace']:.3f} -> {row['mapped_trace']:.3f}"
            )
        converged = (
            excess_residual < residual_tolerance
            and row["mean_gap"] < residual_tolerance
            and row["covariance_gap"] < residual_tolerance
        )
        stable_iterations = stable_iterations + 1 if converged else 0
        if stable_iterations >= patience or iteration == max_iterations:
            return mixture, history, mapped_by_state

        target, _ = fit_gmm(
            mapped, n_components, initial=mixture, min_variance=min_variance,
            max_iterations=em_max_iterations, seed=seed + 4,
        )
        mixture = damp_gmm(
            mixture, align_components(mixture, target), damping, min_variance
        )

    return mixture, history, mapped_by_state


def simplex_frame():
    """Return symmetric 2D simplex vertices."""
    return np.array([
        [0.0, 1.0],
        [np.sqrt(3) / 2, -0.5],
        [-np.sqrt(3) / 2, -0.5],
    ])


def logbelief_to_probability(values, basis=None):
    """Convert 2D log-belief coordinates to state probabilities."""
    basis = default_basis() if basis is None else np.asarray(basis, dtype=float)
    log_beliefs = np.asarray(values) @ basis.T
    probabilities = np.exp(log_beliefs - log_beliefs.max(axis=-1, keepdims=True))
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def probability_to_simplex(probabilities):
    """Map state probabilities to symmetric simplex coordinates."""
    return np.asarray(probabilities) @ simplex_frame()


def gmm_quasi_samples(mixture, power=14, seed=0):
    """Return low-discrepancy GMM points with deterministic masses."""
    uniforms = qmc.Sobol(2, scramble=True, seed=seed).random_base2(power)
    standard = ndtri(np.clip(uniforms, 1e-12, 1 - 1e-12))
    points = []
    masses = []
    for weight, mean, covariance in zip(
            mixture.weights, mixture.means, mixture.covariances):
        points.append(mean + standard @ np.linalg.cholesky(covariance).T)
        masses.append(np.full(len(standard), weight / len(standard)))
    return np.vstack(points), np.concatenate(masses)


def simplex_histogram(mixture, n_bins=50, qmc_power=14, basis=None):
    """Integrate GMM mass into equal-area simplex bins."""
    points, masses = gmm_quasi_samples(mixture, qmc_power)
    xy = probability_to_simplex(logbelief_to_probability(points, basis))
    x_edges = np.linspace(-np.sqrt(3) / 2, np.sqrt(3) / 2, n_bins + 1)
    y_edges = np.linspace(-0.5, 1.0, n_bins + 1)
    histogram, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=[x_edges, y_edges], weights=masses
    )
    return histogram.T, x_edges, y_edges
