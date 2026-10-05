"""Conditional-GMM solver for the self-consistent noisy-belief prior."""

import numpy as np

from self_consistent_gmm_solver import (
    GaussianMixture,
    align_components,
    combine_gmms,
    damp_gmm,
    default_basis,
    empirical_moments,
    fit_gmm,
    gmm_denoise,
    gmm_moments,
    initial_gmm,
    sample_gmm,
    signal_displacements,
    simulate_map,
    sliced_wasserstein,
)


def has_sickness_exchange_symmetry(likelihood, state_prior, atol=1e-12):
    """Check invariance under exchanging S1/Cough with S2/Scratch."""
    likelihood = np.asarray(likelihood)
    state_prior = np.asarray(state_prior)
    order = [0, 2, 1]
    return bool(
        np.allclose(likelihood, likelihood[np.ix_(order, order)], rtol=0, atol=atol)
        and np.isclose(state_prior[1], state_prior[2], rtol=0, atol=atol)
    )


def _reflect_gmm(mixture, reflection):
    return GaussianMixture(
        mixture.weights.copy(),
        mixture.means @ reflection.T,
        np.array([reflection @ covariance @ reflection.T
                  for covariance in mixture.covariances]),
    )


def simulate_fixed_age_distributions(
        mixture, likelihood, noise_sigma, ages, n_samples=20000,
        state=1, seed=0):
    """Simulate noisy and noise-free beliefs at fixed ages with shared signals."""
    likelihood = np.asarray(likelihood, dtype=float)
    ages = np.asarray(ages, dtype=int)
    if ages.ndim != 1 or len(ages) == 0 or np.any(ages < 1):
        raise ValueError("ages must be a nonempty sequence of positive integers")
    rng = np.random.default_rng(seed)
    displacements = signal_displacements(likelihood)
    noisy = np.zeros((n_samples, 2))
    bayes = np.zeros((n_samples, 2))
    snapshots = {}

    for step in range(1, ages.max() + 1):
        signals = rng.choice(3, n_samples, p=likelihood[:, state])
        increments = displacements[signals]
        noisy = gmm_denoise(
            noisy + increments + rng.normal(0, noise_sigma, noisy.shape),
            mixture, noise_sigma,
        )
        bayes += increments
        if step in ages:
            snapshots[step] = noisy.copy(), bayes.copy()

    return (
        np.stack([snapshots[age][0] for age in ages]),
        np.stack([snapshots[age][1] for age in ages]),
    )


def simulate_fixed_sequence(
        mixture, likelihood, noise_sigma, sequence, n_samples=20000, seed=0):
    """Simulate observers receiving the same ordered signal sequence."""
    sequence = np.asarray(sequence, dtype=int)
    if sequence.ndim != 1 or len(sequence) == 0 or np.any((sequence < 0) | (sequence > 2)):
        raise ValueError("sequence must contain signal indices 0, 1, or 2")
    rng = np.random.default_rng(seed)
    displacements = signal_displacements(likelihood)
    beliefs = np.zeros((n_samples, 2))
    for signal in sequence:
        beliefs = gmm_denoise(
            beliefs + displacements[signal]
            + rng.normal(0, noise_sigma, beliefs.shape),
            mixture, noise_sigma,
        )
    return beliefs


def solve_conditional_gmm_fixed_point(
        likelihood, n_components=5, noise_sigma=0.15, age_lambda=0.9,
        samples_per_state=5000, max_age=80, initial_scale=2.0,
        min_std=0.2, damping=0.5, max_iterations=30,
        em_max_iterations=100, residual_tolerance=0.05,
        moment_tolerance=0.01, patience=3, seed=15,
        stratified=True, verbose=True, state_prior=None):
    """Solve the projected fixed point with one GMM per state."""
    likelihood = np.asarray(likelihood, dtype=float)
    state_prior = (np.full(3, 1 / 3) if state_prior is None
                   else np.asarray(state_prior, dtype=float))
    if (state_prior.shape != (3,) or np.any(state_prior < 0)
            or not np.isclose(state_prior.sum(), 1)):
        raise ValueError("state_prior must be three nonnegative weights summing to one")
    displacements = signal_displacements(likelihood)
    conditionals = [initial_gmm(n_components, initial_scale) for _ in range(3)]
    sickness_symmetry = has_sickness_exchange_symmetry(likelihood, state_prior)
    permutation = np.eye(3)[[0, 2, 1]]
    basis = default_basis()
    reflection = basis.T @ permutation @ basis
    min_variance = min_std**2
    history = []
    stable_iterations = 0

    if verbose:
        print(f"S1/S2 symmetry: {'used' if sickness_symmetry else 'not used'}")

    for iteration in range(1, max_iterations + 1):
        marginal = combine_gmms(conditionals, state_prior)
        mapped, ages = simulate_map(
            marginal, likelihood, displacements, noise_sigma, age_lambda,
            samples_per_state, max_age=max_age, seed=seed + 1,
            stratified=stratified,
        )
        residuals = []
        baselines = []
        relative_mean_gaps = []
        relative_covariance_gaps = []
        for state, conditional in enumerate(conditionals):
            current = sample_gmm(
                conditional, samples_per_state,
                np.random.default_rng(seed + 10 + state),
            )
            reference = sample_gmm(
                conditional, samples_per_state,
                np.random.default_rng(seed + 20 + state),
            )
            residuals.append(sliced_wasserstein(
                current, mapped[state], seed=seed + 30 + state
            ))
            baselines.append(sliced_wasserstein(
                current, reference, seed=seed + 30 + state
            ))
            current_mean, current_covariance = gmm_moments(conditional)
            mapped_mean, mapped_covariance = empirical_moments(mapped[state])
            mean_gap = np.linalg.norm(current_mean - mapped_mean)
            covariance_gap = np.linalg.norm(current_covariance - mapped_covariance)
            relative_mean_gaps.append(mean_gap / np.sqrt(np.trace(current_covariance)))
            relative_covariance_gaps.append(
                covariance_gap / np.linalg.norm(current_covariance)
            )

        excess = np.maximum(np.asarray(residuals) - np.asarray(baselines), 0)
        _, marginal_covariance = gmm_moments(marginal)
        mapped_moments = [empirical_moments(values) for values in mapped]
        mapped_mean = sum(weight * moment[0]
                          for weight, moment in zip(state_prior, mapped_moments))
        mapped_covariance = sum(
            weight * (covariance + np.outer(mean - mapped_mean, mean - mapped_mean))
            for weight, (mean, covariance) in zip(state_prior, mapped_moments)
        )
        row = {
            "iteration": iteration,
            "state_residuals": residuals,
            "state_baselines": baselines,
            "max_excess_residual": float(excess.max()),
            "max_relative_mean_gap": float(np.max(relative_mean_gaps)),
            "max_relative_covariance_gap": float(np.max(relative_covariance_gaps)),
            "trace": float(np.trace(marginal_covariance)),
            "mapped_trace": float(np.trace(mapped_covariance)),
            "mean_age": float(ages.mean()),
            "sickness_symmetry": sickness_symmetry,
        }
        history.append(row)

        if verbose:
            print(
                f"iter {iteration:02d} | max excess W={row['max_excess_residual']:.4f} "
                f"| trace={row['trace']:.3f} -> {row['mapped_trace']:.3f}"
            )
        converged = (
            row["max_excess_residual"] < residual_tolerance
            and row["max_relative_mean_gap"] < moment_tolerance
            and row["max_relative_covariance_gap"] < moment_tolerance
        )
        stable_iterations = stable_iterations + 1 if converged else 0
        if stable_iterations >= patience or iteration == max_iterations:
            return conditionals, marginal, history, mapped

        if sickness_symmetry:
            targets = [
                fit_gmm(
                    mapped[0], n_components, initial=conditionals[0],
                    min_variance=min_variance, max_iterations=em_max_iterations,
                    seed=seed + 40,
                )[0],
                fit_gmm(
                    np.vstack([mapped[1], mapped[2] @ reflection.T]),
                    n_components, initial=conditionals[1],
                    min_variance=min_variance, max_iterations=em_max_iterations,
                    seed=seed + 41,
                )[0],
            ]
            healthy = damp_gmm(
                conditionals[0], align_components(conditionals[0], targets[0]),
                damping, min_variance,
            )
            sickness_1 = damp_gmm(
                conditionals[1], align_components(conditionals[1], targets[1]),
                damping, min_variance,
            )
            conditionals = [healthy, sickness_1,
                            _reflect_gmm(sickness_1, reflection)]
        else:
            targets = [
                fit_gmm(
                    mapped[state], n_components, initial=conditionals[state],
                    min_variance=min_variance, max_iterations=em_max_iterations,
                    seed=seed + 40 + state,
                )[0]
                for state in range(3)
            ]
            conditionals = [
                damp_gmm(
                    conditionals[state],
                    align_components(conditionals[state], targets[state]),
                    damping, min_variance,
                )
                for state in range(3)
            ]

    return conditionals, combine_gmms(conditionals, state_prior), history, mapped
