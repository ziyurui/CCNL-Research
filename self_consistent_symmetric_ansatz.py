"""Analytical symmetric Gaussian ansatz for the self-consistent noisy-belief prior.

Implements step 1 ("Analytical") of CLAUDE.md: each state-conditioned prior p_s
is a single Gaussian, p = (1/3) sum_s p_s, and the ansatz respects the sickness
swap symmetry (Sickness 1 <-> Sickness 2 together with Cough <-> Scratch).
Everything is written in a symmetry-adapted orthonormal basis of the zero-mean
hyperplane H, in which the swap acts as the reflection z1 -> -z1:

    z1 = (l_S1 - l_S2) / sqrt(2)      (sickness log-odds axis, flips under swap)
    z2 = (2 l_H - l_S1 - l_S2) / sqrt(6)  (healthy-vs-sick axis, invariant)

For a Gaussian-mixture prior the denoiser F_p is a softmax-weighted average of
affine maps, so the self-consistency condition (Eq. 37 of the notes, geometric
age distribution with parameter lambda) reduces to moment equations for the
component means and covariances. These are solved by damped fixed-point
iteration with continuation in the internal-noise scale sigma (Sigma =
sigma^2 I on H). Bifurcation diagnostics: the Jacobian DF_p = Cov_p(L|Y) / sigma^2
at the inter-sickness midpoint, fixed points of the deterministic skeleton
l -> F_p(l + Delta_s), and the symmetry-split spectrum of the linearized
moment map.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

STATES = ["Healthy", "Sickness 1", "Sickness 2"]
SIGNALS = ["Whistle", "Cough", "Scratch"]
REFLECTION = np.diag([-1.0, 1.0])
SWAP = [0, 2, 1]


def symmetry_basis():
    """Return an orthonormal basis of H in which the sickness swap is z1 -> -z1."""
    b1 = np.array([0.0, 1.0, -1.0]) / np.sqrt(2)
    b2 = np.array([2.0, -1.0, -1.0]) / np.sqrt(6)
    return np.column_stack([b1, b2])


def signal_displacements(likelihood, basis=None):
    """Project the log-likelihood displacements delta_x into 2D coordinates."""
    likelihood = np.asarray(likelihood, dtype=float)
    if likelihood.shape != (3, 3) or np.any(likelihood <= 0):
        raise ValueError("likelihood must be a positive 3 x 3 array")
    if not np.allclose(likelihood.sum(axis=0), 1.0):
        raise ValueError("each likelihood column must sum to one")
    basis = symmetry_basis() if basis is None else np.asarray(basis, dtype=float)
    centered_log = np.log(likelihood) - np.log(likelihood).mean(axis=1, keepdims=True)
    return centered_log @ basis


def state_drifts(likelihood, displacements):
    """Return the expected displacements Delta_s = E[delta_x | S=s]."""
    return np.asarray(likelihood, dtype=float).T @ displacements


def state_directions(basis=None):
    """Return unit rays in H pointing toward each state vertex (for plots)."""
    basis = symmetry_basis() if basis is None else np.asarray(basis, dtype=float)
    rays = (np.eye(3) - 1.0 / 3.0) @ basis
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def is_swap_symmetric(likelihood, tolerance=1e-12):
    """Check invariance of the likelihood under the S1<->S2, Cough<->Scratch swap."""
    likelihood = np.asarray(likelihood, dtype=float)
    return bool(np.max(np.abs(likelihood[np.ix_(SWAP, SWAP)] - likelihood)) < tolerance)


@dataclass
class Ansatz:
    """Gaussian ansatz p_s = N(means[s], covariances[s]) with weights 1/3 each."""

    means: np.ndarray        # (3, 2), states ordered [Healthy, Sickness 1, Sickness 2]
    covariances: np.ndarray  # (3, 2, 2)

    def copy(self):
        return Ansatz(self.means.copy(), self.covariances.copy())


def swap_reflect(ansatz):
    """Apply the sickness-swap symmetry to the ansatz parameters."""
    means = (ansatz.means @ REFLECTION)[SWAP]
    covariances = (REFLECTION @ ansatz.covariances @ REFLECTION)[SWAP]
    return Ansatz(means, covariances)


def symmetrize(ansatz, shared_covariance=True):
    """Project onto the swap-symmetric (optionally shared-covariance) family."""
    mirrored = swap_reflect(ansatz)
    means = 0.5 * (ansatz.means + mirrored.means)
    covariances = 0.5 * (ansatz.covariances + mirrored.covariances)
    if shared_covariance:
        covariances = np.repeat(covariances.mean(axis=0)[None], 3, axis=0)
    return Ansatz(means, covariances)


def free_parameters(ansatz):
    """Read out the symmetric-family parameters (h, a, b, c1, c2)."""
    return {
        "h": float(ansatz.means[0, 1]),
        "a": float(ansatz.means[1, 0]),
        "b": float(ansatz.means[1, 1]),
        "c1": float(ansatz.covariances[0, 0, 0]),
        "c2": float(ansatz.covariances[0, 1, 1]),
    }


def mixture_moments(ansatz):
    """Return the mean and covariance of the marginal p = (1/3) sum_s p_s."""
    mean = ansatz.means.mean(axis=0)
    centered = ansatz.means - mean
    covariance = ansatz.covariances.mean(axis=0) + (centered.T @ centered) / 3.0
    return mean, covariance


def _component_logpdf(values, means, covariances):
    values = np.atleast_2d(np.asarray(values, dtype=float))
    output = np.empty((len(values), len(means)))
    for k, (mean, covariance) in enumerate(zip(means, covariances)):
        chol = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(chol, (values - mean).T).T
        output[:, k] = -(np.log(2 * np.pi) + np.log(np.diag(chol)).sum()
                         + 0.5 * np.sum(whitened**2, axis=1))
    return output


def ansatz_logpdf(values, ansatz):
    """Evaluate log p for the marginal three-component mixture."""
    component = _component_logpdf(values, ansatz.means, ansatz.covariances)
    return logsumexp(component, axis=1) - np.log(3.0)


def _responsibilities(values, ansatz, sigma):
    noisy_covariances = ansatz.covariances + sigma**2 * np.eye(2)
    log_joint = _component_logpdf(values, ansatz.means, noisy_covariances)
    return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True)), noisy_covariances


def denoise(values, ansatz, sigma):
    """Posterior-mean denoiser F_p: a softmax-weighted average of affine maps."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    responsibilities, noisy_covariances = _responsibilities(values, ansatz, sigma)
    output = np.zeros_like(values)
    for k in range(3):
        gain = np.linalg.solve(noisy_covariances[k], ansatz.covariances[k]).T
        posterior_mean = ansatz.means[k] + (values - ansatz.means[k]) @ gain.T
        output += responsibilities[:, k, None] * posterior_mean
    return output


def denoiser_jacobian(values, ansatz, sigma):
    """Return DF_p(y) = Cov_p(L | Y=y) / sigma^2, one 2x2 matrix per input point."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    responsibilities, noisy_covariances = _responsibilities(values, ansatz, sigma)
    first = np.zeros((len(values), 2))
    second = np.zeros((len(values), 2, 2))
    for k in range(3):
        gain = np.linalg.solve(noisy_covariances[k], ansatz.covariances[k]).T
        posterior_covariance = sigma**2 * gain
        posterior_mean = ansatz.means[k] + (values - ansatz.means[k]) @ gain.T
        first += responsibilities[:, k, None] * posterior_mean
        second += responsibilities[:, k, None, None] * (
            posterior_covariance
            + posterior_mean[:, :, None] * posterior_mean[:, None, :]
        )
    conditional_covariance = second - first[:, :, None] * first[:, None, :]
    return conditional_covariance / sigma**2


def bayes_ansatz(likelihood, displacements, age_lambda, shared_covariance=True):
    """Moment-match the noise-free Bayes accumulation (the sigma = 0 base case)."""
    likelihood = np.asarray(likelihood, dtype=float)
    drifts = state_drifts(likelihood, displacements)
    mean_age = 1.0 / (1.0 - age_lambda)
    var_age = age_lambda / (1.0 - age_lambda) ** 2
    covariances = np.empty((3, 2, 2))
    for state in range(3):
        centered = displacements - drifts[state]
        step_covariance = (likelihood[:, state, None] * centered).T @ centered
        covariances[state] = (mean_age * step_covariance
                              + var_age * np.outer(drifts[state], drifts[state]))
    return symmetrize(Ansatz(mean_age * drifts, covariances), shared_covariance)


def gauss_hermite_2d(order):
    """Return 2D Gauss-Hermite nodes and weights for a standard normal."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    weights = weights / weights.sum()
    grid = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
    return grid, np.outer(weights, weights).ravel()


def _input_mixture(ansatz, likelihood, displacements, sigma, age_lambda, state):
    """Gaussian-mixture law of the denoiser input Y_s = Lhat'_s + delta_x + eps."""
    noise = sigma**2 * np.eye(2)
    weights, means, covariances = [], [], []
    for x in range(3):
        weights.append((1.0 - age_lambda) * likelihood[x, state])
        means.append(displacements[x])
        covariances.append(noise)
        weights.append(age_lambda * likelihood[x, state])
        means.append(ansatz.means[state] + displacements[x])
        covariances.append(ansatz.covariances[state] + noise)
    return np.array(weights), np.array(means), np.array(covariances)


def mapped_moments(ansatz, likelihood, displacements, sigma, age_lambda, order=24):
    """Apply the self-consistency map once; return raw per-state moments."""
    grid, grid_weights = gauss_hermite_2d(order)
    means = np.empty((3, 2))
    covariances = np.empty((3, 2, 2))
    for state in range(3):
        weights, input_means, input_covariances = _input_mixture(
            ansatz, likelihood, displacements, sigma, age_lambda, state)
        first = np.zeros(2)
        second = np.zeros((2, 2))
        for weight, mean, covariance in zip(weights, input_means, input_covariances):
            nodes = mean + grid @ np.linalg.cholesky(covariance).T
            mapped = denoise(nodes, ansatz, sigma)
            first += weight * grid_weights @ mapped
            second += weight * (mapped.T * grid_weights) @ mapped
        means[state] = first
        covariances[state] = second - np.outer(first, first)
    return means, covariances


def moment_map(ansatz, likelihood, displacements, sigma, age_lambda, order=24,
               symmetric=True, shared_covariance=True):
    """One application of the (optionally projected) self-consistency map."""
    means, covariances = mapped_moments(
        ansatz, likelihood, displacements, sigma, age_lambda, order)
    mapped = Ansatz(means, covariances)
    return symmetrize(mapped, shared_covariance) if symmetric else mapped


def solve_ansatz(likelihood, sigma, age_lambda, init=None, damping=0.6, order=24,
                 tolerance=1e-9, max_iterations=500, symmetric=True,
                 shared_covariance=True, verbose=False):
    """Solve the moment equations by damped fixed-point iteration."""
    likelihood = np.asarray(likelihood, dtype=float)
    if symmetric and not is_swap_symmetric(likelihood):
        raise ValueError("symmetric ansatz requires a swap-symmetric likelihood")
    displacements = signal_displacements(likelihood)
    ansatz = (bayes_ansatz(likelihood, displacements, age_lambda, shared_covariance)
              if init is None else init.copy())
    if symmetric:
        ansatz = symmetrize(ansatz, shared_covariance)
    history = []
    for iteration in range(1, max_iterations + 1):
        mapped = moment_map(ansatz, likelihood, displacements, sigma, age_lambda,
                            order, symmetric, shared_covariance)
        residual = max(np.max(np.abs(mapped.means - ansatz.means)),
                       np.max(np.abs(mapped.covariances - ansatz.covariances)))
        history.append(residual)
        ansatz = Ansatz(
            (1 - damping) * ansatz.means + damping * mapped.means,
            (1 - damping) * ansatz.covariances + damping * mapped.covariances,
        )
        if verbose and (iteration % 25 == 0 or residual < tolerance):
            print(f"iter {iteration:03d} | residual {residual:.3e}")
        if residual < tolerance or not np.isfinite(residual):
            break
    return ansatz, history


def _pack(ansatz):
    triangle = ansatz.covariances[:, [0, 0, 1], [0, 1, 1]]
    return np.concatenate([ansatz.means.ravel(), triangle.ravel()])


def _unpack(vector):
    means = vector[:6].reshape(3, 2)
    triangle = vector[6:].reshape(3, 3)
    return Ansatz(means, triangle[:, [0, 1, 1, 2]].reshape(3, 2, 2))


def moment_map_jacobian(ansatz, likelihood, displacements, sigma, age_lambda,
                        order=24, step=1e-6):
    """Central-difference Jacobian of the raw (unprojected) moment map."""

    def apply(vector):
        mapped = moment_map(_unpack(vector), likelihood, displacements, sigma,
                            age_lambda, order, symmetric=False)
        return _pack(mapped)

    base = _pack(ansatz)
    jacobian = np.empty((base.size, base.size))
    for index in range(base.size):
        forward, backward = base.copy(), base.copy()
        forward[index] += step
        backward[index] -= step
        jacobian[:, index] = (apply(forward) - apply(backward)) / (2 * step)
    return jacobian


def symmetry_split_spectrum(jacobian):
    """Split the moment-map Jacobian spectrum into swap-symmetric and antisymmetric
    blocks (valid at a swap-symmetric fixed point, where the map commutes with the
    swap). Returns (symmetric eigenvalues, antisymmetric eigenvalues)."""
    size = jacobian.shape[0]
    swap_matrix = np.column_stack(
        [_pack(swap_reflect(_unpack(vector))) for vector in np.eye(size)])
    eigenvalues, eigenvectors = np.linalg.eigh(swap_matrix)
    symmetric_block = eigenvectors[:, eigenvalues > 0]
    antisymmetric_block = eigenvectors[:, eigenvalues < 0]
    return (np.linalg.eigvals(symmetric_block.T @ jacobian @ symmetric_block),
            np.linalg.eigvals(antisymmetric_block.T @ jacobian @ antisymmetric_block))


def midpoint_gain(ansatz, sigma):
    """Top eigenvalue of DF_p at the midpoint between the two sickness components."""
    midpoint = 0.5 * (ansatz.means[1] + ansatz.means[2])
    return float(np.linalg.eigvalsh(
        denoiser_jacobian(midpoint[None], ansatz, sigma)[0])[-1])


def skeleton_fixed_points(ansatz, sigma, drift, seeds=None, tolerance=1e-11,
                          max_iterations=200):
    """Find fixed points of the deterministic skeleton l -> F_p(l + Delta_s) by
    Newton iteration from several seeds. Returns (points, eigenvalue pairs of
    DF_p there); a fixed point is stable when its top eigenvalue is below 1."""
    if seeds is None:
        pair_midpoints = 0.5 * (ansatz.means[[0, 0, 1]] + ansatz.means[[1, 2, 2]])
        seeds = np.vstack([ansatz.means, pair_midpoints, np.zeros(2),
                           ansatz.means.mean(axis=0)])
    points, gains = [], []
    for seed in np.atleast_2d(seeds):
        point = np.array(seed, dtype=float)
        converged = False
        for _ in range(max_iterations):
            query = point[None] + drift
            value = denoise(query, ansatz, sigma)[0]
            jacobian = denoiser_jacobian(query, ansatz, sigma)[0]
            try:
                update = np.linalg.solve(jacobian - np.eye(2), point - value)
            except np.linalg.LinAlgError:
                break
            point = point + update
            if not np.all(np.isfinite(point)) or np.linalg.norm(point) > 1e6:
                break
            if np.linalg.norm(update) < tolerance:
                converged = True
                break
        if not converged:
            continue
        if any(np.linalg.norm(point - existing) < 1e-6 for existing in points):
            continue
        points.append(point)
        gains.append(np.linalg.eigvalsh(
            denoiser_jacobian(point[None] + drift, ansatz, sigma)[0]))
    order = np.argsort([p[0] for p in points]) if points else []
    return np.array(points)[order], np.array(gains)[order]


def midline_map(z1_values, ansatz, sigma, drift, height=None):
    """Reduced skeleton slice along the sickness log-odds axis: the z1 component
    of F_p((z1, height) + Delta_s), evaluated at the sickness-cluster height."""
    z1_values = np.asarray(z1_values, dtype=float)
    height = ansatz.means[1, 1] if height is None else height
    queries = np.column_stack([z1_values, np.full_like(z1_values, height)]) + drift
    return denoise(queries, ansatz, sigma)[:, 0]


def midline_fixed_points(ansatz, sigma, drift, span=15.0, samples=1201, height=None):
    """Fixed points of the reduced midline map with their slopes (stable when
    |slope| < 1), found by bisection on a sign-change scan."""
    z1 = np.linspace(-span, span, samples)
    values = midline_map(z1, ansatz, sigma, drift, height) - z1
    roots, slopes = [], []
    for index in np.flatnonzero(np.sign(values[1:]) != np.sign(values[:-1])):
        low, high = z1[index], z1[index + 1]
        for _ in range(60):
            middle = 0.5 * (low + high)
            value = midline_map([middle], ansatz, sigma, drift, height)[0] - middle
            if np.sign(value) == np.sign(values[index]):
                low = middle
            else:
                high = middle
        root = 0.5 * (low + high)
        if roots and abs(root - roots[-1]) < 1e-3:
            continue
        step = 1e-5
        slope = (midline_map([root + step], ansatz, sigma, drift, height)[0]
                 - midline_map([root - step], ansatz, sigma, drift, height)[0]) / (2 * step)
        roots.append(root)
        slopes.append(slope)
    return np.array(roots), np.array(slopes)


def bifurcation_scan(likelihood, sigmas, age_lambda, order=24, damping=0.6,
                     tolerance=1e-9, max_iterations=500, jacobian_step=1e-6,
                     init=None, verbose=True):
    """Continuation in sigma along the symmetric branch with per-sigma diagnostics."""
    likelihood = np.asarray(likelihood, dtype=float)
    displacements = signal_displacements(likelihood)
    drifts = state_drifts(likelihood, displacements)
    ansatz = init
    records = []
    for sigma in sigmas:
        ansatz, history = solve_ansatz(
            likelihood, sigma, age_lambda, init=ansatz, damping=damping,
            order=order, tolerance=tolerance, max_iterations=max_iterations)
        jacobian = moment_map_jacobian(ansatz, likelihood, displacements, sigma,
                                       age_lambda, order, jacobian_step)
        symmetric_eigs, antisymmetric_eigs = symmetry_split_spectrum(jacobian)
        healthy_points, healthy_gains = skeleton_fixed_points(ansatz, sigma, drifts[0])
        sick1_points, sick1_gains = skeleton_fixed_points(ansatz, sigma, drifts[1])
        _, marginal_covariance = mixture_moments(ansatz)
        record = {
            "sigma": float(sigma),
            "ansatz": ansatz.copy(),
            "residual": history[-1],
            "iterations": len(history),
            "params": free_parameters(ansatz),
            "marginal_trace": float(np.trace(marginal_covariance)),
            "midpoint_gain": midpoint_gain(ansatz, sigma),
            "symmetric_radius": float(np.max(np.abs(symmetric_eigs))),
            "antisymmetric_radius": float(np.max(np.abs(antisymmetric_eigs))),
            "healthy_fixed_points": healthy_points,
            "healthy_gains": healthy_gains,
            "sickness1_fixed_points": sick1_points,
            "sickness1_gains": sick1_gains,
        }
        records.append(record)
        if verbose:
            print(f"sigma {sigma:6.3f} | iters {record['iterations']:3d} "
                  f"| midpoint gain {record['midpoint_gain']:7.3f} "
                  f"| antisym radius {record['antisymmetric_radius']:6.3f} "
                  f"| healthy FPs {len(healthy_points)}")
    return records
