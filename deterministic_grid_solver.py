"""Deterministic grid solver for the self-consistent prior."""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


def make_grid(limit=8.0, step=0.4, growth=1.0):
    """Create a symmetric grid, optionally coarsened in the tails."""
    if growth == 1:
        n = int(round(2 * limit / step)) + 1
        xs = np.linspace(-limit, limit, n)
    else:
        positive = [0.0]
        increment = step
        while positive[-1] + increment < limit:
            positive.append(positive[-1] + increment)
            increment *= growth
        if len(positive) > 1 and limit - positive[-1] < 0.5 * (positive[-1] - positive[-2]):
            positive[-1] = limit
        else:
            positive.append(limit)
        xs = np.r_[-np.array(positive[:0:-1]), positive]
        n = len(xs)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    widths = np.empty(n)
    widths[1:-1] = 0.5 * (xs[2:] - xs[:-2])
    widths[0] = 0.5 * (xs[1] - xs[0])
    widths[-1] = 0.5 * (xs[-1] - xs[-2])
    edges = np.r_[xs[0], 0.5 * (xs[:-1] + xs[1:]), xs[-1]]
    area = np.outer(widths, widths).ravel()
    return {"xs": xs, "edges": edges, "points": np.column_stack([X.ravel(), Y.ravel()]), "n": n, "area": area}


def normalize(weights):
    """Clip numerical negatives and normalize mass."""
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = weights.sum()
    if total <= 0:
        raise ValueError("Weights must contain positive mass.")
    return weights / total


def add_floor(weights, floor_mass, floor=None):
    """Mix a small uniform floor into a distribution."""
    weights = normalize(weights)
    floor = np.ones_like(weights) if floor is None else floor
    return normalize((1 - floor_mass) * weights + floor_mass * normalize(floor))


def initialize_prior(grid, kind="wide_gaussian", scale=2.0):
    """Initialize a broad grid distribution."""
    points = grid["points"]
    r2 = np.sum(points**2, axis=1)
    if kind == "wide_gaussian":
        weights = np.exp(-0.5 * r2 / scale**2)
    elif kind == "narrow_gaussian":
        weights = np.exp(-0.5 * r2 / (0.5 * scale) ** 2)
    elif kind == "three_bumps":
        centers = np.array([[scale, 0], [-0.5 * scale, 0.85 * scale], [-0.5 * scale, -0.85 * scale]])
        weights = sum(np.exp(-0.5 * np.sum((points - center) ** 2, axis=1) / (0.35 * scale) ** 2) for center in centers)
    else:
        weights = np.ones(len(points))
    return normalize(weights * grid["area"])


def gaussian_matrix(queries, points, sigma, cutoff_sigma=7.0):
    """Build a sparse Gaussian-likelihood matrix for arbitrary queries."""
    tree = cKDTree(points)
    nearest = tree.query(queries, k=1, workers=-1)[0]
    radii = np.sqrt(nearest**2 + (cutoff_sigma * sigma) ** 2)
    neighbors = tree.query_ball_point(queries, radii, workers=-1)
    lengths = np.fromiter((len(x) for x in neighbors), dtype=int, count=len(neighbors))
    rows = np.repeat(np.arange(len(queries)), lengths)
    cols = np.concatenate(neighbors)
    d2 = np.sum((queries[rows] - points[cols]) ** 2, axis=1)
    values = np.exp(-0.5 * (d2 - nearest[rows] ** 2) / sigma**2)
    return csr_matrix((values, (rows, cols)), shape=(len(queries), len(points)))


def prepare_map(grid, z_delta, sigma, quadrature_order=3, cutoff_sigma=7.0):
    """Precompute deterministic update queries and Gaussian likelihoods."""
    points = grid["points"]
    nodes, node_weights = hermgauss(quadrature_order)
    eps_1d = np.sqrt(2) * sigma * nodes
    eps = np.array([(x, y) for y in eps_1d for x in eps_1d])
    quad_weights = np.outer(node_weights, node_weights).ravel() / np.pi

    queries = points[:, None, None, :] + z_delta[None, :, None, :] + eps[None, None, :, :]
    queries = queries.reshape(-1, 2)
    n_quad = len(quad_weights)
    source = np.repeat(np.arange(len(points)), len(z_delta) * n_quad)
    signal = np.tile(np.repeat(np.arange(len(z_delta)), n_quad), len(points))
    noise_weight = np.tile(np.tile(quad_weights, len(z_delta)), len(points))

    return {
        "gaussian": gaussian_matrix(queries, points, sigma, cutoff_sigma),
        "queries": queries,
        "sigma": sigma,
        "source": source,
        "signal": signal,
        "noise_weight": noise_weight,
    }


def denoise_queries(weights, grid, prepared):
    """Evaluate the posterior-mean denoiser at all prepared queries."""
    points = grid["points"]
    gaussian = prepared["gaussian"]
    denominator = gaussian @ weights
    numerator_x = gaussian @ (weights * points[:, 0])
    numerator_y = gaussian @ (weights * points[:, 1])
    output = np.empty((len(denominator), 2))
    valid = denominator > 0
    output[valid] = np.column_stack([numerator_x[valid] / denominator[valid], numerator_y[valid] / denominator[valid]])

    support = np.flatnonzero(weights > 0)
    invalid_rows = np.flatnonzero(~valid)
    for start in range(0, len(invalid_rows), 256):
        rows = invalid_rows[start:start + 256]
        diff = prepared["queries"][rows, None, :] - points[support][None, :, :]
        log_kernel = np.log(weights[support])[None, :] - 0.5 * np.sum(diff**2, axis=2) / prepared["sigma"]**2
        log_kernel -= log_kernel.max(axis=1, keepdims=True)
        kernel = np.exp(log_kernel)
        output[rows] = kernel @ points[support] / kernel.sum(axis=1, keepdims=True)
    return output


def denoise_points(values, weights, grid, sigma, cutoff_sigma=6.0):
    """Evaluate the grid-prior denoiser at arbitrary points."""
    prepared = {
        "gaussian": gaussian_matrix(values, grid["points"], sigma, cutoff_sigma),
        "queries": values,
        "sigma": sigma,
    }
    return denoise_queries(weights, grid, prepared)


def interpolation_targets(values, grid):
    """Return four bilinear target cells and weights for each point."""
    xs = grid["xs"]
    n = grid["n"]
    values = np.clip(values, xs[0], xs[-1])
    ix = np.clip(np.searchsorted(xs, values[:, 0], side="right") - 1, 0, n - 2)
    iy = np.clip(np.searchsorted(xs, values[:, 1], side="right") - 1, 0, n - 2)
    tx = (values[:, 0] - xs[ix]) / (xs[ix + 1] - xs[ix])
    ty = (values[:, 1] - xs[iy]) / (xs[iy + 1] - xs[iy])
    cells = np.column_stack([iy * n + ix, iy * n + ix + 1, (iy + 1) * n + ix, (iy + 1) * n + ix + 1])
    weights = np.column_stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty])
    return cells, weights


def transition_kernel(state, denoised, grid, prepared, likelihood):
    """Build one deterministic state-conditioned transition kernel."""
    cells, cell_weights = interpolation_targets(denoised, grid)
    probability = prepared["noise_weight"] * likelihood[prepared["signal"], state]
    rows = np.repeat(prepared["source"], 4)
    cols = cells.ravel()
    data = (probability[:, None] * cell_weights).ravel()
    size = len(grid["points"])
    return coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()


def geometric_age_distribution(kernel, p0, lam):
    """Average the transition law over positive geometric ages."""
    rhs = np.asarray(p0 @ kernel).ravel()
    system = eye(kernel.shape[0], format="csr") - lam * kernel
    return normalize((1 - lam) * spsolve(system.T.tocsr(), rhs))


def fixed_point_map(weights, grid, prepared, likelihood, lam, p0):
    """Apply the marginal self-consistency map once."""
    denoised = denoise_queries(weights, grid, prepared)
    conditioned = np.vstack([
        geometric_age_distribution(transition_kernel(state, denoised, grid, prepared, likelihood), p0, lam)
        for state in range(likelihood.shape[1])
    ])
    return normalize(conditioned.mean(axis=0)), conditioned


def summarize(weights, grid):
    """Return mean, covariance trace, and boundary mass."""
    points = grid["points"]
    mean = weights @ points
    centered = points - mean
    covariance = centered.T @ (centered * weights[:, None])
    W = weights.reshape(grid["n"], grid["n"])
    boundary_mass = W[0].sum() + W[-1].sum() + W[1:-1, 0].sum() + W[1:-1, -1].sum()
    return {"mean": mean, "covariance": covariance, "trace": float(np.trace(covariance)), "boundary_mass": float(boundary_mass)}


def solve(initial, grid, prepared, likelihood, lam=0.9, damping=0.1, floor_mass=0.0,
          max_iterations=300, tolerance=1e-4, verbose=True):
    """Run damped fixed-point iteration and report the undamped residual."""
    weights = add_floor(initial, floor_mass, grid["area"])
    p0 = np.zeros(len(weights))
    p0[np.argmin(np.sum(grid["points"] ** 2, axis=1))] = 1.0
    history = []

    for iteration in range(1, max_iterations + 1):
        mapped, conditioned = fixed_point_map(weights, grid, prepared, likelihood, lam, p0)
        target = add_floor(mapped, floor_mass, grid["area"])
        raw_residual = float(np.sum(np.abs(mapped - weights)))
        residual = float(np.sum(np.abs(target - weights)))
        row = {"iteration": iteration, "residual": residual, "raw_residual": raw_residual, **summarize(weights, grid)}
        history.append(row)

        if verbose and (iteration == 1 or iteration % 10 == 0 or residual < tolerance):
            print(f"iter {iteration:03d} | residual={residual:.6g} | trace={row['trace']:.4f} | boundary={row['boundary_mass']:.3g}")
        if residual < tolerance:
            return weights, conditioned, history
        weights = normalize((1 - damping) * weights + damping * target)

    mapped, conditioned = fixed_point_map(weights, grid, prepared, likelihood, lam, p0)
    history.append({
        "iteration": max_iterations + 1,
        "residual": float(np.sum(np.abs(add_floor(mapped, floor_mass, grid["area"]) - weights))),
        "raw_residual": float(np.sum(np.abs(mapped - weights))),
        **summarize(weights, grid),
    })
    return weights, conditioned, history
