"""
circbayes.py -- core model for "Sequential Meta-Bayesian Inference With
Continuous Circular Data" (note dated 12 Oct 2026).

Everything is vectorised over a leading batch axis so that 10^4-10^5 trials
run in seconds.  Shapes: data x is (N, T); 2D vectors carry a trailing axis
of length 2.

Stage 1  : exact sequential Bayes, belief vector eta_t = kappa * S_t
Stage 2  : readout noise or per-step (accumulating) Gaussian noise on eta
Stage 3  : denoisers
             - "gauss"   : isotropic Gaussian prior N(m, s^2 I) on eta (the note)
             - "mixture" : the generative model's own prior on eta, i.e. a
                           3-component (CLT) Gaussian mixture with anisotropic
                           covariances -> Bayes-optimal p(k | eta_tilde)
           readouts
             - plug-in  softmax(log pi + E eta_hat)
             - expected E[softmax(log pi + E eta)] by 2D Gauss-Hermite
"""
from __future__ import annotations

import numpy as np
from scipy.special import i0e, i1e, ive, logsumexp

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
MU = np.deg2rad([0.0, 120.0, 240.0])                    # hypothesis centres
E = np.stack([np.cos(MU), np.sin(MU)], axis=1)           # 3x2, rows e_k
P = np.eye(3) - np.ones((3, 3)) / 3                      # sum-zero projector
K = 3


def A1(kappa: float) -> float:
    """Mean resultant length of VM(kappa): I1/I0."""
    return float(i1e(kappa) / i0e(kappa))


def A2(kappa: float) -> float:
    """E[cos 2(x-mu)] under VM(kappa): I2/I0."""
    return float(ive(2, kappa) / i0e(kappa))


def unit(x: np.ndarray) -> np.ndarray:
    """Angle(s) -> unit vectors, trailing axis 2."""
    return np.stack([np.cos(x), np.sin(x)], axis=-1)


def softmax(l: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(l - logsumexp(l, axis=axis, keepdims=True))


def logits(eta: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    """l = log pi + E eta   (eta (...,2) -> (...,3))."""
    return log_pi + eta @ E.T


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def sample_data(rng, N, T, kappa, pi=None, k=None):
    """Draw true hypotheses k ~ pi and x_t ~ VM(mu_k, kappa)."""
    pi = np.full(K, 1 / K) if pi is None else np.asarray(pi)
    if k is None:
        k = rng.choice(K, size=N, p=pi)
    x = rng.vonmises(MU[k][:, None], kappa, size=(N, T))
    return k, x


# --------------------------------------------------------------------------
# Stage 1 + 2: belief trajectories
# --------------------------------------------------------------------------
def clean_eta(x, kappa):
    """eta_t = kappa * sum_{s<=t} e^{i x_s};  (N,T) -> (N,T,2)."""
    return kappa * np.cumsum(unit(x), axis=1)


def noisy_eta(x, kappa, sigma, rng, kind="perstep"):
    """
    kind = "readout": eta_tilde_t = eta_t + eps_t,  eps_t iid N(0, sigma^2 I)
    kind = "perstep": eta_tilde_t = eta_tilde_{t-1} + kappa e^{ix_t} + eps_t
    Returns (eta_tilde, tau2) with tau2[t] the noise variance in force.
    """
    eta = clean_eta(x, kappa)
    N, T = x.shape
    z = sigma * rng.standard_normal((N, T, 2))
    t = np.arange(1, T + 1)
    if kind == "readout":
        return eta + z, np.full(T, sigma**2)
    if kind == "perstep":
        return eta + np.cumsum(z, axis=1), t * sigma**2
    raise ValueError(kind)


# --------------------------------------------------------------------------
# The model's own prior over eta_t (CLT Gaussian mixture)
# --------------------------------------------------------------------------
def component_moments(t, kappa):
    """
    Per-hypothesis mean and covariance of eta_t = kappa S_t given k.
    One datum u = e^{ix}, x ~ VM(mu_k, kappa):
        E u        = A1 e_k
        Var along e_k        = (1 + A2)/2 - A1^2
        Var perpendicular    = (1 - A2)/2
    Returns means (3,2) and covs (3,2,2) for eta_t.
    """
    a1, a2 = A1(kappa), A2(kappa)
    v_par, v_perp = (1 + a2) / 2 - a1**2, (1 - a2) / 2
    means = kappa * t * a1 * E
    covs = np.empty((K, 2, 2))
    for k in range(K):
        R = np.array([[np.cos(MU[k]), -np.sin(MU[k])],
                      [np.sin(MU[k]),  np.cos(MU[k])]])
        covs[k] = kappa**2 * t * R @ np.diag([v_par, v_perp]) @ R.T
    return means, covs


def moment_matched_s2(t, kappa):
    """Per-dimension 2nd moment of eta_t about 0 under a flat prior:
       E|eta_t|^2 / 2 = kappa^2 (t + t(t-1) A1^2) / 2  ~ t^2 (coherent regime)."""
    a1 = A1(kappa)
    return kappa**2 * (t + t * (t - 1) * a1**2) / 2


# --------------------------------------------------------------------------
# Stage 3: denoisers
# --------------------------------------------------------------------------
def denoise_gauss(eta_t, m, s2, tau2):
    """Note's denoiser. eta_hat = m + w (eta_t - m), v = s2 tau2/(s2+tau2)."""
    s2 = np.asarray(s2, float)
    tau2 = np.asarray(tau2, float)
    w = s2 / (s2 + tau2)
    v = s2 * tau2 / (s2 + tau2)
    return m + w[..., None] * (eta_t - m), v, w


def _gauss_logpdf2(x, mean, cov):
    """log N(x; mean, cov) for 2D, x (...,2), cov (2,2) or (...,2,2)."""
    d = x - mean
    det = cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] * cov[..., 1, 0]
    inv00 = cov[..., 1, 1] / det
    inv11 = cov[..., 0, 0] / det
    inv01 = -cov[..., 0, 1] / det
    q = d[..., 0]**2 * inv00 + 2 * d[..., 0] * d[..., 1] * inv01 + d[..., 1]**2 * inv11
    return -0.5 * q - 0.5 * np.log(det) - np.log(2 * np.pi)


def denoise_mixture(eta_t, t, kappa, tau2, log_pi):
    """
    Bayes denoiser under the model's own prior on eta_t.
    Returns (eta_hat, resp) where resp = p(k | eta_tilde) (the Bayes-optimal
    hypothesis posterior given the noisy state) and eta_hat = E[eta | eta_tilde].
    """
    means, covs = component_moments(t, kappa)
    I2 = np.eye(2)
    logr = np.empty(eta_t.shape[:-1] + (K,))
    post_means = np.empty(eta_t.shape[:-1] + (K, 2))
    for k in range(K):
        C = covs[k] + tau2 * I2
        logr[..., k] = log_pi[k] + _gauss_logpdf2(eta_t, means[k], C)
        G = covs[k] @ np.linalg.inv(C)                      # gain
        post_means[..., k, :] = means[k] + (eta_t - means[k]) @ G.T
    resp = softmax(logr)
    eta_hat = np.einsum("...k,...kd->...d", resp, post_means)
    return eta_hat, resp


# --------------------------------------------------------------------------
# Readouts
# --------------------------------------------------------------------------
_GH_N = 16
_gh_x, _gh_w = np.polynomial.hermite_e.hermegauss(_GH_N)   # N(0,1) nodes
_gh_w = _gh_w / _gh_w.sum()
GH_Z = np.stack(np.meshgrid(_gh_x, _gh_x, indexing="ij"), -1).reshape(-1, 2)
GH_W = np.outer(_gh_w, _gh_w).reshape(-1)


def q_plugin(eta_hat, log_pi):
    return softmax(logits(eta_hat, log_pi))


def q_expected(eta_hat, v, log_pi):
    """E[softmax(log pi + E eta)], eta ~ N(eta_hat, v I), 2D Gauss-Hermite."""
    v = np.asarray(v, float)
    nodes = eta_hat[..., None, :] + np.sqrt(v)[..., None, None] * GH_Z
    return np.einsum("...gk,g->...k", softmax(logits(nodes, log_pi)), GH_W)


# --------------------------------------------------------------------------
# The full meta-Bayesian loop (Section 4), both choice-point options
# --------------------------------------------------------------------------
def run_loop(x, kappa, sigma, rng, log_pi, mode, denoiser="gauss",
             m=np.zeros(2), s2_rule="moment", s2_fixed=None):
    """
    mode = "readout"   : propagate eta_tilde; denoise only to report q_t,
                         using tau_t^2 = t sigma^2 (per-step noise)
    mode = "propagate" : denoise every step and feed eta_hat back as the state;
                         fresh noise per step so tau^2 = sigma^2 (the note's literal loop)
    denoiser = "gauss" | "mixture"
    Returns q (N,T,3) and the state trajectory (N,T,2).
    """
    N, T = x.shape
    u = kappa * unit(x)
    q = np.empty((N, T, K))
    states = np.empty((N, T, 2))
    state = np.zeros((N, 2))
    for t in range(1, T + 1):
        state = state + u[:, t - 1] + sigma * rng.standard_normal((N, 2))
        tau2 = t * sigma**2 if mode == "readout" else sigma**2
        if denoiser == "gauss":
            s2 = moment_matched_s2(t, kappa) if s2_rule == "moment" else s2_fixed
            eh, v, _ = denoise_gauss(state, m, np.full(N, s2), np.full(N, tau2))
            q[:, t - 1] = q_expected(eh, v, log_pi)
        else:
            eh, r = denoise_mixture(state, t, kappa, tau2, log_pi)
            q[:, t - 1] = r
        if mode == "propagate":
            state = eh
        states[:, t - 1] = state
    return q, states


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def logloss(q, k, eps=1e-12):
    return -np.log(np.clip(np.take_along_axis(q, k[:, None], -1)[:, 0], eps, 1))


def ece(q, k, bins=15):
    """Top-label expected calibration error."""
    conf = q.max(-1)
    correct = (q.argmax(-1) == k).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    out = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            out += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return out
