"""
verify.py -- numerical checks of every closed-form claim in the note.
Each check prints PASS/FAIL with the max error it saw.
"""
import numpy as np
from scipy.stats import vonmises
from scipy.special import logsumexp
from circbayes import *

rng = np.random.default_rng(0)
results = []


def check(name, err, tol):
    ok = err < tol
    results.append((name, err, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<62s} err={err:.2e}")


# 1. frame identities -------------------------------------------------------
check("sum_k e^{i mu_k} = 0", np.abs(E.sum(0)).max(), 1e-12)
check("E^T E = (3/2) I", np.abs(E.T @ E - 1.5 * np.eye(2)).max(), 1e-12)
check("E E^T = (3/2) P", np.abs(E @ E.T - 1.5 * P).max(), 1e-12)

# 2. sequential logits == brute-force Bayes ---------------------------------
kappa, N, T = 1.7, 200, 25
pis = rng.dirichlet(np.ones(3), size=N)
_, x = sample_data(rng, N, T, kappa)
eta = clean_eta(x, kappa)
post_vec = softmax(logits(eta, np.log(pis)[:, None, :]))
ll = np.stack([vonmises.logpdf(x, kappa, loc=m) for m in MU], -1)  # (N,T,3)
brute = softmax(np.log(pis)[:, None, :] + np.cumsum(ll, 1))
check("softmax(log pi + E eta_t) == brute-force posterior", np.abs(post_vec - brute).max(), 1e-10)

# 3. worked example --------------------------------------------------------
S2 = np.exp(1j * np.deg2rad(30)) + np.exp(1j * np.deg2rad(150))
check("worked example: S_2 = i", abs(S2 - 1j), 1e-12)
lg = E @ np.array([S2.real, S2.imag])
check("worked example: logits/kappa = (0, .866, -.866)",
      np.abs(lg - np.array([0, np.sqrt(3) / 2, -np.sqrt(3) / 2])).max(), 1e-12)
sep = rng.uniform(0, 2 * np.pi, 1000)
check("|S| = 2|cos(Delta/2)| for two points",
      np.abs(np.abs(1 + np.exp(1j * sep)) - 2 * np.abs(np.cos(sep / 2))).max(), 1e-12)

# 4. logit-noise <-> eta-noise equivalence ----------------------------------
sig_l = 0.9
M = 400_000
z = rng.standard_normal((M, 3)) @ (sig_l * P)          # N(0, sig_l^2 P)  (P idempotent)
eps_eta = z @ np.linalg.pinv(E).T                     # pull back: E^+ eps_l
cov = np.cov(eps_eta.T)
check("eta-noise cov = sig_l^2/(3/2) I  (rel err, MC 4e5)",
      np.abs(cov / (sig_l**2 / 1.5) - np.eye(2)).max(), 1e-2)
check("E E^+ eps_l == eps_l (noise lies in the sum-zero plane)",
      np.abs(eps_eta @ E.T - z).max(), 1e-10)

# 5. posterior predictive = 3-component VM mixture -----------------------
# compare to brute-force ratio p(x_{1:t+1}) / p(x_{1:t}) with flat prior
pi0 = np.full(3, 1 / 3)
xn = rng.uniform(-np.pi, np.pi, size=N)
lp_t = logsumexp(np.log(pi0) + ll[:, :10].sum(1), -1)
ll_next = np.stack([vonmises.logpdf(xn, kappa, loc=m) for m in MU], -1)
lp_t1 = logsumexp(np.log(pi0) + ll[:, :10].sum(1) + ll_next, -1)
pred_brute = np.exp(lp_t1 - lp_t)
w10 = softmax(logits(eta[:, 9], np.log(pi0)))
pred_mix = (w10 * np.exp(ll_next)).sum(-1)
check("posterior predictive == sum_k softmax(l_t)_k VM(x; mu_k, kappa)",
      np.abs(pred_mix - pred_brute).max(), 1e-10)

# 6. per-step noise accumulates to t sigma^2 --------------------------------
sigma, N2, T2 = 0.7, 200_000, 12
_, x2 = sample_data(rng, N2, T2, kappa)
et, tau2 = noisy_eta(x2, kappa, sigma, rng, "perstep")
emp = (et - clean_eta(x2, kappa)).var(0).mean(-1)
check("per-step noise: Var(eta_tilde_t - eta_t) = t sigma^2 (rel)",
      np.abs(emp / tau2 - 1).max(), 2e-2)

# 7. CLT mixture moments of eta_t | k ---------------------------------------
t = 15
k = np.zeros(200_000, int)
_, x3 = sample_data(rng, len(k), t, kappa, k=k)
eta_t = clean_eta(x3, kappa)[:, -1]
means, covs = component_moments(t, kappa)
check("E[eta_t|k=0] matches kappa t A1 e_0 (rel)",
      np.abs(eta_t.mean(0) - means[0]).max() / np.abs(means[0]).max(), 1e-2)
check("Cov[eta_t|k=0] matches CLT anisotropic covariance (rel)",
      np.abs(np.cov(eta_t.T) - covs[0]).max() / covs[0].max(), 2e-2)
ratio = covs[0][0, 0] / covs[0][1, 1]
print(f"       anisotropy (radial/angular variance) at kappa={kappa}: {ratio:.3f}")

# 8. moment-matched s_t^2 ----------------------------------------------------
_, x4 = sample_data(rng, 200_000, t, kappa)
e4 = clean_eta(x4, kappa)[:, -1]
check("moment-matched s_t^2 = kappa^2(t+t(t-1)A1^2)/2 (rel)",
      abs((e4**2).mean() / moment_matched_s2(t, kappa) - 1), 1e-2)

# 9. Gauss-Hermite E[softmax] vs Monte Carlo -------------------------------
eh = rng.normal(size=(50, 2)) * 3
v = rng.uniform(0.1, 6, size=50)
qgh = q_expected(eh, v, np.log(pi0))
zz = rng.standard_normal((50, 200_000, 2))
qmc = softmax(logits(eh[:, None] + np.sqrt(v)[:, None, None] * zz, np.log(pi0))).mean(1)
check("2D Gauss-Hermite (16x16) E[softmax] vs MC 2e5", np.abs(qgh - qmc).max(), 5e-3)

# 10. 'plug-in is overconfident' ------------------------------------------
eh = rng.normal(size=(100_000, 2)) * rng.uniform(0, 5, size=(100_000, 1))
v = rng.uniform(0.01, 10, size=100_000)
lp = np.log(rng.dirichlet(np.ones(3), size=100_000))
qe, qp = q_expected(eh, v, lp[:, None, :]), q_plugin(eh, lp)
ent = lambda q: -(q * np.log(np.clip(q, 1e-300, 1))).sum(-1)
frac_max = (qe.max(-1) <= qp.max(-1) + 1e-12).mean()
frac_ent = (ent(qe) >= ent(qp) - 1e-12).mean()
print(f"       plug-in max-prob >= expected max-prob in {frac_max:.4%} of 1e5 random cases")
print(f"       expected entropy >= plug-in entropy  in {frac_ent:.4%} of 1e5 random cases")
# find the worst violation of the entropy ordering, if any
d = ent(qp) - ent(qe)
i = d.argmax()
print(f"       worst entropy violation: H(plug)-H(exp) = {d[i]:+.4f} at "
      f"pi={np.exp(lp[i]).round(3)}, |eta_hat|={np.linalg.norm(eh[i]):.2f}, v={v[i]:.2f}")

# 11. Mixture CLT denoiser vs exact Monte-Carlo p(k | eta_tilde) -----------
for t in (1, 3, 10):
    tau2 = t * 0.5**2
    k_true, xs = sample_data(rng, 300, t, kappa)
    etil = clean_eta(xs, kappa)[:, -1] + np.sqrt(tau2) * rng.standard_normal((300, 2))
    _, r_clt = denoise_mixture(etil, t, kappa, tau2, np.log(pi0))
    # exact: p(etil|k) = E_x[N(etil; kappa S_t(x), tau2 I)]
    Mm = 20_000
    lik = np.empty((300, 3))
    for kk in range(3):
        _, xm = sample_data(rng, Mm, t, kappa, k=np.full(Mm, kk))
        Sm = clean_eta(xm, kappa)[:, -1]
        d2 = ((etil[:, None, :] - Sm[None]) ** 2).sum(-1)
        lik[:, kk] = logsumexp(-d2 / (2 * tau2), axis=1)
    r_exact = softmax(lik)
    print(f"       t={t:2d}: CLT-mixture vs exact p(k|eta~): mean |diff| = "
          f"{np.abs(r_clt - r_exact).mean():.4f}, max = {np.abs(r_clt - r_exact).max():.4f}")

print()
print(f"{sum(ok for *_, ok in results)}/{len(results)} checks passed")
