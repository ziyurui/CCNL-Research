"""
experiments.py -- simulations of the open questions in the note.

  Exp 1  Which denoiser?  raw vs Gaussian-shrinkage (plug-in / expected) vs the
         model's own mixture prior (Bayes-optimal), per-step noise.
  Exp 2  Which w_t regime actually holds? (fixed s^2, s^2 ~ t, s^2 ~ t^2)
  Exp 3  Order dependence: readout-only vs denoise-and-propagate.
  Exp 4  Polarization: pairs of agents, identical data, independent internal
         noise; does meta-inference produce consensus or camps?
  Exp 5  Normative contrary updating with three hypotheses (no noise at all).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from circbayes import *

OUT = "figures"
import os
os.makedirs(OUT, exist_ok=True)

# ---- plotting style (reference palette, light) -----------------------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5,
})

rng = np.random.default_rng(2026)
KAPPA = 1.0
FLAT = np.log(np.full(3, 1 / 3))


def line(ax, x, y, i, label, **kw):
    ax.plot(x, y, color=C[i], label=label, marker="o", **kw)


# ============================================================================
# Exp 1: denoiser comparison under per-step noise
# ============================================================================
print("=" * 78)
print("EXP 1  Denoisers under per-step noise   (kappa=%.1f, flat prior, m=0)" % KAPPA)
print("=" * 78)
N, T = 40_000, 30
k, x = sample_data(rng, N, T, KAPPA)
eta = clean_eta(x, KAPPA)
p_clean = softmax(logits(eta, FLAT))

methods = ["raw plug-in", "Gauss + plug-in", "Gauss + E[softmax]", "mixture (Bayes)"]


def all_q(et, t, tau2):
    """q for every method given noisy state et at time t (1-indexed)."""
    s2 = moment_matched_s2(t, KAPPA)
    eh, v, w = denoise_gauss(et, np.zeros(2), np.full(len(et), s2), np.full(len(et), tau2))
    _, r = denoise_mixture(et, t, KAPPA, tau2, FLAT)
    return [q_plugin(et, FLAT), q_plugin(eh, FLAT), q_expected(eh, v, FLAT), r], w[0]


sigmas = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
tq = 10                                  # evaluate at t = 10
kl_sig = np.zeros((len(sigmas), 4))
ece_sig = np.zeros((len(sigmas), 4))
print(f"\nAt t={tq}:  mean KL(p_clean || q)  [nats]   |  top-label ECE")
print(f"{'sigma':>6} | " + " | ".join(f"{m:>18s}" for m in methods) + " |  w_t")
for i, s in enumerate(sigmas):
    et, tau2 = noisy_eta(x[:, :tq], KAPPA, s, rng, "perstep")
    qs, w = all_q(et[:, -1], tq, tau2[-1])
    kl_sig[i] = [kl(p_clean[:, tq - 1], q).mean() for q in qs]
    ece_sig[i] = [ece(q, k) for q in qs]
    print(f"{s:6.2f} | " + " | ".join(f"{a:8.4f} / {b:6.3f}  " for a, b in zip(kl_sig[i], ece_sig[i]))
          + f" | {w:.3f}")
print(f"(clean posterior ECE for reference: {ece(p_clean[:, tq-1], k):.4f})")

# KL vs t at sigma = 1.5
s_fix = 1.5
et, tau2 = noisy_eta(x, KAPPA, s_fix, rng, "perstep")
ts = np.arange(1, T + 1)
kl_t = np.zeros((T, 4))
w_t = np.zeros(T)
for t in ts:
    qs, w_t[t - 1] = all_q(et[:, t - 1], t, tau2[t - 1])
    kl_t[t - 1] = [kl(p_clean[:, t - 1], q).mean() for q in qs]
print(f"\nsigma={s_fix}: KL at t=1,5,10,20,30:")
for j, m in enumerate(methods):
    print(f"   {m:>20s}: " + "  ".join(f"{kl_t[t-1, j]:.4f}" for t in (1, 5, 10, 20, 30)))
print("   w_t (Gauss, moment-matched s_t^2): " + "  ".join(f"{w_t[t-1]:.3f}" for t in (1, 5, 10, 20, 30)))

# Reliability diagram at sigma=1.5, t=10
et10, tau10 = noisy_eta(x[:, :10], KAPPA, s_fix, rng, "perstep")
qs10, _ = all_q(et10[:, -1], 10, tau10[-1])

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axs[0]
for j, m in enumerate(methods):
    line(ax, sigmas, kl_sig[:, j], j, m)
ax.set_xlabel("per-step noise σ"); ax.set_ylabel("mean KL(p_clean ‖ q)  [nats]")
ax.set_title(f"Distance to the noise-free posterior, t = {tq}")
ax.legend(loc="upper left")
ax = axs[1]
for j, m in enumerate(methods):
    ax.plot(ts, kl_t[:, j], color=C[j], label=m)
ax.set_xlabel("t (observations)"); ax.set_ylabel("mean KL(p_clean ‖ q)  [nats]")
ax.set_title(f"Over time, σ = {s_fix}")
ax = axs[2]
edges = np.linspace(1 / 3, 1, 11)
for j, (m, q) in enumerate(zip(methods, qs10)):
    conf, corr = q.max(-1), (q.argmax(-1) == k)
    idx = np.digitize(conf, edges) - 1
    xs_, ys_ = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() > 200:
            xs_.append(conf[sel].mean()); ys_.append(corr[sel].mean())
    ax.plot(xs_, ys_, color=C[j], marker="o", label=m)
ax.plot([1 / 3, 1], [1 / 3, 1], color=INK2, lw=1, ls="--")
ax.set_xlabel("stated confidence (max q)"); ax.set_ylabel("empirical accuracy")
ax.set_title(f"Calibration, σ = {s_fix}, t = 10")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_denoisers.png", dpi=160); plt.close(fig)

# ============================================================================
# Exp 2: which w_t regime holds?
# ============================================================================
print("\n" + "=" * 78)
print("EXP 2  Growth of E|eta_t|^2 : coherent (model) data vs incoherent (uniform) data")
print("=" * 78)
_, xc = sample_data(rng, 50_000, 200, KAPPA)
xu = rng.uniform(-np.pi, np.pi, (50_000, 200))
for name, xx in (("coherent", xc), ("incoherent", xu)):
    m2 = (clean_eta(xx, KAPPA) ** 2).sum(-1).mean(0) / 2
    tt = np.arange(1, 201)
    slope = np.polyfit(np.log(tt[20:]), np.log(m2[20:]), 1)[0]
    print(f"  {name:>10s}: log-log slope of s_t^2 over t in [21,200] = {slope:.3f}")
print("  => model-generated data are in the s^2 ~ t^2 regime: w_t -> 1 under readout-only;")
print("     only data with no net direction give s^2 ~ t (constant w).")

# ============================================================================
# Exp 3: order dependence
# ============================================================================
print("\n" + "=" * 78)
print("EXP 3  Order effects: same multiset of data, two orders")
print("=" * 78)
N3, half = 20_000, 10
sigma3 = 1.0
xa = rng.vonmises(MU[0], KAPPA, (N3, half))     # H0-favouring block
xb = rng.vonmises(MU[1], KAPPA, (N3, half))     # H1-favouring block
order_AB = np.concatenate([xa, xb], 1)          # H0 evidence first
order_BA = np.concatenate([xb, xa], 1)          # H1 evidence first
T3 = 2 * half
configs = [
    ("readout-only, Gauss", dict(mode="readout", denoiser="gauss")),
    ("readout-only, mixture", dict(mode="readout", denoiser="mixture")),
    ("propagate, Gauss s_t^2 moment", dict(mode="propagate", denoiser="gauss")),
    ("propagate, Gauss s^2 fixed=4", dict(mode="propagate", denoiser="gauss",
                                          s2_rule="fixed", s2_fixed=4.0)),
    ("propagate, mixture", dict(mode="propagate", denoiser="mixture")),
]
p_cl = softmax(logits(clean_eta(order_AB, KAPPA)[:, -1], FLAT))
print(f"  clean Bayes (both orders identical): mean q0-q1 = {(p_cl[:,0]-p_cl[:,1]).mean():+.4f}")
order_rows = []
for name, cfg in configs:
    seed = rng.integers(1 << 31)
    # common random numbers: same internal noise stream for both orders
    qAB, _ = run_loop(order_AB, KAPPA, sigma3, np.random.default_rng(seed), FLAT, **cfg)
    qBA, _ = run_loop(order_BA, KAPPA, sigma3, np.random.default_rng(seed), FLAT, **cfg)
    dAB = (qAB[:, -1, 0] - qAB[:, -1, 1]).mean()
    dBA = (qBA[:, -1, 0] - qBA[:, -1, 1]).mean()
    order_rows.append((name, dAB, dBA))
    print(f"  {name:<32s}: mean q0-q1  H0-first {dAB:+.4f}   H1-first {dBA:+.4f}   "
          f"order effect {dAB - dBA:+.4f}")

# analytic recency kernel of Gaussian denoise-and-propagate (m = 0)
T_k = 30
kernels = {}
for lab, s2fun in ((r"$s_t^2$ moment-matched", lambda t: moment_matched_s2(t, KAPPA)),
                   (r"$s^2$ fixed = 4", lambda t: 4.0), (r"$s^2$ fixed = 1", lambda t: 1.0)):
    w = np.array([s2fun(t) / (s2fun(t) + sigma3**2) for t in range(1, T_k + 1)])
    kernels[lab] = np.array([np.prod(w[t - 1:]) for t in range(1, T_k + 1)])
    print(f"  weight of datum 1 vs datum {T_k} in eta_hat_{T_k}  ({lab}): "
          f"{kernels[lab][0]:.4f} vs {kernels[lab][-1]:.4f}")

# ============================================================================
# Exp 4: polarization from internal noise + meta-inference
# ============================================================================
print("\n" + "=" * 78)
print("EXP 4  Pairs of agents, identical mixed evidence, independent internal noise")
print("=" * 78)
N4, T4, sigma4 = 6_000, 40, 1.0
src = rng.choice([0, 1], size=(N4, T4))                     # each datum from H0 or H1
x4 = rng.vonmises(MU[src], KAPPA)
x4_pair = np.concatenate([x4, x4], 0)                       # agent A rows, agent B rows
p4 = softmax(logits(clean_eta(x4, KAPPA), FLAT))
pol_cfg = [
    ("no denoising (raw η̃)", None),
    ("readout-only, Gauss", dict(mode="readout", denoiser="gauss")),
    ("propagate, Gauss s²=4", dict(mode="propagate", denoiser="gauss", s2_rule="fixed", s2_fixed=4.0)),
    ("readout-only, mixture", dict(mode="readout", denoiser="mixture")),
    ("propagate, mixture", dict(mode="propagate", denoiser="mixture")),
]
pol = {}
for name, cfg in pol_cfg:
    if cfg is None:
        et, _ = noisy_eta(x4_pair, KAPPA, sigma4, rng, "perstep")
        q = softmax(logits(et, FLAT))
    else:
        q, _ = run_loop(x4_pair, KAPPA, sigma4, rng, FLAT, **cfg)
    qa, qb = q[:N4], q[N4:]
    tv = 0.5 * np.abs(qa - qb).sum(-1)                         # (N4, T4)
    camps = ((qa.argmax(-1) != qb.argmax(-1)) & (qa.max(-1) > .9) & (qb.max(-1) > .9))
    # error relative to each agent's own noise-free posterior
    err = 0.5 * np.abs(qa - p4).sum(-1)
    pol[name] = dict(tv=tv.mean(0), camps=camps.mean(0), err=err.mean(0), qa=qa)
    print(f"  {name:<24s}: TV(A,B) at t=5,20,40 = "
          + "  ".join(f"{tv[:, t-1].mean():.3f}" for t in (5, 20, 40))
          + f" | opposite confident camps at t=40: {camps[:, -1].mean():.3f}"
          + f" | TV(agent, clean) t=40: {err[:, -1].mean():.3f}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
t4 = np.arange(1, T4 + 1)
for j, (name, _) in enumerate(pol_cfg):
    axs[0].plot(t4, pol[name]["tv"], color=C[j], label=name)
    axs[1].plot(t4, pol[name]["camps"], color=C[j], label=name)
    axs[2].plot(t4, pol[name]["err"], color=C[j], label=name)
axs[0].set_title("Disagreement between agents on identical data")
axs[0].set_ylabel("mean TV(q_A, q_B)")
axs[1].set_title("Pairs in opposite camps (both > 0.9)")
axs[1].set_ylabel("fraction of pairs")
axs[2].set_title("Distance from the noise-free posterior")
axs[2].set_ylabel("mean TV(q_A, p_clean)")
for a in axs: a.set_xlabel("t (observations)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=9)
fig.suptitle(f"Mixed evidence (each datum from H0 or H1), κ = {KAPPA}, per-step σ = {sigma4}",
             color=INK2, fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{OUT}/fig3_polarization.png", dpi=160); plt.close(fig)

# final-belief histograms (small multiples), H0-vs-H1 split
fig, axs = plt.subplots(1, 6, figsize=(15, 2.8), sharey=True)
bins = np.linspace(0, 1, 26)
axs[0].hist(p4[:, -1, 0] / (p4[:, -1, 0] + p4[:, -1, 1]), bins=bins, color=INK2)
axs[0].set_title("noise-free", fontsize=9)
for j, (name, _) in enumerate(pol_cfg):
    qa = pol[name]["qa"][:, -1]
    axs[j + 1].hist(qa[:, 0] / (qa[:, 0] + qa[:, 1]), bins=bins, color=C[j])
    axs[j + 1].set_title(name, fontsize=9)
for a in axs: a.set_xlabel("q(H0)/(q(H0)+q(H1))", fontsize=8)
axs[0].set_ylabel("agents")
fig.suptitle("Final belief split between the two supported hypotheses, t = 40", color=INK2, fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_final_beliefs.png", dpi=160); plt.close(fig)

# order/recency figure
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
for j, (lab, ker) in enumerate(kernels.items()):
    axs[0].plot(np.arange(1, T_k + 1), ker, color=C[j], label=lab)
axs[0].set_xlabel("position of datum in the sequence"); axs[0].set_ylabel("weight in η̂_30")
axs[0].set_title("Recency kernel of Gaussian denoise-and-propagate (σ = 1)")
axs[0].legend(loc="upper left")
names = [r[0] for r in order_rows]
eff = [r[1] - r[2] for r in order_rows]
yy = np.arange(len(names))
axs[1].barh(yy, eff, color=C[0], height=0.5)
axs[1].set_yticks(yy, names, fontsize=9); axs[1].invert_yaxis()
axs[1].axvline(0, color=INK2, lw=1)
for yi, e in zip(yy, eff):
    axs[1].text(max(e, 0) + 0.015, yi, f"{e:+.3f}", va="center", ha="left",
                fontsize=9, color=INK)
axs[1].set_xlim(-0.85, 0.3)
axs[1].set_xlabel("Δ mean(q0 − q1): H0-first minus H1-first")
axs[1].set_title("Order effect, same 20 data (10 H0-like + 10 H1-like)")
axs[1].grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_order_effects.png", dpi=160); plt.close(fig)

# ============================================================================
# Exp 5: normative contrary updating (no noise)
# ============================================================================
print("\n" + "=" * 78)
print("EXP 5  Contrary updating on a shared datum, exact Bayes, no noise")
print("=" * 78)
pA, pB = np.array([.45, .10, .45]), np.array([.45, .45, .10])
d_eta = KAPPA * unit(np.deg2rad(90.0))
for nm, p in (("A", pA), ("B", pB)):
    post = softmax(np.log(p) + E @ d_eta)
    print(f"  agent {nm}: prior p(H0)={p[0]:.3f} -> posterior p(H0)={post[0]:.3f}")
# exact criterion check: p_k rises  <=>  e_k.d_eta > log sum_j p_j exp(e_j.d_eta)
M5 = 200_000
pp = rng.dirichlet(np.ones(3), M5)
de = rng.normal(size=(M5, 2)) * 2
post = softmax(np.log(pp) + de @ E.T)
lhs = de @ E.T
rhs = logsumexp(np.log(pp) + lhs, axis=-1, keepdims=True)
agree = ((post > pp) == (lhs > rhs)).mean()
print(f"  criterion  p_k up <=> e_k·Δη > log E_p[exp(e·Δη)]  holds in {agree:.4%} of cases")
# how common is contrary updating on H0 for random pairs of agents sharing a datum?
pq = rng.dirichlet(np.ones(3), M5)
post_q = softmax(np.log(pq) + de @ E.T)
contrary = np.sign(post[:, 0] - pp[:, 0]) != np.sign(post_q[:, 0] - pq[:, 0])
print(f"  random agent pairs, random shared datum: P(they move H0 in opposite directions) = "
      f"{contrary.mean():.3f}")
print("\nfigures written to ./figures/")
