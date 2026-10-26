"""
same_prior_polarization.py -- identical priors, identical data, independent
internal noise.  Noise-free Bayesian twins can never disagree here, so any
disagreement is manufactured by internal noise + the meta-inference step.

Evidence conditions
  well-specified : one true hypothesis k ~ U{0,1,2}; x ~ VM(mu_k, kappa)
  mixed          : each datum from H0 or H1 with prob 1/2 (violates single-source)
  directionless  : x ~ Uniform(circle) (no hypothesis true)
"""
import numpy as np
from circbayes import *

KAPPA, SIGMA, T, NP = 1.0, 1.0, 60, 4000
FLAT = np.log(np.full(3, 1 / 3))
rng = np.random.default_rng(7)


def make(cond, n):
    if cond == "well-specified":
        k = rng.integers(0, 3, n)
        return rng.vonmises(MU[k][:, None], KAPPA, (n, T)), k
    if cond == "mixed":
        src = rng.integers(0, 2, (n, T))
        return rng.vonmises(MU[src], KAPPA), None
    return rng.uniform(-np.pi, np.pi, (n, T)), None


METHODS = [
    ("raw (no denoising)", None),
    ("Gauss readout", dict(mode="readout", denoiser="gauss")),
    ("Gauss propagate s2=4", dict(mode="propagate", denoiser="gauss", s2_rule="fixed", s2_fixed=4.0)),
    ("mixture readout", dict(mode="readout", denoiser="mixture")),
    ("mixture propagate", dict(mode="propagate", denoiser="mixture")),
]


def run(method_cfg, x2, sigma):
    if method_cfg is None:
        et, _ = noisy_eta(x2, KAPPA, sigma, rng, "perstep")
        return softmax(logits(et, FLAT))
    return run_loop(x2, KAPPA, sigma, rng, FLAT, **method_cfg)[0]


def stats(q, n, k):
    qa, qb = q[:n], q[n:]
    camps = (qa.argmax(-1) != qb.argmax(-1)) & (qa.max(-1) > .9) & (qb.max(-1) > .9)
    tv = 0.5 * np.abs(qa - qb).sum(-1)
    acc = None if k is None else (qa.argmax(-1) == k[:, None]).mean(0)
    conf = qa.max(-1).mean(0)
    return camps.mean(0), tv.mean(0), acc, conf


ts = (5, 15, 30, 60)
print(f"kappa={KAPPA}, per-step sigma={SIGMA}, {NP} twin pairs, flat identical priors\n")
for cond in ("well-specified", "mixed", "directionless"):
    x, k = make(cond, NP)
    x2 = np.concatenate([x, x])
    pc = softmax(logits(clean_eta(x, KAPPA), FLAT))
    print("=" * 96)
    print(f"{cond.upper()}   noise-free twins: disagreement 0 by construction;"
          f" mean confidence at t=5/15/30/60 = " + "/".join(f"{pc[:, t-1].max(-1).mean():.2f}" for t in ts)
          + ("" if k is None else "; accuracy " + "/".join(f"{(pc[:, t-1].argmax(-1)==k).mean():.2f}" for t in ts)))
    print(f"{'method':<22s} | {'opposite confident camps':^27s} | {'mean TV(A,B)':^27s} | "
          + ("accuracy t=60" if k is not None else "confidence t=60"))
    print(f"{'':<22s} | " + " ".join(f"t={t:<4d}" for t in ts) + " | "
          + " ".join(f"t={t:<4d}" for t in ts) + " |")
    for name, cfg in METHODS:
        q = run(cfg, x2, SIGMA)
        camps, tv, acc, conf = stats(q, NP, k)
        tail = f"{acc[-1]:.3f}" if acc is not None else f"{conf[-1]:.3f}"
        print(f"{name:<22s} | " + " ".join(f"{camps[t-1]:.3f} " for t in ts) + " | "
              + " ".join(f"{tv[t-1]:.3f} " for t in ts) + f" | {tail}")
    print()

# sigma sweep for the two strongest candidates
print("=" * 96)
print("Opposite confident camps at t=60 vs internal noise sigma")
print(f"{'condition':<16s} {'method':<20s} " + " ".join(f"s={s:<5}" for s in (0.25, 0.5, 1.0, 2.0)))
for cond in ("well-specified", "mixed", "directionless"):
    x, k = make(cond, 3000)
    x2 = np.concatenate([x, x])
    for name, cfg in (METHODS[0], METHODS[4]):
        row = []
        for s in (0.25, 0.5, 1.0, 2.0):
            camps, *_ = stats(run(cfg, x2, s), 3000, k)
            row.append(camps[-1])
        print(f"{cond:<16s} {name:<20s} " + " ".join(f"{r:.3f}  " for r in row))
