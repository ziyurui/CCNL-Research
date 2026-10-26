import numpy as np
from circbayes import *
rng = np.random.default_rng(11)
KAPPA, SIGMA, T, N = 1.0, 1.0, 300, 3000
FLAT = np.log(np.full(3, 1/3))
k = rng.integers(0, 3, N)
x = rng.vonmises(MU[k][:, None], KAPPA, (N, T))
x2 = np.concatenate([x, x])
q, st = run_loop(x2, KAPPA, SIGMA, rng, FLAT, mode="propagate", denoiser="mixture")
qa, qb = q[:N], q[N:]
camps = (qa.argmax(-1) != qb.argmax(-1)) & (qa.max(-1) > .9) & (qb.max(-1) > .9)
acc = (qa.argmax(-1) == k[:, None])
wrong_conf = (~acc) & (qa.max(-1) > .99)
print("t     camps   accuracy   P(confident>0.99 & wrong)")
for t in (10, 30, 60, 100, 150, 200, 300):
    print(f"{t:<5d} {camps[:, t-1].mean():.3f}   {acc[:, t-1].mean():.3f}      {wrong_conf[:, t-1].mean():.3f}")
# do wrong-locked agents ever escape? fraction wrong at t=60 that are right at t=300
w60 = ~acc[:, 59]
print(f"\nagents wrong at t=60: {w60.mean():.3f}; of those, right at t=300: {acc[w60, -1].mean():.3f}")
# raw baseline long-run
et, _ = noisy_eta(x2, KAPPA, SIGMA, rng, "perstep")
qr = softmax(logits(et, FLAT)); ra, rb = qr[:N], qr[N:]
cr = (ra.argmax(-1) != rb.argmax(-1)) & (ra.max(-1) > .9) & (rb.max(-1) > .9)
print("raw baseline camps t=60/300:", cr[:, 59].mean().round(3), cr[:, -1].mean().round(3))
