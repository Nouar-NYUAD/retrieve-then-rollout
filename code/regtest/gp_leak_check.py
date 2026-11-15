"""Independent check of the GP's accuracy on clean Lorenz: a plain sklearn GP (optimised kernel)
fitted ONLY on 500 transitions from separate training trajectories (no test history), rolled out
300 steps on a fresh test trajectory; compared with persistence."""
import sys; sys.path.insert(0, sys.argv[1])
import numpy as np, pilot as P
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
rng = np.random.default_rng(123)
trains = [P.simulate("lorenz", 1000, rng) for _ in range(4)]
X = np.vstack([P.transitions(S, U)[0] for S, U in trains]); Y = np.vstack([P.transitions(S, U)[1] for S, U in trains])
idx = rng.choice(len(X), 500, replace=False); mx, sx = X[idx].mean(0), X[idx].std(0)
gps = []
for j in range(3):
    my, sy = Y[idx, j].mean(), Y[idx, j].std()
    k = ConstantKernel() * RBF(np.ones(3), (1e-2, 1e3)) + WhiteKernel(1e-3, (1e-8, 1e1))
    gps.append((GaussianProcessRegressor(k, normalize_y=False).fit((X[idx] - mx) / sx, (Y[idx, j] - my) / sy), my, sy))
S_te, _ = P.simulate("lorenz", 2000, np.random.default_rng(999)); sd = np.vstack([S for S, _ in trains]).std(0)
for t0 in [600, 1000, 1500]:
    s = S_te[t0].copy(); errs = []
    for h in range(300):
        s = s + np.array([g.predict(((s - mx) / sx)[None])[0] * sy + my for g, my, sy in gps])
        errs.append(np.sqrt(np.mean(((s - S_te[t0 + h + 1]) / sd) ** 2)))
    print(f"t0={t0}: GP rollout nRMSE @1 {errs[0]:.2e} @100 {errs[99]:.2e} @300 {errs[299]:.2e} | persistence @300 {np.sqrt(np.mean(((S_te[t0]-S_te[t0+300])/sd)**2)):.2f}")
print("kernel dim 0:", gps[0][0].kernel_)
