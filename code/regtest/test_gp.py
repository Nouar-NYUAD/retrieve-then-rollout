import sys; sys.path.insert(0, sys.argv[1])
import numpy as np, study as St, pilot as P
from sklearn.gaussian_process import GaussianProcessRegressor
rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(2000, 2)); f = lambda Z: np.stack([np.sin(3 * Z[:, 0]) * Z[:, 1], np.cos(4 * Z[:, 1]) + 0.3 * Z[:, 0] ** 2], 1); Y = f(X) + 0.1 * rng.normal(size=(len(X), 2))
ks = St.fit_gp_kernels(X, Y, 300, np.random.default_rng(1))
idx = np.random.default_rng(1).choice(len(X), 300, replace=False)          # the same subset
Xq = rng.uniform(-3, 3, size=(200, 2)); Yq = f(Xq)
for j, (k, mx, sx, my, sy) in enumerate(ks):
    ref = GaussianProcessRegressor(kernel=k, optimizer=None, normalize_y=False).fit((X[idx] - mx) / sx, (Y[idx, j] - my) / sy)
    a = St.FixedKernelGP(k, mx, sx, my, sy).fit(X[idx], Y[idx, j]).predict(Xq)
    print(f"dim {j}: identical to reference GP on fit subset: {np.allclose(a, ref.predict((Xq - mx) / sx) * sy + my)}",
          f"| RMSE on test {np.sqrt(np.mean((a - Yq[:, j])**2)):.4f}")
# local (retrieved) context: new scaling vs the old buggy re-standardisation
pool = P.Pool(X, Y)
new_e, old_e = [], []
for q, yq in zip(Xq[:50], Yq[:50]):
    nb = pool.nearest(100, q)
    for j, (k, mx, sx, my, sy) in enumerate(ks):
        new_e.append(St.FixedKernelGP(k, mx, sx, my, sy).fit(X[nb], Y[nb, j]).predict(q[None])[0] - yq[j])
        mu, sd = X[nb].mean(0), X[nb].std(0) + 1e-12
        old = GaussianProcessRegressor(kernel=k, optimizer=None, normalize_y=True).fit((X[nb] - mu) / sd, Y[nb, j])
        old_e.append(old.predict(((q - mu) / sd)[None])[0] - yq[j])
print(f"retrieved-context RMSE: fixed {np.sqrt(np.mean(np.square(new_e))):.6f}  vs old scaling {np.sqrt(np.mean(np.square(old_e))):.6f}")
