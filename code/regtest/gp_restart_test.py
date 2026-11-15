import sys, types, numpy as np; sys.path.insert(0, sys.argv[1]); import study as St
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
for lags in [4, 1]:
    a = types.SimpleNamespace(system="pendulum", obs_noise=0.05, n_train=4, T_train=1000, T_test=2000, lags=lags, l96_dim=20)
    Xtr, Ytr, *_ = St.synthetic_seed(a, np.random.default_rng(7))
    idx = np.random.default_rng(1).choice(len(Xtr), 500, replace=False); X = Xtr[idx]; Z = (X - X.mean(0)) / X.std(0)
    y = Ytr[idx, 0]; y = (y - y.mean()) / y.std()
    for nr in [0, 5]:
        k = ConstantKernel(1.0) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(1e-3, (1e-8, 1e1))
        g = GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=nr, random_state=0).fit(Z, y)
        print(f"lags={lags} restarts={nr}: logML={g.log_marginal_likelihood_value_:.1f}  noise={g.kernel_.k2.noise_level:.3f}  "
              f"amp={g.kernel_.k1.k1.constant_value:.3f}  min/median ls={np.min(g.kernel_.k1.k2.length_scale):.3f}/{np.median(g.kernel_.k1.k2.length_scale):.3f}")
