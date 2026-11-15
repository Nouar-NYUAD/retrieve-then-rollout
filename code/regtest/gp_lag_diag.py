import sys, types, numpy as np; sys.path.insert(0, sys.argv[1]); import study as St
a = types.SimpleNamespace(system="pendulum", obs_noise=0.05, n_train=4, T_train=1000, T_test=2000, lags=4, l96_dim=20)
Xtr, Ytr, *_ = St.synthetic_seed(a, np.random.default_rng(7))
print("X shape", Xtr.shape)
ks = St.fit_gp_kernels(Xtr, Ytr, 500, np.random.default_rng(1))
for j, (k, *_) in enumerate(ks): print("dim", j, k)
