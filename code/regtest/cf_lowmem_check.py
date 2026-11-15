"""LowMemPredictor (counterfactual.py) must give exactly the predictions of study.Predictor for TabPFN."""
import sys, argparse, numpy as np
sys.path.insert(0, ".")
import counterfactual as C, study as S, pilot as P
rng = np.random.default_rng(0)
a = argparse.Namespace(system="pendulum", n_train=1, T_train=300, T_test=300, gain=1.0, sigma=0.3)
Xtr, Ytr, *_ = C.seed_data(a, rng)
pool = P.Pool(Xtr, Ytr)
m = argparse.Namespace(target="delta", tabpfn_version="v3.5", tabpfn_n_estimators=0, dynapfn_degree=4,
                       gp_fit_n=500, gp_restarts=5, tabpfn_fit_mode=None)
idx = pool.nearest(60, Xtr[5]); Xq = Xtr[rng.choice(len(Xtr), 7)]
ref = S.Predictor(m, 0, pool, rng)("tabpfn", idx, Xq)
new = C.LowMemPredictor(m, 0, pool, rng)("tabpfn", idx, Xq)
print("max abs diff", np.abs(ref - new).max(), "max |pred|", np.abs(ref).max())
assert np.abs(ref - new).max() < 1e-6
print("LOWMEM CHECK PASSED")
