import sys, types, numpy as np
sys.path.insert(0, sys.argv[1]); import study as St
f = "/scratch/naa9497/icwm/data/camels/camels18.npz"; d = dict(np.load(f))
a = types.SimpleNamespace(camels_file=f, cur_seed=3, lags=7, camels_features="ext")
St._CAMELS.clear(); Xtr, Ytr, S, _, U, sd = St.camels_seed(a, None)
b = sorted(str(x) for x in d["basins"])[3]; cut = int(np.searchsorted(d[f"{b}_dates"], "2005-10-01")); P = d[f"{b}_forcing"][:, 0]; Tx = d[f"{b}_forcing"][:, 1]
for t in [0, 10, 500]:
    day = cut + 1 + t                                   # target day of test row t
    assert np.allclose(U[t, :6], d[f"{b}_forcing"][day])
    assert np.isclose(U[t, 6], P[day - 6:day + 1].sum()) and np.isclose(U[t, 7], P[day - 29:day + 1].sum())
    assert np.isclose(U[t, 8], P[day - 89:day + 1].sum()) and np.isclose(U[t, 9], Tx[day - 6:day + 1].mean())
    assert np.isclose(U[t, 10], Tx[day - 29:day + 1].mean())
print("ext feature alignment OK; U shape", U.shape, "train rows", len(Xtr), "X cols", Xtr.shape[1], "(7 lags + 11 forcing)")
