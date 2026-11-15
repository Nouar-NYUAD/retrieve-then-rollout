import sys, types, numpy as np
sys.path.insert(0, "/scratch/naa9497/icwm/pilot"); import study as St
d = dict(np.load("/scratch/naa9497/icwm/data/camels/camels18.npz"))
for seed in range(6):
    a = types.SimpleNamespace(camels_file="/scratch/naa9497/icwm/data/camels/camels18.npz", cur_seed=seed, lags=7)
    Xtr, Ytr, S, _, U, sd = St.camels_seed(a, None)
    b = sorted(str(x) for x in d["basins"])[seed]; dates = d[f"{b}_dates"]; cut = int(np.searchsorted(dates, "2005-10-01"))
    # alignment: U[t] must be the forcing of the day AFTER S[t]
    assert np.allclose(U[0], d[f"{b}_forcing"][cut + 1]) and np.allclose(S[0, 0], d[f"{b}_q"][cut])
    starts = np.linspace(600, len(S) - 1 - 90 - 1, 3).astype(int)
    out = []
    for t0 in starts:
        obs = S[t0 + 1:t0 + 91, 0]; pers = np.full_like(obs, S[t0, 0]); clim = np.full_like(obs, S[:t0, 0].mean())
        nse = lambda sim: 1 - ((sim - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
        out.append((round(nse(pers), 2), round(nse(clim), 2), round(obs.std() / sd[0], 2)))
    print(f"basin {b} (seed {seed}): per start (NSE persistence, NSE long-term mean, window sd / train sd): {out}")
print("alignment asserts passed")
