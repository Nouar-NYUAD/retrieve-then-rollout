"""One-off data preparation on Jubail (run with PYTHONPATH pointing at a temporary h5py install).
1) D4RL walker2d_medium-v2.hdf5 -> npz (same fields as hopper/halfcheetah).
2) CAMELS-US: the 18 basins of the TabPFN environmental paper (one per HUC region, fewest missing
   flow days 1980-2014) -> camels18.npz with daily dates, flow q (mm/day, NaN if missing) and Daymet
   forcing prcp, tmax, tmin, srad, vp, dayl. Conversion cfs -> mm/day as in envbench/datasets.py."""
import glob, os, sys
import numpy as np
D4 = "/scratch/naa9497/icwm/data/d4rl"
if "walker" in sys.argv:
    import h5py
    f = h5py.File(f"{D4}/walker2d_medium-v2.hdf5")
    d = {k: np.asarray(f[k]) for k in ["observations", "actions", "next_observations", "rewards", "terminals", "timeouts"]}
    ends = np.where(d["terminals"] | d["timeouts"])[0]
    cont = np.abs(d["next_observations"][:-1] - d["observations"][1:]).max(1)
    print("walker2d", {k: v.shape for k, v in d.items()}, "episodes", len(ends),
          "non-end discontinuities", int(((cont > 1e-6) & ~np.isin(np.arange(len(cont)), ends)).sum()))
    np.savez_compressed(f"{D4}/walker2d_medium-v2.npz",
                        **{k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in d.items()})
if "camels" in sys.argv:
    import pandas as pd
    root = "/scratch/naa9497/envbench/data/raw/camels_data"
    find = lambda pat: glob.glob(f"{root}/**/{pat}", recursive=True)
    forcing = {os.path.basename(f).split("_")[0]: f for f in find("*_lump_cida_forcing_leap.txt") if "daymet" in f}
    flow = {os.path.basename(f).split("_")[0]: f for f in find("*_streamflow_qc.txt")}
    region = {b: os.path.basename(os.path.dirname(f)) for b, f in forcing.items()}
    n_days = len(pd.date_range("1980-01-01", "2014-12-31", freq="D"))
    best = {}
    for b in sorted(set(forcing) & set(flow)):
        q = pd.read_csv(flow[b], sep=r"\s+", header=None, usecols=[1, 4], names=["year", "q"])
        q = q[(q["year"] >= 1980) & (q["year"] <= 2014)]
        missing = 1.0 - float((q["q"] >= 0).sum()) / n_days
        if region[b] not in best or (missing, b) < best[region[b]]:
            best[region[b]] = (missing, b)
    basins = [best[r][1] for r in sorted(best)]
    out = {"basins": np.array(basins, dtype="U8")}
    for b in basins:
        with open(forcing[b]) as fh:
            lat, elev, area = (float(fh.readline()) for _ in range(3))
        met = pd.read_csv(forcing[b], sep=r"\s+", skiprows=3)
        met.columns = [c.split("(")[0].strip().lower() for c in met.columns]
        met.index = pd.to_datetime(dict(year=met["year"], month=met["mnth"], day=met["day"]))
        met = met[["prcp", "tmax", "tmin", "srad", "vp", "dayl"]]
        q = pd.read_csv(flow[b], sep=r"\s+", header=None, names=["gauge", "year", "mnth", "day", "q_cfs", "flag"], dtype={"gauge": str})
        q.index = pd.to_datetime(dict(year=q["year"], month=q["mnth"], day=q["day"]))
        q_mm = q["q_cfs"].where(q["q_cfs"] >= 0) * 0.028316846592 * 86400 / area * 1000
        d = met.join(q_mm.rename("q"), how="left")
        d = d[~d.index.duplicated()].asfreq("D")
        d = d[(d.index >= "1980-10-01") & (d.index <= "2014-09-30")]
        out[f"{b}_dates"] = np.array(d.index.strftime("%Y-%m-%d").tolist(), dtype="U10")
        out[f"{b}_q"] = d["q"].to_numpy(float)
        out[f"{b}_forcing"] = d[["prcp", "tmax", "tmin", "srad", "vp", "dayl"]].to_numpy(float)
        print(b, region[b], len(d), "missing q", int(np.isnan(out[f"{b}_q"]).sum()), "forcing NaN", int(np.isnan(out[f"{b}_forcing"]).sum()))
    os.makedirs("/scratch/naa9497/icwm/data/camels", exist_ok=True)
    np.savez_compressed("/scratch/naa9497/icwm/data/camels/camels18.npz", **out)
