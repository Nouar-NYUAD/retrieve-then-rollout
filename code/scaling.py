"""Cost and accuracy of retrieved context vs the full pool as the pool grows (D4RL Hopper).

For pools of n_train D4RL episodes (+ the test-episode history before the start, as in study.py) we roll
out TabPFN-3.5 (one regressor per state dimension, predicting the state change) for `horizon` steps
from the seed's first start, with
  retrieval : the H=100 pool rows nearest to the current query, re-selected and refitted every step;
  full      : the WHOLE pool as context (no cap), fitted once, predicting every step.
Measured per configuration: rollout error (normalised RMSE) at several horizons, wall time split into
fitting, prediction and neighbour search, and peak GPU memory (torch.cuda.max_memory_allocated). A CUDA
out-of-memory error is recorded as a result ("oom"), not a crash. One process per GPU so the time and
memory numbers are not shared with other jobs.
Usage: python scaling.py --system hopper --n_train 40 --seed_ids 0 1 2 --out scale_hopper_nt40.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import pilot as P
import study as S

CODE_VERSION = "2026-11-15scale1"


def gpu_reset():
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def gpu_peak_gb():
    return float(torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else 0.0


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def rollout(strategy, pool, X0, U, steps, H, seed, d):
    t_fit = t_pred = t_knn = 0.0
    s = X0.copy(); traj = []
    models = None
    if strategy == "full":
        tic = time.time()
        models = [P.TabPFN("v3.5", seed).fit(pool.X, pool.Y[:, j]) for j in range(d)]
        sync(); t_fit += time.time() - tic
    for h in range(steps):
        q = np.concatenate([s, U[h]])[None]
        if strategy == "full":
            tic = time.time(); delta = np.array([m.predict(q)[0] for m in models]); sync(); t_pred += time.time() - tic
        else:
            tic = time.time(); idx = pool.nearest(H, q[0]); t_knn += time.time() - tic
            delta = np.empty(d)
            for j in range(d):
                tic = time.time(); m = P.TabPFN("v3.5", seed).fit(pool.X[idx], pool.Y[idx, j]); sync(); t_fit += time.time() - tic
                tic = time.time(); delta[j] = m.predict(q)[0]; sync(); t_pred += time.time() - tic
                del m
        s = s + delta
        if not np.all(np.isfinite(s)):
            s = np.full_like(s, np.nan)
        traj.append(s.copy())
    return np.array(traj), t_fit, t_pred, t_knn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="hopper", choices=list(S.D4RL))
    p.add_argument("--n_train", type=int, default=10)
    p.add_argument("--seed_ids", nargs="+", type=int, default=[0])
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--H", type=int, default=100)
    p.add_argument("--strategies", nargs="+", default=["retrieval", "full"])
    p.add_argument("--data_dir", default="/scratch/naa9497/icwm/data/d4rl")
    p.add_argument("--out", default="scale.json")
    a = p.parse_args()
    report = [h for h in (1, 10, 25, 50, 100) if h <= a.horizon]
    out = dict(args=vars(a), code_version=CODE_VERSION, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               results=[])
    for seed in a.seed_ids:
        args = argparse.Namespace(system=a.system, data_dir=a.data_dir, t0_min=100, horizon=100, n_train=a.n_train, lags=1)
        rng = np.random.default_rng(1000 * seed + 7)                         # same rule as study.run
        Xtr, Ytr, S_te, _, U_te, sd = S.d4rl_seed(args, rng)
        X_te, Y_te = S.lagged(S_te, U_te, 1)
        t0 = 100
        pool = S.RPool(np.vstack([Xtr, X_te[:t0]]), np.vstack([Ytr, Y_te[:t0]]))
        d = S_te.shape[1]
        for st in a.strategies:
            gpu_reset(); tic = time.time()
            rec = dict(system=a.system, n_train=a.n_train, pool_rows=int(len(pool.X)), seed=seed, t0=t0, strategy=st,
                       H=a.H if st == "retrieval" else int(len(pool.X)), horizon=a.horizon, code_version=CODE_VERSION)
            try:
                traj, tf, tp, tk = rollout(st, pool, S_te[t0], U_te[t0:t0 + a.horizon], a.horizon, a.H, seed, d)
                err = np.sqrt(np.mean(((traj - S_te[t0 + 1:t0 + 1 + a.horizon]) / sd) ** 2, axis=1))
                rec.update(status="ok", rollout_nrmse={str(h): float(err[h - 1]) for h in report},
                           secs_total=time.time() - tic, secs_fit=tf, secs_predict=tp, secs_knn=tk,
                           secs_per_step=(time.time() - tic) / a.horizon, gpu_peak_gb=gpu_peak_gb())
            except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                if "out of memory" not in str(e).lower() and not isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError)):
                    raise
                rec.update(status="oom", secs_total=time.time() - tic, gpu_peak_gb=gpu_peak_gb(), error=str(e)[:200])
                gpu_reset()
            out["results"].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "error"}), flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
