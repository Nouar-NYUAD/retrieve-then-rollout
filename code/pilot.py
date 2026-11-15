"""Pilot: does retrieval-selected context beat a recent window for an in-context (TabPFN) world model?

Systems
  lorenz   : Lorenz-63 (sigma=10, rho=28, beta=8/3), RK4, dt=0.01, autonomous. State d=3.
  pendulum : damped pendulum with torque, theta'' = -(g/l) sin(theta) - b theta' + u/(m l^2),
             g/l=9.81, b=0.1, m=l=1, RK4, dt=0.05; u ~ N(0, 1) held per step. State (theta, omega).

Data protocol (identical for every context strategy and model)
  pool   = transitions from n_train training trajectories + the test trajectory's own history
           before the rollout start t0 (the only data a recent window can see).
  query  = (state, action) at each step; target = state change s_{t+1} - s_t (one regressor per
           state dimension, every input dimension used as features, as in arXiv 2611.11921).

Context strategies (budget H rows taken from the same pool)
  recent    : the H most recent transitions of the test trajectory before t0, fixed during rollout.
  random    : H rows sampled uniformly from the pool (fixed per rollout).
  retrieval : the H nearest pool rows to the current query in standardized input space,
              re-selected at every rollout step.
  full      : the whole pool (upper reference; capped at --full_cap rows by uniform subsampling).

Models
  tabpfn    : TabPFN regressor, default settings (version from --tabpfn_version).
  knn       : mean of the k=10 nearest context rows (is TabPFN more than neighbour averaging?).
  locallin  : ridge regression on the context rows (local linear model).
  parrot    : 1-nearest-neighbour copy (context parroting for state-action inputs).
  zero      : persistence (predicted change = 0).

Metrics, all in state units normalised by the per-dimension std of the pool states
  one-step nRMSE on held-out test transitions;
  rollout nRMSE at chosen horizons; valid prediction time (first step where nRMSE > vpt_thresh).
Rollouts use the recorded actions of the test trajectory (open loop), point predictions fed back.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- systems
def rk4(f, x, u, dt):
    k1 = f(x, u); k2 = f(x + dt / 2 * k1, u); k3 = f(x + dt / 2 * k2, u); k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz_f(x, u):
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    return np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])


def pendulum_f(x, u):
    return np.array([x[1], -9.81 * np.sin(x[0]) - 0.1 * x[1] + u[0]])


SYSTEMS = {
    "lorenz": dict(f=lorenz_f, dt=0.01, d=3, a=0, burn=1000),
    "pendulum": dict(f=pendulum_f, dt=0.05, d=2, a=1, burn=0),
}


def simulate(name, T, rng):
    sp = SYSTEMS[name]
    if name == "lorenz":
        x = rng.normal(0, 1, 3) + np.array([1.0, 1.0, 25.0])
    else:
        x = np.array([rng.uniform(-np.pi, np.pi), rng.normal(0, 1)])
    U = rng.normal(0, 1, (T + sp["burn"], sp["a"]))
    S = np.empty((T + sp["burn"] + 1, sp["d"])); S[0] = x
    for t in range(T + sp["burn"]):
        S[t + 1] = rk4(sp["f"], S[t], U[t], sp["dt"])
    return S[sp["burn"]:], U[sp["burn"]:]          # states T+1, actions T


def transitions(S, U):
    X = np.hstack([S[:-1], U]) if U.shape[1] else S[:-1].copy()
    return X, S[1:] - S[:-1]


# ----------------------------------------------------------------------------- models
class Zero:
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))


class KNNMean:
    def __init__(self, k): self.k = k
    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12; self.X = (X - self.mu) / self.sd; self.y = y
        return self
    def predict(self, X):
        Z = (X - self.mu) / self.sd
        d2 = ((Z[:, None, :] - self.X[None]) ** 2).sum(-1)
        k = min(self.k, len(self.X))
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        return self.y[idx].mean(1)


class Ridge:
    def __init__(self, lam=1e-3): self.lam = lam
    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        P = self.lam * np.eye(Z.shape[1]); P[-1, -1] = 0
        self.w = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
        return self
    def predict(self, X):
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        return Z @ self.w


class TabPFN:
    def __init__(self, version, seed, fit_mode=None):
        import torch
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        self.m = TabPFNRegressor.create_default_for_version(ModelVersion(version)).set_params(
            random_state=seed, device="cuda" if torch.cuda.is_available() else "cpu",
            ignore_pretraining_limits=True)
        if fit_mode:                      # e.g. "fit_with_cache" for many predicts per fit (MPC)
            self.m.set_params(fit_mode=fit_mode)
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict(self, X): return np.asarray(self.m.predict(X), dtype=float)


def make_model(name, args, seed):
    if name == "tabpfn": return TabPFN(args.tabpfn_version, seed, getattr(args, "tabpfn_fit_mode", None))
    if name == "knn": return KNNMean(10)
    if name == "parrot": return KNNMean(1)
    if name == "locallin": return Ridge()
    if name == "zero": return Zero()
    raise ValueError(name)


def predict_delta(model_name, args, seed, Xc, Yc, Xq):
    """One regressor per output dimension; returns (len(Xq), d)."""
    out = np.empty((len(Xq), Yc.shape[1]))
    for j in range(Yc.shape[1]):
        m = make_model(model_name, args, seed).fit(Xc, Yc[:, j])
        out[:, j] = m.predict(Xq)
    return out


# ----------------------------------------------------------------------------- context selection
class Pool:
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.Z = (X - self.mu) / self.sd

    def recent(self, H, hist_idx):
        return hist_idx[-H:]

    def random(self, H, rng):
        return rng.choice(len(self.X), size=min(H, len(self.X)), replace=False)

    def nearest(self, H, q):
        d2 = (((q - self.mu) / self.sd - self.Z) ** 2).sum(1)
        H = min(H, len(self.X))
        return np.argpartition(d2, H - 1)[:H]


def select(strategy, pool, H, hist_idx, q, rng, full_cap):
    if strategy == "recent": return pool.recent(H, hist_idx)
    if strategy == "random": return pool.random(H, rng)
    if strategy == "retrieval": return pool.nearest(H, q)
    if strategy == "full":
        return pool.random(full_cap, rng) if len(pool.X) > full_cap else np.arange(len(pool.X))
    raise ValueError(strategy)


# ----------------------------------------------------------------------------- experiment
def run(args):
    sp = SYSTEMS[args.system]
    results = []
    for seed in (args.seed_ids if args.seed_ids else range(args.seeds)):
        rng = np.random.default_rng(1000 * seed + 7)
        trains = [transitions(*simulate(args.system, args.T_train, rng)) for _ in range(args.n_train)]
        S_te, U_te = simulate(args.system, args.T_test, rng)
        X_te, Y_te = transitions(S_te, U_te)
        Xtr = np.vstack([x for x, _ in trains]); Ytr = np.vstack([y for _, y in trains])
        state_sd = np.vstack([S_te[: args.t0_min]] + [x[:, : sp["d"]] for x, _ in trains]).std(0)

        starts = np.linspace(args.t0_min, len(X_te) - args.horizon - 1, args.n_starts).astype(int)
        for t0 in starts:
            # pool = training transitions + test history before t0
            X_pool = np.vstack([Xtr, X_te[:t0]]); Y_pool = np.vstack([Ytr, Y_te[:t0]])
            hist_idx = np.arange(len(Xtr), len(X_pool))
            pool = Pool(X_pool, Y_pool)
            for strategy in args.strategies:
                for H in (args.H if strategy != "full" else [0]):
                    if strategy == "recent" and H > len(hist_idx):
                        continue
                    for model in args.models:
                        if strategy == "full" and model in ("parrot", "knn", "zero"):
                            continue                      # identical to retrieval for these
                        crng = np.random.default_rng(seed * 7919 + int(t0))
                        tic = time.time()
                        # one-step error at t0 .. t0+horizon-1 using the TRUE state as query
                        # (rollout below feeds back predictions)
                        s = S_te[t0].copy(); err_roll = []; err_one = []
                        idx_fixed = None
                        if strategy in ("recent", "random", "full"):
                            idx_fixed = select(strategy, pool, H, hist_idx, None, crng, args.full_cap)
                        for h in range(args.horizon):
                            u = U_te[t0 + h]
                            q = np.concatenate([s, u])[None]
                            q_true = X_te[t0 + h][None]
                            idx = idx_fixed if idx_fixed is not None else select(
                                strategy, pool, H, hist_idx, q[0], crng, args.full_cap)
                            one = h < args.one_step_n
                            if one and idx_fixed is None:
                                # retrieval: the one-step query gets its own neighbours
                                idx1 = select(strategy, pool, H, hist_idx, q_true[0], crng,
                                              args.full_cap)
                                d1 = predict_delta(model, args, seed, pool.X[idx1], pool.Y[idx1],
                                                   q_true)[0]
                                dq = predict_delta(model, args, seed, pool.X[idx], pool.Y[idx], q)
                            else:
                                dq = predict_delta(model, args, seed, pool.X[idx], pool.Y[idx],
                                                   np.vstack([q, q_true]) if one else q)
                                d1 = dq[1] if one else None
                            if one:
                                err_one.append((d1 - Y_te[t0 + h]) / state_sd)
                            s = s + dq[0]
                            if not np.all(np.isfinite(s)):
                                s = np.full_like(s, np.nan)
                            err_roll.append(np.sqrt(np.mean(((s - S_te[t0 + h + 1]) / state_sd) ** 2)))
                        err_roll = np.array(err_roll)
                        bad = np.where(~(err_roll <= args.vpt_thresh))[0]
                        results.append(dict(
                            system=args.system, seed=seed, t0=int(t0), strategy=strategy, H=int(H),
                            model=model, secs=time.time() - tic,
                            one_step_nrmse=float(np.sqrt(np.mean(np.square(err_one)))),
                            rollout_nrmse={str(k): float(err_roll[k - 1]) for k in args.report_h
                                           if k <= args.horizon},
                            vpt_steps=int(bad[0]) if len(bad) else args.horizon))
                        r = results[-1]
                        print(f"[{args.system} s{seed} t0={t0}] {strategy:9s} H={H:4d} {model:8s} "
                              f"1step={r['one_step_nrmse']:.4f} vpt={r['vpt_steps']:3d} "
                              f"({r['secs']:.1f}s)", flush=True)
            Path(args.out).write_text(json.dumps(dict(args=vars(args), results=results), indent=1))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="lorenz", choices=list(SYSTEMS))
    p.add_argument("--models", nargs="+", default=["tabpfn", "knn", "locallin", "parrot", "zero"])
    p.add_argument("--strategies", nargs="+", default=["recent", "random", "retrieval", "full"])
    p.add_argument("--H", nargs="+", type=int, default=[100, 500])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed_ids", nargs="+", type=int, default=None)
    p.add_argument("--n_train", type=int, default=4)
    p.add_argument("--T_train", type=int, default=1000)
    p.add_argument("--T_test", type=int, default=2000)
    p.add_argument("--t0_min", type=int, default=600)
    p.add_argument("--n_starts", type=int, default=5)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--one_step_n", type=int, default=20)
    p.add_argument("--report_h", nargs="+", type=int, default=[1, 10, 25, 50, 100])
    p.add_argument("--vpt_thresh", type=float, default=0.3)
    p.add_argument("--full_cap", type=int, default=5000)
    p.add_argument("--tabpfn_version", default="v2")
    p.add_argument("--out", default="pilot_results.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
