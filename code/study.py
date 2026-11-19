"""Full open-loop study: in-context world models with retrieval context (extends pilot.py).

Adds to the pilot (pilot.py stays frozen for reproducibility):
  data      lorenz / pendulum with optional observation noise (--obs_noise, in units of the per-
            dimension state std; noise enters every observed state; errors are measured against
            the CLEAN trajectory), and D4RL hopper / halfcheetah medium-v2 (real MuJoCo data,
            11/17-dim state, 3/6-dim action; loaded from npz converted from the official hdf5).
  models    gp     : Gaussian process per state dimension, kernel C*RBF(ARD)+White. Kernel
                     hyperparameters are fitted ONCE per rollout start on a random subset
                     (--gp_fit_n rows) of the pool and then kept fixed, so refitting on a new
                     context each step is a single Cholesky solve.
            mlp    : trained neural network (2x256 ReLU, Adam, early stopping), multi-output,
                     trained once per rollout start on the whole pool (strategy "trained").
            tabpfn : as in the pilot; --tabpfn_n_estimators sets the ensemble size (cost ablation;
                     default = TabPFN default).
            dynapfn: DynaPFN-style baseline (feature library G, absolute targets, TabPFN v2, 1
                     estimator), usable with any context strategy.
            tabpfn_stack : one TabPFN call for all state dimensions (one-hot dimension feature,
                     per-dimension standardised targets) - cost reduction.
  protocol  pool = n_train trajectories/episodes + the test trajectory's history before t0;
            open-loop rollout with recorded actions; longer horizons via --horizon / --report_h.
Metrics as in the pilot (normalised RMSE, valid prediction time).
"""
import argparse
import json
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np

import pilot as P

D4RL = {"hopper": "hopper_medium-v2.npz", "halfcheetah": "halfcheetah_medium-v2.npz",
        "walker2d": "walker2d_medium-v2.npz"}
EXTRA = ["lorenz96", "camels", "pendulum_box"]


def lagged(S, U, L):
    """Transitions with a delay-embedded state: X_t = [S_t, S_{t-1}, ..., S_{t-L+1}, U_t],
    Y_t = S_{t+1} - S_t, rows aligned with t (rows t < L-1 are NaN). L=1 equals P.transitions."""
    T = len(S) - 1
    parts = []
    for k in range(L):
        Sk = np.full((T, S.shape[1]), np.nan); Sk[k:] = S[:T - k]; parts.append(Sk)
    X = np.hstack(parts + ([U[:T]] if U.shape[1] else []))
    return X, S[1:] - S[:-1]


def valid(X, Y):
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    return X[ok], Y[ok]


def simulate_l96(T, rng, d, F=8.0, dt=0.01, burn=2000):
    f = lambda x: (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    x = F + 0.01 * rng.normal(size=d); S = np.empty((T + 1, d))
    for t in range(burn + T + 1):
        if t >= burn: S[t - burn] = x
        if t < burn + T:
            k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return S, np.zeros((T, 0))


_CAMELS = {}


def camels_seed(args, rng):
    """Basin = sorted CAMELS basin number `seed` (18 basins, one per HUC region). State = daily
    flow q (mm/day), delay-embedded with --lags days; input U_t = Daymet weather of day t+1 (known
    forcing, the analogue of recorded actions). Train = water years 1981-2005, test = 2006-2014."""
    if not _CAMELS:
        _CAMELS.update(dict(np.load(Path(args.camels_file), allow_pickle=False)))
    b = sorted(str(x) for x in _CAMELS["basins"])[args.cur_seed % 18]
    dates = _CAMELS[f"{b}_dates"].astype(str); q = _CAMELS[f"{b}_q"][:, None]; Fz = _CAMELS[f"{b}_forcing"]
    cut = int(np.searchsorted(dates, "2005-10-01"))
    if args.camels_features == "ext":
        # longer memory from the (known, exogenous) weather, aligned with the TARGET day t+1:
        # precipitation totals over the last 7 / 30 / 90 days and mean tmax over 7 / 30 days,
        # each window ending on the target day (Daymet forcing is observed for every day).
        def roll(x, w, fn):
            c = np.concatenate([[0.0], np.cumsum(x)])
            out = np.full(len(x), np.nan); out[w - 1:] = (c[w:] - c[:-w]) / (1.0 if fn == "sum" else w)
            return out
        pr, tx = Fz[:, 0], Fz[:, 1]
        Fz = np.column_stack([Fz, roll(pr, 7, "sum"), roll(pr, 30, "sum"), roll(pr, 90, "sum"),
                              roll(tx, 7, "mean"), roll(tx, 30, "mean")])
    S_tr, U_tr = q[:cut], Fz[1:cut]
    S_te, U_te = q[cut:], Fz[cut + 1:]
    Xtr, Ytr = valid(*lagged(S_tr, U_tr, args.lags))
    sd = S_tr.std(0) + 1e-8
    return Xtr, Ytr, S_te, S_te, U_te, sd


# ----------------------------------------------------------------------------- data
BOX_RESTITUTION = 0.8


def simulate_box(T, rng, wall):
    """Forced damped pendulum (pilot.pendulum_f, RK4, dt 0.05, random forcing N(0,1)) started at small
    angles |theta| < 0.3; with wall > 0 two hard walls at +-wall rad: when a step ends beyond a wall the
    angle is reflected back and the angular velocity reversed with restitution 0.8 (impact). wall = 0:
    the same pendulum without walls (smooth). Controlled test of smooth vs non-smooth dynamics."""
    dt = P.SYSTEMS["pendulum"]["dt"]
    x = np.array([rng.uniform(-0.3, 0.3), rng.normal(0, 1)])
    U = rng.normal(0, 1, (T, 1))
    S = np.empty((T + 1, 2)); S[0] = x
    for t in range(T):
        x = P.rk4(P.pendulum_f, x, U[t], dt)
        if wall > 0 and abs(x[0]) > wall:
            x = np.array([np.sign(x[0]) * (2 * wall - abs(x[0])), -BOX_RESTITUTION * x[1]])
        S[t + 1] = x
    return S, U


def synthetic_seed(args, rng):
    """Returns pool transitions from training trajectories, clean + observed test trajectory."""
    # (system parameters come from pilot.SYSTEMS or simulate_l96)
    def observe(S, sd):
        return S + args.obs_noise * sd * rng.normal(size=S.shape) if args.obs_noise > 0 else S
    sim = ((lambda T: simulate_l96(T, rng, args.l96_dim)) if args.system == "lorenz96"
           else (lambda T: simulate_box(T, rng, args.wall)) if args.system == "pendulum_box"
           else (lambda T: P.simulate(args.system, T, rng)))
    trains = [sim(args.T_train) for _ in range(args.n_train)]
    S_te, U_te = sim(args.T_test)
    sd = np.vstack([S for S, _ in trains]).std(0)
    Xs, Ys = [], []
    for S, U in trains:
        x, y = valid(*lagged(observe(S, sd), U, args.lags)); Xs.append(x); Ys.append(y)
    return np.vstack(Xs), np.vstack(Ys), S_te, observe(S_te, sd), U_te, sd


_D4RL_CACHE = {}


def d4rl_episodes(name, data_dir):
    if name not in _D4RL_CACHE:
        d = np.load(Path(data_dir) / D4RL[name])
        obs, act, nxt = d["observations"], d["actions"], d["next_observations"]
        ends = np.where(d["terminals"] | d["timeouts"])[0]
        eps, start = [], 0
        for e in ends:
            S = np.vstack([obs[start:e + 1], nxt[e:e + 1]]).astype(float)   # T+1 states
            eps.append((S, act[start:e + 1].astype(float))); start = e + 1
        _D4RL_CACHE[name] = eps
    return _D4RL_CACHE[name]


def d4rl_seed(args, rng):
    eps = d4rl_episodes(args.system, args.data_dir)
    need = args.t0_min + args.horizon + 1
    long_ok = [i for i, (S, _) in enumerate(eps) if len(S) - 1 >= need]
    te = rng.choice(long_ok)
    others = [i for i in range(len(eps)) if i != te]
    tr = rng.choice(others, size=args.n_train, replace=False)
    Xs, Ys = [], []
    for i in tr:
        x, y = valid(*lagged(*eps[i], args.lags)); Xs.append(x); Ys.append(y)
    S_te, U_te = eps[te]
    sd = np.vstack([eps[i][0] for i in tr]).std(0) + 1e-8
    return np.vstack(Xs), np.vstack(Ys), S_te, S_te, U_te, sd


# ----------------------------------------------------------------------------- extra models
CODE_VERSION = "2026-11-18j"   # j: sliding: a diverged (NaN) or unfittable window ends the rollout as diverged (NaN); results otherwise identical to i. i: new strategy "sliding" (Lee et al. 2611.11921 window); nothing else changed. h: new system pendulum_box (--wall); nothing else changed. c: GP scaling fix; d: --target abs; e: --retrieval_metric; f: --camels_features ext;
                               # g: GP kernel fit with optimizer restarts (single start collapsed to an
                               #    all-noise fit on lagged noisy inputs). Non-GP behaviour unchanged.


class RPool(P.Pool):
    """pilot.Pool with a choice of retrieval distance:
    std        : Euclidean on inputs standardised by the pool (default, = pilot.Pool.nearest)
    state_only : as std, but on the state / lag columns only (actions or forcing ignored)
    raw        : Euclidean on unscaled inputs"""
    def __init__(self, X, Y, metric="std", n_state=None):
        super().__init__(X, Y); self.metric, self.n_state = metric, n_state
        if metric == "state_only":
            self.Zs = self.Z[:, :n_state]

    def nearest(self, H, q):
        H = min(H, len(self.X))
        if self.metric == "std":
            return super().nearest(H, q)
        if self.metric == "state_only":
            d2 = ((((q - self.mu) / self.sd)[:n] - self.Zs) ** 2).sum(1) if (n := self.n_state) else None
        else:
            d2 = ((q - self.X) ** 2).sum(1)
        return np.argpartition(d2, H - 1)[:H]


class FixedKernelGP:
    """GP whose kernel hyperparameters were fitted elsewhere (fit_gp_kernels). Inputs and target
    are scaled with the statistics used during that kernel fit (NOT re-standardised on the
    context: a local retrieved context has a much smaller spread, which would silently rescale the
    learned length scales and amplitude). fit = one Cholesky solve on the context."""
    def __init__(self, kernel, mx, sx, my, sy): self.kernel, self.mx, self.sx, self.my, self.sy = kernel, mx, sx, my, sy
    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        self.m = GaussianProcessRegressor(kernel=self.kernel, optimizer=None, normalize_y=False)
        self.m.fit((X - self.mx) / self.sx, (y - self.my) / self.sy)
        return self
    def predict(self, X): return self.m.predict((X - self.mx) / self.sx) * self.sy + self.my


def fit_gp_kernels(Xp, Yp, n, rng, restarts=5):
    """Per output dimension: C*RBF(ARD)+White fitted by marginal likelihood on a random subset of
    n pool rows, with `restarts` random restarts of the optimiser (best log marginal likelihood
    kept); returns (kernel, mx, sx, my, sy) with the scaling used for the fit."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    idx = rng.choice(len(Xp), min(n, len(Xp)), replace=False)
    X = Xp[idx]; mx, sx = X.mean(0), X.std(0) + 1e-12; Z = (X - mx) / sx
    out = []
    for j in range(Yp.shape[1]):
        y = Yp[idx, j]; my, sy = y.mean(), y.std() + 1e-12
        k = ConstantKernel(1.0) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(1e-3, (1e-8, 1e1))
        kern = GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=restarts,
                                        random_state=int(rng.integers(2 ** 31))).fit(Z, (y - my) / sy).kernel_
        out.append((kern, mx, sx, my, sy))
    return out


def train_mlp(Xp, Yp, seed):
    from sklearn.neural_network import MLPRegressor
    mx, sx = Xp.mean(0), Xp.std(0) + 1e-12; my, sy = Yp.mean(0), Yp.std(0) + 1e-12
    m = MLPRegressor(hidden_layer_sizes=(256, 256), early_stopping=True, max_iter=500, random_state=seed)
    Z = (Yp - my) / sy
    m.fit((Xp - mx) / sx, Z.ravel() if Z.shape[1] == 1 else Z)
    return lambda X: m.predict((X - mx) / sx).reshape(len(X), -1) * sy + my


def dynapfn_library(S, degree=4):
    """DynaPFN feature library G (Roverato et al., NeurIPS 2026; official code
    github.com/chiararov/DynaPFN, utils.extend_features): cos and sin of every state dimension
    plus all monomials of degree 2..degree (combinations with replacement)."""
    from itertools import combinations_with_replacement
    cols = [S, np.cos(S), np.sin(S)]
    for deg in range(2, degree + 1):
        for combo in combinations_with_replacement(range(S.shape[1]), deg):
            cols.append(np.prod(S[:, list(combo)], axis=1, keepdims=True))
    return np.hstack(cols)


class Predictor:
    """Per-start state: GP kernels and the trained MLP are built once per (seed, t0)."""
    def __init__(self, args, seed, pool, rng):
        self.args, self.seed, self.pool, self.rng = args, seed, pool, rng
        self.gp_k = None; self.mlp = None

    def __call__(self, model, idx, Xq):
        a, pool = self.args, self.pool
        if model == "mlp":
            if self.mlp is None: self.mlp = train_mlp(pool.X, pool.Y, self.seed)
            return self.mlp(Xq)
        if model == "gp":
            if self.gp_k is None: self.gp_k = fit_gp_kernels(pool.X, pool.Y, a.gp_fit_n, self.rng, a.gp_restarts)
            Xc, Yc = pool.X[idx], pool.Y[idx]
            return np.stack([FixedKernelGP(*kp).fit(Xc, Yc[:, j]).predict(Xq)
                             for j, kp in enumerate(self.gp_k)], 1)
        if model == "dynapfn":
            # DynaPFN-style regressor (official defaults): inputs = feature library G of the state
            # (+ actions, if any), target = ABSOLUTE next state per dimension, TabPFN v2 with one
            # estimator. Differences from the official code: lag b=1 (our step-wise protocol; the
            # paper's default is b=5), no timestamp feature (the pool mixes trajectories), and the
            # context comes from the chosen strategy (recent window ~ official C-point context).
            Xc, Yc = pool.X[idx], pool.Y[idx]; d = Yc.shape[1]
            feat = lambda X: np.hstack([dynapfn_library(X[:, :d], a.dynapfn_degree), X[:, d:]])
            Fc, Fq = feat(Xc), feat(Xq)
            out = []
            for j in range(d):
                m = P.TabPFN("v2", self.seed); m.m.set_params(n_estimators=1)
                out.append(m.fit(Fc, Xc[:, j] + Yc[:, j]).predict(Fq) - Xq[:, j])
            return np.stack(out, 1)
        if model == "tabpfn_stack":
            # ONE TabPFN call for all d state dimensions: each context row is repeated d times with
            # a one-hot dimension tag; targets are standardised per dimension on the context so
            # TabPFN's single target normalisation does not favour large-scale dimensions.
            Xc, Yc = pool.X[idx], pool.Y[idx]; d = Yc.shape[1]
            my, sy = Yc.mean(0), Yc.std(0) + 1e-12
            E = np.eye(d)
            Xs = np.vstack([np.hstack([Xc, np.repeat(E[j:j + 1], len(Xc), 0)]) for j in range(d)])
            ys = ((Yc - my) / sy).T.ravel()
            Xqs = np.vstack([np.hstack([Xq, np.repeat(E[j:j + 1], len(Xq), 0)]) for j in range(d)])
            m = P.TabPFN(a.tabpfn_version, self.seed)
            if a.tabpfn_n_estimators: m.m.set_params(n_estimators=a.tabpfn_n_estimators)
            out = m.fit(Xs, ys).predict(Xqs).reshape(d, len(Xq)).T
            return out * sy + my
        if model == "tabpfn" and a.tabpfn_n_estimators:
            assert a.target == "delta", "--target abs is implemented for the default TabPFN path only"
            Xc, Yc = pool.X[idx], pool.Y[idx]
            out = []
            for j in range(Yc.shape[1]):
                m = P.TabPFN(a.tabpfn_version, self.seed)
                m.m.set_params(n_estimators=a.tabpfn_n_estimators)
                out.append(m.fit(Xc, Yc[:, j]).predict(Xq))
            return np.stack(out, 1)
        # generic per-dimension models; a FIXED context (same idx object every step) is fitted once
        # (identity check on the idx object we hold a reference to, so ids cannot be recycled).
        # --target abs: regress the ABSOLUTE next state s_{t+1} = s_t + delta (first d input
        # columns are s_t) and return prediction - s_t, so the caller still receives a change.
        d = pool.Y.shape[1]; absolute = a.target == "abs"
        if getattr(self, "_fit_model", None) != model or getattr(self, "_fit_idx", None) is not idx:
            Xc, Yc = pool.X[idx], pool.Y[idx]
            Tc = Yc + Xc[:, :d] if absolute else Yc
            self._fit = [P.make_model(model, a, self.seed).fit(Xc, Tc[:, j]) for j in range(d)]
            self._fit_model, self._fit_idx = model, idx
        out = np.stack([m.predict(Xq) for m in self._fit], 1)
        return out - Xq[:, :d] if absolute else out


# ----------------------------------------------------------------------------- experiment
def run(args):
    results = []
    for seed in args.seed_ids:
        rng = np.random.default_rng(1000 * seed + 7); args.cur_seed = seed
        get = (d4rl_seed if args.system in D4RL else camels_seed if args.system == "camels"
               else synthetic_seed)
        Xtr, Ytr, S_clean, S_obs, U_te, state_sd = get(args, rng)
        X_te, Y_te = lagged(S_obs, U_te, args.lags)               # observed (possibly noisy)
        L = args.lags
        last = len(X_te) - args.horizon - 1
        starts = np.linspace(args.t0_min, last, args.n_starts).astype(int)
        for t0 in starts:
            X_pool = np.vstack([Xtr, X_te[L - 1:t0]]); Y_pool = np.vstack([Ytr, Y_te[L - 1:t0]])
            hist_idx = np.arange(len(Xtr), len(X_pool))
            pool = RPool(X_pool, Y_pool, args.retrieval_metric, n_state=Y_pool.shape[1] * L)
            pred = Predictor(args, seed, pool, np.random.default_rng(seed * 104729 + int(t0)))
            for strategy in args.strategies:
                for H in (args.H if strategy not in ("full", "trained") else [0]):
                    if strategy in ("recent", "sliding") and H > len(hist_idx):
                        continue
                    for model in args.models:
                        if (strategy == "trained") != (model == "mlp"):
                            continue                     # the MLP is only a trained global model
                        if strategy == "full" and model in ("parrot", "knn", "zero", "gp"):
                            continue
                        crng = np.random.default_rng(seed * 7919 + int(t0))
                        tic = time.time()
                        s = S_obs[t0].copy(); err_roll, err_one, sims = [], [], []
                        past = [S_obs[t0 - k].copy() for k in range(L - 1, 0, -1)]   # oldest first
                        fixed = None
                        if strategy in ("recent", "random", "full"):
                            fixed = P.select(strategy, pool, H, hist_idx, None, crng, args.full_cap)
                        if strategy == "sliding":
                            # sliding window of Lee et al. (2611.11921): starts as the H latest observed
                            # transitions; after every step the oldest row is dropped and the query with
                            # the PREDICTED change is appended, so the window fills with the model's own
                            # predictions. Only for models that use nothing but the context.
                            assert model in ("tabpfn", "dynapfn", "locallin", "knn", "parrot"), model
                            w0 = P.select("recent", pool, H, hist_idx, None, crng, args.full_cap)
                            Xw, Yw = pool.X[w0].copy(), pool.Y[w0].copy()
                        for h in range(args.horizon):
                            u = U_te[t0 + h]
                            # lags come from the ROLLED-OUT trajectory (true values before t0)
                            q = np.concatenate([s] + past[::-1] + [u])[None]
                            q_true = X_te[t0 + h][None]
                            one = h < args.one_step_n
                            if strategy == "trained":
                                d = pred(model, None, np.vstack([q, q_true]))
                                dq, d1 = d[:1], d[1]
                            elif strategy == "sliding" and not np.all(np.isfinite(s)):
                                # rollout already diverged: its error stays NaN for the remaining steps
                                # (as for the other strategies); the window would otherwise receive NaN
                                # targets, on which the model cannot be fitted
                                dq = np.full((1, len(s)), np.nan); d1 = np.full(len(s), np.nan) if one else None
                            elif strategy == "sliding":
                                Xq2 = np.vstack([q, q_true]) if one else q
                                if not (np.all(np.isfinite(Xw)) and np.all(np.isfinite(Yw))):
                                    print(f"[sliding] non-finite window at h={h}", flush=True)
                                if model == "tabpfn" and not args.tabpfn_n_estimators and args.target == "delta":
                                    # low-memory equivalent of the generic Predictor path (same model, seed,
                                    # data): fit, predict and free one state dimension at a time, so only one
                                    # TabPFN copy is on the GPU (the window changes every step anyway)
                                    d = np.empty((len(Xq2), Yw.shape[1]))
                                    try:
                                        for j in range(Yw.shape[1]):
                                            m_ = P.make_model("tabpfn", args, seed).fit(Xw, Yw[:, j])
                                            d[:, j] = m_.predict(Xq2); del m_
                                    except ValueError as e:
                                        # the window (filled with the model's own predictions) became
                                        # unfittable: counted as a diverged rollout (NaN from here on)
                                        print(f"[sliding] model failed at h={h}: {e}; window |Y| max "
                                              f"{np.nanmax(np.abs(Yw)):.3g}, sd per dim {np.nanstd(Yw, 0)}, "
                                              f"|X| max {np.nanmax(np.abs(Xw)):.3g}", flush=True)
                                        d = np.full((len(Xq2), Yw.shape[1]), np.nan)
                                else:
                                    pred.pool = SimpleNamespace(X=Xw, Y=Yw)    # context = current window
                                    try:
                                        d = pred(model, np.arange(len(Xw)), Xq2)
                                    finally:
                                        pred.pool = pool
                                dq, d1 = d[:1], (d[1] if one else None)
                                Xw = np.vstack([Xw[1:], q]); Yw = np.vstack([Yw[1:], dq])
                            elif fixed is not None:
                                d = pred(model, fixed, np.vstack([q, q_true]) if one else q)
                                dq, d1 = d[:1], (d[1] if one else None)
                            else:
                                if h % args.retrieve_every == 0:
                                    idx = pool.nearest(H, q[0])      # kept for retrieve_every steps
                                dq = pred(model, idx, q)
                                d1 = pred(model, pool.nearest(H, q_true[0]), q_true)[0] if one else None
                            if one:
                                err_one.append((d1 - Y_te[t0 + h]) / state_sd)
                            if L > 1:
                                past = (past + [s.copy()])[1:]
                            s = s + dq[0]
                            if not np.all(np.isfinite(s)):
                                s = np.full_like(s, np.nan)
                            err_roll.append(np.sqrt(np.mean(((s - S_clean[t0 + h + 1]) / state_sd) ** 2)))
                            sims.append(s.copy())
                        err_roll = np.array(err_roll)
                        bad = np.where(~(err_roll <= args.vpt_thresh))[0]
                        # Nash-Sutcliffe efficiency of the whole rolled-out window, per dimension,
                        # then averaged (1 = perfect, 0 = as good as the window mean, NaN if diverged)
                        obs_w = S_clean[t0 + 1:t0 + 1 + args.horizon]; sim_w = np.array(sims)
                        nse = 1 - ((sim_w - obs_w) ** 2).sum(0) / (((obs_w - obs_w.mean(0)) ** 2).sum(0) + 1e-12)
                        results.append(dict(
                            system=args.system, obs_noise=args.obs_noise, seed=seed, t0=int(t0),
                            lags=L, l96_dim=args.l96_dim if args.system == "lorenz96" else None,
                            tabpfn_version=args.tabpfn_version, code_version=CODE_VERSION,
                            target=args.target, retrieval_metric=args.retrieval_metric,
                            camels_features=args.camels_features if args.system == "camels" else None,
                            wall=args.wall if args.system == "pendulum_box" else None,
                            strategy=strategy, H=int(H), model=model, secs=time.time() - tic,
                            n_est=args.tabpfn_n_estimators, retrieve_every=args.retrieve_every,
                            one_step_nrmse=float(np.sqrt(np.mean(np.square(err_one)))),
                            rollout_nrmse={str(k): float(err_roll[k - 1]) for k in args.report_h
                                           if k <= args.horizon},
                            vpt_steps=int(bad[0]) if len(bad) else args.horizon,
                            nse=float(np.mean(nse)) if np.all(np.isfinite(nse)) else None))
                        r = results[-1]
                        print(f"[{args.system} n{args.obs_noise} s{seed} t0={t0}] {strategy:9s} "
                              f"H={H:4d} {model:8s} 1step={r['one_step_nrmse']:.4f} "
                              f"vpt={r['vpt_steps']:4d} ({r['secs']:.1f}s)", flush=True)
            Path(args.out).write_text(json.dumps(dict(args=vars(args), results=results), indent=1))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="lorenz", choices=list(P.SYSTEMS) + list(D4RL) + EXTRA)
    p.add_argument("--lags", type=int, default=1, help="delay embedding of the state (days for camels)")
    p.add_argument("--retrieval_metric", default="std", choices=["std", "state_only", "raw"])
    p.add_argument("--target", default="delta", choices=["delta", "abs"],
                   help="generic models (tabpfn, locallin, knn, parrot) regress the change (delta) or "
                        "the absolute next state (abs); gp / mlp / dynapfn are unaffected")
    p.add_argument("--l96_dim", type=int, default=20)
    p.add_argument("--wall", type=float, default=0.0, help="pendulum_box: wall angle (rad), 0 = no walls")
    p.add_argument("--camels_file", default="/scratch/naa9497/icwm/data/camels/camels18.npz")
    p.add_argument("--camels_features", default="basic", choices=["basic", "ext"])
    p.add_argument("--models", nargs="+", default=["tabpfn", "gp", "mlp", "locallin", "knn", "parrot"])
    p.add_argument("--strategies", nargs="+", default=["recent", "random", "retrieval", "full", "trained"],
                   help="also: sliding (window of Lee et al. that fills with predictions)")
    p.add_argument("--H", nargs="+", type=int, default=[100, 500])
    p.add_argument("--seed_ids", nargs="+", type=int, default=[0])
    p.add_argument("--obs_noise", type=float, default=0.0)
    p.add_argument("--data_dir", default="/scratch/naa9497/icwm/data/d4rl")
    p.add_argument("--n_train", type=int, default=4)
    p.add_argument("--T_train", type=int, default=1000)
    p.add_argument("--T_test", type=int, default=2000)
    p.add_argument("--t0_min", type=int, default=600)
    p.add_argument("--n_starts", type=int, default=5)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--one_step_n", type=int, default=20)
    p.add_argument("--report_h", nargs="+", type=int, default=[1, 10, 50, 100, 250, 500])
    p.add_argument("--vpt_thresh", type=float, default=0.3)
    p.add_argument("--full_cap", type=int, default=5000)
    p.add_argument("--gp_fit_n", type=int, default=500)
    p.add_argument("--gp_restarts", type=int, default=5)
    p.add_argument("--dynapfn_degree", type=int, default=4)
    p.add_argument("--retrieve_every", type=int, default=1, help="re-select retrieval context every k steps")
    p.add_argument("--tabpfn_version", default="v3.5")
    p.add_argument("--tabpfn_n_estimators", type=int, default=0, help="0 = TabPFN default")
    p.add_argument("--out", default="study_results.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
