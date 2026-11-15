"""Closed-loop control of MuJoCo HalfCheetah with an in-context world model (CEM-MPC).

Environment  Gymnasium HalfCheetah-v5 (MuJoCo). Observation = qpos[1:] (8) + qvel (9) - the same
             layout as D4RL halfcheetah-medium-v2, whose transitions form the world model's pool.
             Flat ground, no early termination. dt = 0.05 s (frame_skip 5).
Reward       simulator reward (forward velocity - 0.1 |a|^2) for evaluation. Planning uses the same
             reward written in terms of the predicted next state: obs[8] (x velocity) - 0.1 |a|^2.
Planner      CEM over action sequences in [-1, 1]^6: n_cand candidates, horizon P, n_iter
             iterations, warm start from the shifted previous mean.
World models tabpfn  : TabPFN-3.5, one regressor per state dimension predicting the state change,
                       context selected once per control step:
                         trajretrieval = H pool rows nearest to the current (state, planned action)
                                         and to points on the previous plan's predicted path;
                         recent / random = last H / random H pool rows.
             mlp     : MLP (2 x 256) trained once on the offline pool.
             oracle  : the MuJoCo simulator itself (upper reference).
             random  : uniform random actions (lower reference).
Pool         n_train D4RL episodes + every transition observed during the episode (appended online).
Outputs      JSON with return, distance travelled, per-step time; optional MP4 video and PNG frames.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import pilot as P

ENVS = {  # gym id, D4RL file, action dim, obs dim, obs index of forward velocity, ctrl cost, healthy rule
    "halfcheetah": ("HalfCheetah-v5", "halfcheetah_medium-v2.npz", 6, 17, 8, 0.1, None),
    # Gymnasium v5 defaults: Hopper healthy if z > 0.7, |angle| < 0.2, |obs[1:]| < 100; Walker2d
    # healthy if 0.8 < z < 2.0 and |angle| < 1; healthy reward 1; ctrl cost 1e-3; episode ends on a fall
    "hopper": ("Hopper-v5", "hopper_medium-v2.npz", 3, 11, 5, 1e-3, "hopper"),
    "walker2d": ("Walker2d-v5", "walker2d_medium-v2.npz", 6, 17, 8, 1e-3, "walker2d"),
}
ENV = "halfcheetah"
GYM_ID, D4RL_FILE, ACT_DIM, OBS_DIM, VX, CTRL, HEALTHY = None, None, None, None, None, None, None


def configure(env):
    global ENV, GYM_ID, D4RL_FILE, ACT_DIM, OBS_DIM, VX, CTRL, HEALTHY
    ENV = env
    GYM_ID, f, ACT_DIM, OBS_DIM, VX, CTRL, HEALTHY = ENVS[env]
    D4RL_FILE = f"/scratch/naa9497/icwm/data/d4rl/{f}"


configure("halfcheetah")


FALL_PENALTY = 0.0      # planning-only cost of a (predicted) fall; set by --fall_penalty (0 = env reward only)


def healthy(S):
    """(n,) bool: the robot has not fallen, with Gymnasium's own rule on the observation"""
    if HEALTHY is None:
        return np.ones(len(S), bool)
    z, ang = S[:, 0], S[:, 1]
    if HEALTHY == "hopper":
        return (z > 0.7) & (np.abs(ang) < 0.2) & np.all(np.abs(S[:, 1:]) < 100, axis=1)
    return (z > 0.8) & (z < 2.0) & (np.abs(ang) < 1.0)


def plan_reward(S_next, A):
    """Gymnasium reward written in the predicted next state: forward velocity (+ healthy bonus) - ctrl"""
    r = S_next[:, VX] - CTRL * np.sum(A ** 2, axis=1)
    return r + (1.0 if HEALTHY else 0.0)


def offline_pool(n_train, rng):
    d = np.load(D4RL_FILE)
    obs, act, nxt = d["observations"], d["actions"], d["next_observations"]
    ends = np.where(d["terminals"] | d["timeouts"])[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    pick = rng.choice(len(ends), n_train, replace=False)
    idx = np.concatenate([np.arange(starts[i], ends[i] + 1) for i in pick])
    X = np.hstack([obs[idx], act[idx]]).astype(float)
    Y = (nxt[idx] - obs[idx]).astype(float)
    return X, Y


def nearest_rows(X, Q, H):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    d2 = np.min([((Z - (q - mu) / sd) ** 2).sum(1) for q in Q], axis=0)
    H = min(H, len(X))
    return np.argpartition(d2, H - 1)[:H]


class ContextModel:
    """Serial version: one TabPFN regressor per state dimension."""
    def __init__(self, args, seed, Xc, Yc):
        self.ms = [P.make_model("tabpfn", args, seed).fit(Xc, Yc[:, j]) for j in range(Yc.shape[1])]
    def step(self, S, A):
        X = np.hstack([S, A])
        return S + np.stack([m.predict(X) for m in self.ms], 1)


def _worker(conn, dims, version, fit_mode, seed):
    """Persistent worker owning the regressors of `dims` (same per-dimension models as serial)."""
    import argparse as _ap
    a = _ap.Namespace(tabpfn_version=version, tabpfn_fit_mode=fit_mode)
    ms = None
    while True:
        msg = conn.recv()
        if msg[0] == "fit":
            Xc, Yc = msg[1], msg[2]
            ms = [P.make_model("tabpfn", a, seed).fit(Xc, Yc[:, j]) for j in dims]
            conn.send("ok")
        elif msg[0] == "pred":
            conn.send(np.stack([m.predict(msg[1]) for m in ms], 1))
        else:
            break


class ParallelTabPFN:
    """The per-dimension TabPFN calls distributed over `n_workers` processes sharing one GPU.
    Predictions are identical to ContextModel (same models, same seed); only wall time changes."""
    def __init__(self, n_workers, version, fit_mode, seed, d=None):
        import multiprocessing as mp
        d = OBS_DIM if d is None else d; self.d = d
        ctx = mp.get_context("spawn")
        self.groups = [list(g) for g in np.array_split(np.arange(d), n_workers)]
        self.conns, self.procs = [], []
        for g in self.groups:
            a, b = ctx.Pipe()
            pr = ctx.Process(target=_worker, args=(b, g, version, fit_mode, seed), daemon=True)
            pr.start(); self.conns.append(a); self.procs.append(pr)
    def fit(self, Xc, Yc):
        for c in self.conns: c.send(("fit", Xc, Yc))
        for c in self.conns: assert c.recv() == "ok"
        return self
    def step(self, S, A):
        X = np.hstack([S, A]); out = np.empty((len(S), self.d))
        for c in self.conns: c.send(("pred", X))
        for c, g in zip(self.conns, self.groups): out[:, g] = c.recv()
        return S + out
    def close(self):
        for c in self.conns: c.send(("stop",))
        for pr in self.procs: pr.join(timeout=30)


class MLPModel:
    def __init__(self, X, Y, seed):
        from sklearn.neural_network import MLPRegressor
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-12; self.my, self.sy = Y.mean(0), Y.std(0) + 1e-12
        self.m = MLPRegressor(hidden_layer_sizes=(256, 256), early_stopping=True, max_iter=500,
                              random_state=seed).fit((X - self.mx) / self.sx, (Y - self.my) / self.sy)
    def step(self, S, A):
        X = np.hstack([S, A])
        return S + self.m.predict((X - self.mx) / self.sx) * self.sy + self.my


class OracleModel:
    """Exact simulator: each candidate is rolled out from the true current MuJoCo state."""
    def __init__(self, sim_env): self.e = sim_env
    def plan_returns(self, qpos, qvel, A):            # A: (n_cand, P, 6) -> returns (n_cand,)
        R = np.zeros(len(A))
        for i, seq in enumerate(A):
            self.e.set_state(qpos, qvel)
            for a in seq:
                _, r, term, _, _ = self.e.step(a)
                R[i] += r
                if term:                                  # fell: no further reward, minus the fall penalty
                    R[i] -= FALL_PENALTY
                    break
        return R


def cem(score, mean, args, rng):
    std = np.full((args.P, ACT_DIM), 0.5)
    for _ in range(args.n_iter):
        A = np.clip(mean + std * rng.normal(size=(args.n_cand, args.P, ACT_DIM)), -1, 1)
        J = score(A)
        el = A[np.argsort(-J)[: args.n_elite]]
        mean, std = el.mean(0), el.std(0) + 1e-3
    return mean


def model_score(model, s0):
    def score(A):
        S = np.repeat(s0[None], len(A), 0); R = np.zeros(len(A)); alive = np.ones(len(A), bool)
        for h in range(A.shape[1]):
            S = model.step(S, A[:, h]); S[~np.isfinite(S)] = 0.0
            R += alive * plan_reward(S, A[:, h])
            fell = alive & ~healthy(S)                    # predicted fall ends the reward (as in the env)
            R -= FALL_PENALTY * fell
            alive &= ~fell
        return R
    return score


def make_env(render):
    """Gymnasium v5 env of the configured robot; for video the model's own "track" camera (follows the
    centre of mass) - the v5 default free camera stays at the origin, so the robot leaves the picture."""
    import gymnasium as gym
    if not render:
        return gym.make(GYM_ID)
    return gym.make(GYM_ID, render_mode="rgb_array", camera_name="track", width=640, height=360)


def save_video(frames, video_path, dt):
    import imageio
    imageio.mimsave(video_path, frames, fps=int(round(1 / dt)))
    for k in np.linspace(0, len(frames) - 1, 5).astype(int):
        imageio.imwrite(str(Path(video_path).with_suffix(f".t{k:03d}.png")), frames[k])


def episode(args, controller, strategy, seed, video_path=None):
    import gymnasium as gym
    rng = np.random.default_rng(seed)
    env = make_env(video_path is not None)
    sim = gym.make(GYM_ID).unwrapped
    sim.reset(seed=seed)                                    # initialise the oracle's simulator copy
    obs, _ = env.reset(seed=seed)
    X, Y = offline_pool(args.n_train, rng)
    mlp = MLPModel(X, Y, seed) if controller == "mlp" else None
    oracle = OracleModel(sim) if controller == "oracle" else None
    mean = np.zeros((args.P, ACT_DIM)); model = None; par = None; queue = []
    ret, frames, secs, actions, xs = 0.0, [], [], [], []
    for t in range(args.T):
        tic = time.time()
        if controller == "random":
            a = rng.uniform(-1, 1, ACT_DIM)
        elif queue:
            a = queue.pop(0)                                  # execute the rest of the last plan
        else:
            if controller == "oracle":
                u = env.unwrapped
                qpos, qvel = u.data.qpos.copy(), u.data.qvel.copy()
                mean = cem(lambda A: oracle.plan_returns(qpos, qvel, A), mean, args, rng)
            else:
                if controller == "mlp":
                    model = mlp
                else:
                    if strategy == "trajretrieval":
                        Q = [np.concatenate([obs, mean[0]])]
                        if model is not None:                      # path of the shifted previous plan
                            Sp = obs[None]
                            for h in range(args.P):
                                if h % args.traj_stride == 0:
                                    Q.append(np.concatenate([Sp[0], mean[h]]))
                                Sp = model.step(Sp, mean[h:h + 1]); Sp[~np.isfinite(Sp)] = 0.0
                        idx = nearest_rows(X, np.array(Q), args.H)
                    elif strategy == "recent":
                        idx = np.arange(max(0, len(X) - args.H), len(X))
                    elif strategy == "random":
                        idx = rng.choice(len(X), min(args.H, len(X)), replace=False)
                    else:
                        raise ValueError(strategy)
                    if args.workers > 1:
                        if par is None:
                            par = ParallelTabPFN(args.workers, args.tabpfn_version, args.tabpfn_fit_mode, seed)
                        model = par.fit(X[idx], Y[idx])
                    else:
                        model = ContextModel(args, seed, X[idx], Y[idx])
                mean = cem(model_score(model, obs), mean, args, rng)
            a = mean[0].copy()
            queue = [mean[k].copy() for k in range(1, min(args.replan_every, args.P))]
        obs2, r, term, trunc, _ = env.step(a)
        X = np.vstack([X, np.concatenate([obs, a])[None]]); Y = np.vstack([Y, (obs2 - obs)[None]])
        ret += r; obs = obs2; secs.append(time.time() - tic)
        actions.append(np.asarray(a, float).tolist()); xs.append(float(env.unwrapped.data.qpos[0]))
        if controller != "random":
            mean = np.vstack([mean[1:], np.zeros((1, ACT_DIM))])
        if video_path:
            frames.append(env.render())
        print(f"  t={t:3d} reward={r:7.3f} x={env.unwrapped.data.qpos[0]:7.3f} ({secs[-1]:.1f}s)", flush=True)
        if (t + 1) % 100 == 0:                       # checkpoint: long runs never lose everything
            Path(f"{args.out}.s{seed}.partial.json").write_text(json.dumps(dict(
                args=vars(args), seed=seed, reset_seed=seed, t=t + 1, ret=float(ret), actions=actions, x=xs)))
        if term:                                     # Hopper / Walker2d fell: the episode ends (as in Gymnasium)
            print(f"  fell at step {t + 1}", flush=True)
            break
    res = dict(env=ENV, steps_alive=len(actions), fell=bool(len(actions) < args.T),
               controller=controller, strategy=strategy, seed=seed, T=args.T, ret=float(ret),
               distance=float(env.unwrapped.data.qpos[0]), secs_per_step=float(np.mean(secs)),
               workers=args.workers, replan_every=args.replan_every, tabpfn_version=args.tabpfn_version,
               reset_seed=seed, actions=actions, x=xs)          # actions: replay / re-render without planning
    if video_path:
        save_video(frames, video_path, env.unwrapped.dt)
    if par is not None: par.close()
    env.close(); sim.close()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="halfcheetah", choices=list(ENVS))
    p.add_argument("--controller", default="tabpfn", choices=["tabpfn", "mlp", "oracle", "random"])
    p.add_argument("--strategy", default="trajretrieval", choices=["trajretrieval", "recent", "random", "-"])
    p.add_argument("--seed_ids", nargs="+", type=int, default=[0])
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--P", type=int, default=15)
    p.add_argument("--n_cand", type=int, default=128)
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--n_elite", type=int, default=12)
    p.add_argument("--H", type=int, default=500)
    p.add_argument("--traj_stride", type=int, default=5)
    p.add_argument("--n_train", type=int, default=10)
    p.add_argument("--tabpfn_version", default="v3.5")
    p.add_argument("--tabpfn_fit_mode", default="fit_preprocessors")
    p.add_argument("--workers", type=int, default=1, help="parallel processes for the per-dim TabPFN calls")
    p.add_argument("--replan_every", type=int, default=1, help="replan every k control steps")
    p.add_argument("--video_dir", default=None)
    p.add_argument("--out", default="cheetah_results.json")
    p.add_argument("--fall_penalty", type=float, default=0.0, help="planning cost of a predicted fall")
    args = p.parse_args()
    configure(args.env)
    global FALL_PENALTY
    FALL_PENALTY = args.fall_penalty
    res = []
    for seed in args.seed_ids:
        vp = None
        if args.video_dir:
            Path(args.video_dir).mkdir(parents=True, exist_ok=True)
            pre = "cheetah" if args.env == "halfcheetah" else args.env
            vp = str(Path(args.video_dir) / f"{pre}_{args.controller}_{args.strategy}_s{seed}.mp4")
        r = episode(args, args.controller, args.strategy, seed, vp); res.append(r)
        print(f"[{args.env} s{seed}] {args.controller} {args.strategy}: return={r['ret']:.1f} "
              f"distance={r['distance']:.2f} m steps={r['steps_alive']} ({r['secs_per_step']:.1f}s/step)", flush=True)
        Path(args.out).write_text(json.dumps(dict(args=vars(args), results=res), indent=1))


if __name__ == "__main__":
    main()
