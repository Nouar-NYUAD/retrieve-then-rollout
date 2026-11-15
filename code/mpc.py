"""Stage 2: closed-loop planning with an in-context world model (swing-up tasks).

Tasks
  pendulum : pilot.py pendulum (theta=0 hanging, pi upright), |u| <= 4, dt=0.05.
             NOTE: its one-step change is LINEAR in (sin, cos, omega, u), so a global linear fit is
             already exact -> kept only as a sanity task, it cannot rank world models.
  cartpole : frictionless cart-pole (Barto et al. 1983 equations), m_cart=1, m_pole=0.1,
             half-length l=0.5, g=9.8, theta=0 UPRIGHT, |F| <= 10, dt=0.05. Starts hanging
             (theta=pi). Dynamics are nonlinear in the inputs (sin*cos*omega^2 terms, state-
             dependent inertia), so it discriminates world models.
Costs     pendulum: d^2 + 0.1 omega^2 + 0.001 u^2, d = wrap(theta - pi)
          cartpole: wrap(theta)^2 + 0.1 x^2 + 0.01 (xdot^2 + thetadot^2) + 1e-4 F^2
Planner   CEM over action sequences: n_cand candidates, plan horizon P, n_iter iterations,
          n_elite elites, warm start from the shifted previous mean.
Model     at each control step the context is selected ONCE (retrieval: the H pool rows closest
          to the current state paired with any of 5 actions spread over [-u_max, u_max];
          trajretrieval: also closest to points every traj_stride steps along the previous plan's
          path predicted by the previous model; hybrid: H/2 retrieval + H/2 random), each
          per-dimension regressor is fitted once, then all candidate rollouts are predicted in
          batch (n_cand rows per planning time step). Angles enter as (sin, cos).
Data      offline pool of random-action transitions (n_train trajectories from random states) +
          every transition observed during the episode (appended online).
Metrics   episode cost, fraction of steps upright (|angle from upright| < 0.3 rad), first step
          upright, wall-clock per control step. Oracle = same CEM with the true dynamics.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import pilot as P


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rk4(f, S, U, dt):
    k1 = f(S, U); k2 = f(S + dt / 2 * k1, U); k3 = f(S + dt / 2 * k2, U); k4 = f(S + dt * k3, U)
    return S + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ----------------------------------------------------------------------------- tasks
class Pendulum:
    d, dt, u_max = 2, 0.05, 4.0
    @staticmethod
    def f(S, U):
        return np.stack([S[:, 1], -9.81 * np.sin(S[:, 0]) - 0.1 * S[:, 1] + U[:, 0]], 1)
    @staticmethod
    def features(S):
        return np.stack([np.sin(S[:, 0]), np.cos(S[:, 0]), S[:, 1]], 1)
    @staticmethod
    def cost(S, U):
        return wrap(S[..., 0] - np.pi) ** 2 + 0.1 * S[..., 1] ** 2 + 0.001 * U[..., 0] ** 2
    @staticmethod
    def upright(S):
        return np.abs(wrap(S[:, 0] - np.pi)) < 0.3
    @staticmethod
    def init(rng):
        return np.array([rng.normal(0, 0.1), 0.0])
    def random_state(self, rng):
        return np.array([rng.uniform(-np.pi, np.pi), rng.normal(0, 1)])


class CartPole:
    d, dt, u_max = 4, 0.05, 10.0
    mc, mp, l, g = 1.0, 0.1, 0.5, 9.8
    @classmethod
    def f(cls, S, U):
        x, xd, th, thd = S.T; F = U[:, 0]; M = cls.mc + cls.mp
        s, c = np.sin(th), np.cos(th)
        temp = (F + cls.mp * cls.l * thd ** 2 * s) / M
        thdd = (cls.g * s - c * temp) / (cls.l * (4.0 / 3.0 - cls.mp * c ** 2 / M))
        xdd = temp - cls.mp * cls.l * thdd * c / M
        return np.stack([xd, xdd, thd, thdd], 1)
    @staticmethod
    def features(S):
        return np.stack([S[:, 0], S[:, 1], np.sin(S[:, 2]), np.cos(S[:, 2]), S[:, 3]], 1)
    @staticmethod
    def cost(S, U):
        return (wrap(S[..., 2]) ** 2 + 0.1 * S[..., 0] ** 2
                + 0.01 * (S[..., 1] ** 2 + S[..., 3] ** 2) + 1e-4 * U[..., 0] ** 2)
    @staticmethod
    def upright(S):
        return np.abs(wrap(S[:, 2])) < 0.3
    @staticmethod
    def init(rng):
        return np.array([0.0, 0.0, np.pi + rng.normal(0, 0.1), 0.0])
    def random_state(self, rng):
        return np.array([rng.uniform(-1, 1), rng.normal(0, 1), rng.uniform(-np.pi, np.pi), rng.normal(0, 2)])


class Acrobot:
    """Acrobot (two-link, torque on the second joint only): Sutton & Barto / Gymnasium "book"
    dynamics, m1=m2=1, l1=1, lc1=lc2=0.5, I1=I2=1, g=9.8; theta = 0 hanging down. Continuous torque
    |u| <= u_max, RK4 with dt=0.05 (Gymnasium uses dt=0.2 with 3 discrete torques). Swing-up:
    start hanging, goal both links upright (theta1 = pi, theta2 = 0). Cost = (2 - tip height)^2
    + small velocity / torque terms: the angle-based cost and u_max=4 did not let even the oracle
    swing up (tested); with this cost, u_max=8 and plan horizon P=40 the oracle reaches the top at
    step ~75 and is upright 70% of a 250-step episode (seeds 0-2)."""
    d, dt, u_max = 4, 0.05, 8.0
    m1 = m2 = 1.0; l1 = 1.0; lc1 = lc2 = 0.5; I1 = I2 = 1.0; g = 9.8
    @classmethod
    def f(cls, S, U):
        th1, th2, w1, w2 = S.T; a = U[:, 0]
        m1, m2, l1, lc1, lc2, I1, I2, g = cls.m1, cls.m2, cls.l1, cls.lc1, cls.lc2, cls.I1, cls.I2, cls.g
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(th2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(th2)) + I2
        phi2 = m2 * lc2 * g * np.cos(th1 + th2 - np.pi / 2.0)
        phi1 = (-m2 * l1 * lc2 * w2 ** 2 * np.sin(th2) - 2 * m2 * l1 * lc2 * w2 * w1 * np.sin(th2)
                + (m1 * lc1 + m2 * l1) * g * np.cos(th1 - np.pi / 2) + phi2)
        dw2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * w1 ** 2 * np.sin(th2) - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        dw1 = -(d2 * dw2 + phi1) / d1
        return np.stack([w1, w2, dw1, dw2], 1)
    @staticmethod
    def features(S):
        return np.stack([np.sin(S[:, 0]), np.cos(S[:, 0]), np.sin(S[:, 1]), np.cos(S[:, 1]), S[:, 2], S[:, 3]], 1)
    @staticmethod
    def cost(S, U):
        h = -np.cos(S[..., 0]) - np.cos(S[..., 0] + S[..., 1])            # tip height in [-2, 2]
        return (2 - h) ** 2 + 0.001 * (S[..., 2] ** 2 + S[..., 3] ** 2) + 1e-3 * U[..., 0] ** 2
    @staticmethod
    def upright(S):            # tip height (max 2 = l1 + l2) above 1.8
        return (-np.cos(S[:, 0]) - np.cos(S[:, 0] + S[:, 1])) > 1.8
    @staticmethod
    def init(rng):
        return np.array([rng.normal(0, 0.1), rng.normal(0, 0.1), 0.0, 0.0])
    def random_state(self, rng):
        return np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi), rng.normal(0, 2), rng.normal(0, 3)])


TASKS = {"pendulum": Pendulum(), "cartpole": CartPole(), "acrobot": Acrobot()}


class TrueModel:
    def __init__(self, task): self.t = task
    def step(self, S, U): return rk4(self.t.f, S, U, self.t.dt)


class ContextModel:
    """Per-dimension regressors (predicting state change) fitted once, then batch-predicted."""
    def __init__(self, task, model_name, args, seed, Xc, Yc):
        self.t = task
        self.ms = [P.make_model(model_name, args, seed).fit(Xc, Yc[:, j]) for j in range(Yc.shape[1])]
    def step(self, S, U):
        X = np.hstack([self.t.features(S), U])
        return S + np.stack([m.predict(X) for m in self.ms], 1)


def offline_data(task, args, rng):
    X, Y = [], []
    for _ in range(args.n_train):
        s = task.random_state(rng)
        for _ in range(args.T_train):
            u = np.clip(rng.normal(0, task.u_max / 2, 1), -task.u_max, task.u_max)
            s2 = rk4(task.f, s[None], u[None], task.dt)[0]
            X.append(np.concatenate([task.features(s[None])[0], u])); Y.append(s2 - s); s = s2
    return np.array(X), np.array(Y)


def cem(task, model, s0, mean, args, rng):
    std = np.full(args.P, task.u_max / 2)
    for _ in range(args.n_iter):
        A = np.clip(mean + std * rng.normal(size=(args.n_cand, args.P)), -task.u_max, task.u_max)
        S = np.repeat(s0[None], args.n_cand, 0); J = np.zeros(args.n_cand)
        for h in range(args.P):
            U = A[:, h:h + 1]
            S = model.step(S, U)
            S[~np.isfinite(S)] = 1e6
            J += task.cost(S, U)
        el = A[np.argsort(J)[: args.n_elite]]
        mean, std = el.mean(0), el.std(0) + 1e-3
    return mean


def nearest_rows(X, Q, H):
    """The H rows of X closest (standardised, min over queries) to any query row in Q."""
    mu, sd = X.mean(0), X.std(0) + 1e-12
    d2 = ((((Q[:, None] - mu) / sd) - ((X[None] - mu) / sd)) ** 2).sum(-1).min(0)
    return np.argpartition(d2, H - 1)[:H]


def select(strategy, task, X, s, H, rng, full_cap, plan_Q=None):
    """plan_Q: query rows (features, action) along the previous plan's predicted path, used by
    'trajretrieval' (falls back to 'retrieval' when None, i.e. at the first control step)."""
    n = len(X); H = min(H, n)
    us = np.linspace(-task.u_max, task.u_max, 5)
    Q0 = np.hstack([np.repeat(task.features(s[None]), len(us), 0), us[:, None]])
    if strategy == "retrieval":
        return nearest_rows(X, Q0, H)
    if strategy == "trajretrieval":
        return nearest_rows(X, Q0 if plan_Q is None else np.vstack([Q0, plan_Q]), H)
    if strategy == "hybrid":                       # half nearest to the current state, half random
        near = nearest_rows(X, Q0, H // 2)
        rest = np.setdiff1d(np.arange(n), near)
        return np.concatenate([near, rng.choice(rest, H - len(near), replace=False)])
    if strategy == "recent":
        return np.arange(n - H, n)
    if strategy == "random":
        return rng.choice(n, H, replace=False)
    return np.arange(n) if n <= full_cap else rng.choice(n, full_cap, replace=False)


def episode(args, task_name, model_name, strategy, H, seed):
    task = TASKS[task_name]
    rng = np.random.default_rng(seed)
    X, Y = offline_data(task, args, rng)
    s = task.init(rng); mean = np.zeros(args.P)
    true = TrueModel(task); traj, acts, secs = [s.copy()], [], []
    for t in range(args.T_ep):
        tic = time.time()
        if model_name == "oracle":
            model = true
        else:
            plan_Q = None
            if strategy == "trajretrieval" and t > 0:
                # predicted path of the (shifted) previous plan under the previous model
                Sp, Qs = s[None], []
                for h in range(0, args.P, args.traj_stride):
                    u_h = mean[h:h + 1][None]
                    Qs.append(np.hstack([task.features(Sp), u_h]))
                    for k in range(h, min(h + args.traj_stride, args.P)):
                        Sp = model.step(Sp, mean[k:k + 1][None])
                    Sp[~np.isfinite(Sp)] = 0.0
                plan_Q = np.vstack(Qs)
            idx = select(strategy, task, X, s, H, rng, args.full_cap, plan_Q)
            model = ContextModel(task, model_name, args, seed, X[idx], Y[idx])
        mean = cem(task, model, s, mean, args, rng)
        u = np.array([mean[0]])
        s_next = true.step(s[None], u[None])[0]
        X = np.vstack([X, np.concatenate([task.features(s[None])[0], u])[None]])
        Y = np.vstack([Y, (s_next - s)[None]])
        s = s_next; traj.append(s.copy()); acts.append(u); secs.append(time.time() - tic)
        mean = np.append(mean[1:], 0.0)
    traj, acts = np.array(traj), np.array(acts)
    up = task.upright(traj[1:])
    return dict(task=task_name, model=model_name, strategy=strategy, H=H, seed=seed,
                cost=float(task.cost(traj[1:], acts).sum()), frac_upright=float(up.mean()),
                first_upright=int(np.argmax(up)) if up.any() else -1,
                secs_per_step=float(np.mean(secs)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="cartpole", choices=list(TASKS))
    p.add_argument("--models", nargs="+", default=["oracle", "tabpfn", "knn", "locallin"])
    p.add_argument("--strategies", nargs="+", default=["retrieval", "recent", "random"])
    p.add_argument("--H", type=int, default=500)
    p.add_argument("--seed_ids", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n_train", type=int, default=10)
    p.add_argument("--T_train", type=int, default=200)
    p.add_argument("--T_ep", type=int, default=150)
    p.add_argument("--P", type=int, default=25)
    p.add_argument("--n_cand", type=int, default=128)
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--n_elite", type=int, default=12)
    p.add_argument("--full_cap", type=int, default=5000)
    p.add_argument("--traj_stride", type=int, default=5)
    p.add_argument("--tabpfn_version", default="v3.5")
    p.add_argument("--tabpfn_fit_mode", default="fit_with_cache")
    p.add_argument("--out", default="mpc_results.json")
    args = p.parse_args()
    res = []
    for seed in args.seed_ids:
        for m in args.models:
            for st in (["-"] if m == "oracle" else args.strategies):
                r = episode(args, args.task, m, st, args.H, seed); res.append(r)
                print(f"[mpc {args.task} s{seed}] {m:8s} {st:9s} cost={r['cost']:8.1f} "
                      f"up={r['frac_upright']:.2f} first={r['first_upright']:4d} "
                      f"({r['secs_per_step']:.2f}s/step)", flush=True)
                Path(args.out).write_text(json.dumps(dict(args=vars(args), results=res), indent=1))


if __name__ == "__main__":
    main()
