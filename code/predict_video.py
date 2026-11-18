"""Prediction-comparison video: the real D4RL motion next to open-loop rollouts of TabPFN with retrieved
context (ours) and with the recent window (Lee et al.).

Data   exactly as study.py: the seed's training episodes + held-out test episode (study.d4rl_seed,
       n_train 10, t0_min 100, horizon 100) and the seed's first rollout start.
Model  TabPFN v3.5, one regressor per state dimension, predicting the state change; actions are the
       recorded ones. retrieval: the H pool rows nearest to the current (predicted state, action),
       re-selected every step (study.RPool, standardised distance); recent: the H latest transitions
       of the test episode before the start (fixed).
Render each (true or predicted) observation is shown by setting the MuJoCo state: qpos = [x, obs[:nq-1]],
       qvel = obs[nq-1:]; the unobserved root x is integrated from the (true or predicted) forward
       velocity, x += vx * dt. Three panels per frame with labels and the running error.
Usage: python predict_video.py --env hopper --seed 0 --steps 250 --fps 25 --out videos_pred
"""
import argparse
import json
from pathlib import Path

import numpy as np

import pilot as P
import study as S

GYM = {"hopper": "Hopper-v5", "walker2d": "Walker2d-v5", "halfcheetah": "HalfCheetah-v5"}
VX = {"hopper": 5, "walker2d": 8, "halfcheetah": 8}


def rollout(Xp, Yp, hist_idx, s0, U, H, strategy, seed, steps):
    pool = S.RPool(Xp, Yp)
    d = Yp.shape[1]
    fixed = None
    if strategy == "recent":
        idx = hist_idx[-H:]
        fixed = [P.TabPFN("v3.5", seed).fit(Xp[idx], Yp[idx, j]) for j in range(d)]
    s, out = s0.copy(), [s0.copy()]
    for h in range(steps):
        q = np.concatenate([s, U[h]])[None]
        if fixed is None:
            idx = pool.nearest(H, q[0])
            delta = np.empty(d)
            for j in range(d):                        # fit, predict, free (low GPU memory)
                m = P.TabPFN("v3.5", seed).fit(Xp[idx], Yp[idx, j]); delta[j] = m.predict(q)[0]; del m
        else:
            delta = np.array([m.predict(q)[0] for m in fixed])
        s = s + delta
        if not np.all(np.isfinite(s)):
            s = out[-1].copy()                        # keep the last finite state for rendering
        out.append(s.copy())
    return np.array(out)


def render(env_name, traj, dt, labels, errs, path, fps):
    import gymnasium as gym
    import imageio
    from PIL import Image, ImageDraw
    envs = [gym.make(GYM[env_name], render_mode="rgb_array", camera_name="track", width=420, height=320).unwrapped
            for _ in traj]
    for e in envs:
        e.reset(seed=0)
    nq = envs[0].model.nq
    xs = [np.concatenate([[0.0], np.cumsum(tr[:-1, VX[env_name]] * dt)]) for tr in traj]
    frames = []
    for t in range(len(traj[0])):
        panels = []
        for k, (e, tr) in enumerate(zip(envs, traj)):
            o = np.clip(tr[t], -50, 50)
            e.set_state(np.concatenate([[xs[k][t]], o[:nq - 1]]), o[nq - 1:])
            img = Image.fromarray(e.render())
            dr = ImageDraw.Draw(img)
            dr.rectangle([0, 0, 420, 44], fill=(255, 255, 255))
            dr.text((8, 4), labels[k], fill=(20, 20, 20))
            dr.text((8, 24), f"step {t}" + ("" if k == 0 else f"   error {errs[k][t]:.2f}"), fill=(90, 90, 90))
            panels.append(np.asarray(img))
        frames.append(np.concatenate(panels, axis=1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    for k in np.linspace(0, len(frames) - 1, 5).astype(int):
        imageio.imwrite(str(Path(path).with_suffix(f".t{k:03d}.png")), frames[k])
    for e in envs:
        e.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="hopper", choices=list(GYM))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--H", type=int, default=100)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--out", default="videos_pred")
    a = p.parse_args()
    args = argparse.Namespace(system=a.env, data_dir="/scratch/naa9497/icwm/data/d4rl", t0_min=100, horizon=100,
                              n_train=10, lags=1)
    rng = np.random.default_rng(1000 * a.seed + 7)                      # same as study.run
    Xtr, Ytr, S_te, _, U_te, sd = S.d4rl_seed(args, rng)
    X_te, Y_te = S.lagged(S_te, U_te, 1)
    t0 = int(args.t0_min)                                               # the seed's first rollout start
    steps = min(a.steps, len(U_te) - t0 - 1)
    Xp = np.vstack([Xtr, X_te[:t0]]); Yp = np.vstack([Ytr, Y_te[:t0]])
    hist_idx = np.arange(len(Xtr), len(Xp))
    true = S_te[t0:t0 + steps + 1]
    U = U_te[t0:t0 + steps]
    ret = rollout(Xp, Yp, hist_idx, true[0], U, a.H, "retrieval", a.seed, steps)
    rec = rollout(Xp, Yp, hist_idx, true[0], U, a.H, "recent", a.seed, steps)
    err = lambda tr: np.sqrt(np.mean(((tr - true) / sd) ** 2, axis=1))
    e_ret, e_rec = err(ret), err(rec)
    import gymnasium as gym
    dt = gym.make(GYM[a.env]).unwrapped.dt
    name = f"{a.out}/{a.env}_s{a.seed}_pred.mp4"
    slow = 1 / (dt * a.fps)
    render(a.env, [true, ret, rec], dt,
           ["Real motion (D4RL)",                       # playback speed is stated in the README, not on the frame
            "TabPFN + retrieval (ours)", "TabPFN, recent window"],
           [None, e_ret, e_rec], name, a.fps)
    summary = dict(env=a.env, seed=a.seed, t0=t0, steps=steps, dt=dt, fps=a.fps,
                   slow_motion=round(1 / (dt * a.fps), 2),
                   err_retrieval={str(h): float(e_ret[h]) for h in (10, 50, 100, steps) if h <= steps},
                   err_recent={str(h): float(e_rec[h]) for h in (10, 50, 100, steps) if h <= steps})
    Path(f"{a.out}/{a.env}_s{a.seed}_pred.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
