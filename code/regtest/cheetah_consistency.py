"""Do Gymnasium HalfCheetah-v5 dynamics match D4RL halfcheetah-medium-v2 transitions?"""
import numpy as np, gymnasium as gym
_d = np.load("/scratch/naa9497/icwm/data/d4rl/halfcheetah_medium-v2.npz")
d = {k: _d[k] for k in ["observations", "actions", "next_observations"]}   # decompress each array ONCE
env = gym.make("HalfCheetah-v5").unwrapped; env.reset(seed=0)
print("obs dim", env.observation_space.shape, "act dim", env.action_space.shape, "dt", env.dt)
rng = np.random.default_rng(0); idx = rng.choice(len(d["observations"]) - 1, 300, replace=False)
err, scale = [], d["observations"].std(0)
for i in idx:
    o, a, o2 = d["observations"][i].astype(float), d["actions"][i].astype(float), d["next_observations"][i].astype(float)
    qpos = np.concatenate([[0.0], o[:8]]); qvel = o[8:]
    env.set_state(qpos, qvel); obs2, *_ = env.step(a)
    err.append(np.abs(obs2 - o2) / (scale + 1e-8))
err = np.array(err)
print("normalised |sim - D4RL next obs|: median %.2e, 95%% %.2e, max %.2e" % (np.median(err), np.quantile(err, 0.95), err.max()))
print("per-dimension median:", np.round(np.median(err, 0), 4))
