"""Re-render a HalfCheetah episode from the actions saved by cheetah_mpc.py (no planning needed).
The simulator is deterministic: reset(seed) + the same actions reproduce the trajectory; the replayed
final x position is checked against the recorded one.
Usage: python replay_video.py <results.json> <out_dir>"""
import json
import sys
from pathlib import Path

import numpy as np

import cheetah_mpc as C

res, out = json.load(open(sys.argv[1])), Path(sys.argv[2])
C.configure(res["args"].get("env", "halfcheetah"))
out.mkdir(parents=True, exist_ok=True)
for r in res["results"]:
    env = C.make_env(True); env.reset(seed=r["reset_seed"]); frames = []
    for a in r["actions"]:
        env.step(np.array(a)); frames.append(env.render())
    x = float(env.unwrapped.data.qpos[0])
    assert abs(x - r["distance"]) < 1e-6 * max(1.0, abs(x)), (x, r["distance"])
    vp = out / f"{C.ENV}_{r['controller']}_{r['strategy']}_s{r['seed']}.mp4"
    C.save_video(frames, str(vp), env.unwrapped.dt)
    print(f"{vp}: x={x:.2f} m (recorded {r['distance']:.2f}) - replay matches", flush=True)
    env.close()
