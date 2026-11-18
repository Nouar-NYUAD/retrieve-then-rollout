"""Task file for the sliding-window baseline (Lee et al. 2611.11921): for every main setting that has a
TabPFN recent-window result (Fig. 1a, Tables 2-3), re-run the SAME arguments (system, seed, pool,
starts, horizon, noise, lags, ...) with --strategies sliding, H = 100, models tabpfn (+ dynapfn
where it was run). Writes tasks_sliding.txt (one study.py command line per original result file).
Usage: python make_tasks_sliding.py   (run from pilot/)"""
import glob
import json
from pathlib import Path

import analyze_full as A

KEEP = {"lorenz+nt4", "pendulum+nt4", "lorenz+noise0.05+nt4", "pendulum+noise0.05+nt4",
        "hopper+nt10", "halfcheetah+nt10", "walker2d+nt10",
        "lorenz96(d=10)+nt4", "lorenz96(d=20)+nt4", "lorenz96(d=40)+nt4", "camels+lags7+feat=ext+nt4"}
SKIP_ARGS = {"models", "strategies", "H", "out", "seed_ids"}   # set below
CLI = {"seed": "seed_ids"}

lines, seen = [], set()
# exactly the result files the paper tables use (paper_tables_full.full_rows): own-format folders excluded
FILES = [f for f in sorted(glob.glob("results/sync/*/*.json"))
         if "diag_" not in Path(f).name and not Path(f).name.startswith("plan")
         and Path(f).parent.name not in ("cheetah", "demo", "scaling")]
for f in FILES:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict) or "args" not in d or not d.get("results"):
        continue
    a, R = d["args"], d["results"]
    r0 = dict(R[0]); r0["_n_train"] = a.get("n_train")
    st = A.setting(r0)
    if st not in KEEP:
        continue
    have = {(r["strategy"], r["model"]) for r in R}
    models = [m for m in ("tabpfn", "dynapfn") if ("recent", m) in have]
    if not models or a.get("tabpfn_version", "v3.5") != "v3.5" or a.get("tabpfn_n_estimators", 0):
        continue
    seeds = sorted({r["seed"] for r in R})
    key = (st, tuple(seeds), tuple(models))
    if key in seen:
        continue
    seen.add(key)
    parts = []
    for k, v in a.items():
        if k in SKIP_ARGS or v is None or k == "cur_seed":
            continue
        if isinstance(v, bool):
            if v: parts.append(f"--{k}")
            continue
        vals = v if isinstance(v, list) else [v]
        parts.append(f"--{k} " + " ".join(str(x) for x in vals))
    tag = "sliding_" + Path(f).stem
    parts += ["--seed_ids " + " ".join(map(str, seeds)), "--models " + " ".join(models),
              "--strategies sliding", "--H 100", f"--out {tag}.json"]
    lines.append(" ".join(parts))
    print(st, seeds, models, Path(f).parent.name + "/" + Path(f).name)
open("tasks_sliding.txt", "w", newline="\n").write("\n".join(lines) + "\n")
print(len(lines), "tasks")
