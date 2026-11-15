"""Summary of scaling.py results: per pool size and context, median time per step (and its split), peak
GPU memory and the geometric-mean rollout error over seeds.
Usage: python scaling_summary.py <dir with scale_*.json>"""
import collections
import glob
import json
import sys

import numpy as np


def summary(folder):
    acc = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{folder}/scale_hopper_nt*.json")):
        for r in json.load(open(f))["results"]:
            acc[(r["n_train"], r["strategy"])].append(r)
    out = {}
    for (n, st), rs in sorted(acc.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        rec = dict(n_seeds=len(ok), status=[r["status"] for r in rs])
        if ok:
            med = lambda k: float(np.median([r[k] for r in ok]))
            gm = lambda h: float(np.exp(np.mean(np.log([r["rollout_nrmse"][h] for r in ok]))))
            rec.update(pool_rows=med("pool_rows"), secs_per_step=med("secs_per_step"), secs_fit=med("secs_fit"),
                       secs_predict=med("secs_predict"), secs_knn=med("secs_knn"), gpu_gb=med("gpu_peak_gb"),
                       err10=gm("10"), err50=gm("50"))
        out[(n, st)] = rec
    return out


if __name__ == "__main__":
    S = summary(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(f"{'n_train':>7} {'context':9} {'rows':>6} {'s/step':>7} {'fit':>6} {'pred':>6} {'knn':>5} {'GPU GB':>6} {'err@10':>7} {'err@50':>7} seeds")
    for (n, st), r in S.items():
        if "pool_rows" not in r:
            print(f"{n:7d} {st:9} {r['status']}"); continue
        print(f"{n:7d} {st:9} {r['pool_rows']:6.0f} {r['secs_per_step']:7.2f} {r['secs_fit']:6.0f} {r['secs_predict']:6.0f} "
              f"{r['secs_knn']:5.2f} {r['gpu_gb']:6.2f} {r['err10']:7.3f} {r['err50']:7.3f} {r['n_seeds']}")
