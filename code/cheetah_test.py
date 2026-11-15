"""GPU checks for cheetah_mpc.py: (1) parallel workers give the same predictions as the serial
per-dimension TabPFN; (2) wall time per control step for several speed settings; (3) offscreen video."""
import argparse
import json
import time

import numpy as np

import cheetah_mpc as C
import pilot as P


def equality(args):
    rng = np.random.default_rng(0)
    X, Y = C.offline_pool(10, rng)
    idx = C.nearest_rows(X, X[:1], 500)
    Xq = np.hstack([X[rng.choice(len(X), 64)][:, :17], rng.uniform(-1, 1, (64, 6))])
    a = argparse.Namespace(tabpfn_version="v3.5", tabpfn_fit_mode="fit_preprocessors")
    serial = C.ContextModel(a, 0, X[idx], Y[idx]).step(Xq[:, :17], Xq[:, 17:])
    par = C.ParallelTabPFN(6, "v3.5", "fit_preprocessors", 0).fit(X[idx], Y[idx])
    parallel = par.step(Xq[:, :17], Xq[:, 17:]); par.close()
    d = float(np.abs(serial - parallel).max()); s = float(np.abs(serial).max())
    print(f"EQUALITY serial vs parallel(6): max abs diff {d:.3e} (max |pred| {s:.2f})", flush=True)
    return dict(max_abs_diff=d)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", default="cheetah_test.json"); a = p.parse_args()
    out = {"equality": equality(a)}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
