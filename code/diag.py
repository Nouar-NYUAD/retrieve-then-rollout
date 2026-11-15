"""Diagnostics: WHY TabPFN (+ retrieval) fails where it fails, and where the GP is limited.

D1 precision floor : noise-free regression of smooth functions (linear, sine, Lorenz-type quadratic)
                     with n = 50..2000 context rows. If TabPFN's error stops falling while the GP's
                     keeps falling, TabPFN has a precision floor (noisy pretraining prior / binned
                     output), which would explain the clean-data gap to the GP.
D2 retrieval width : Lorenz-96 (d = 10, 20) one-step error of TabPFN and GP with retrieval vs random
                     context for H = 50..2000, plus the spread of the retrieved context relative to
                     the pool (does retrieval become too local in higher dimensions?).
D3 abrupt dynamics : Hopper / Walker2d one-step errors binned by how abrupt the true transition is
                     (|true change| quantiles; contacts give the largest changes): GP vs TabPFN, both
                     with the same retrieved contexts.
D4 GP fragility    : fraction of GP kernel hyperparameters at their bounds after fitting, per system,
                     and kernel-fit time.
All errors are RMSE normalised by the target std (D1) or the state std (D2, D3).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import pilot as P
import study as St


def gp_fit_predict(Xc, yc, Xq, rng, restarts=5):
    k = St.fit_gp_kernels(Xc, yc[:, None], min(len(Xc), 1000), rng, restarts)[0]   # kernel fit on <=1000 rows (cost)
    return St.FixedKernelGP(*k).fit(Xc, yc).predict(Xq), k[0]


def tabpfn_fit_predict(Xc, yc, Xq, seed):
    return P.TabPFN("v3.5", seed).fit(Xc, yc).predict(Xq)


def d1(args, out):
    rng = np.random.default_rng(0)
    funcs = {"linear": lambda X: X @ np.array([1.0, -2.0, 0.5]),
             "sine": lambda X: np.sin(2 * X[:, 0]) * np.cos(X[:, 1]) + 0.5 * X[:, 2],
             "lorenz_quadratic": lambda X: X[:, 0] * (28 - X[:, 2]) - X[:, 1]}
    for name, f in funcs.items():
        for n in args.d1_n:
            Xc = rng.uniform(-2, 2, (n, 3)); Xq = rng.uniform(-2, 2, (300, 3))
            yc, yq = f(Xc), f(Xq); sd = yq.std()
            t = time.time(); pt = tabpfn_fit_predict(Xc, yc, Xq, 0); tt = time.time() - t
            t = time.time(); pg, _ = gp_fit_predict(Xc, yc, Xq, np.random.default_rng(1)); tg = time.time() - t
            r = dict(diag="D1", func=name, n=n, tabpfn=float(np.sqrt(np.mean((pt - yq) ** 2)) / sd),
                     gp=float(np.sqrt(np.mean((pg - yq) ** 2)) / sd), t_tabpfn=tt, t_gp=tg)
            out.append(r); print(r, flush=True)


def d2(args, out):
    for d in args.d2_dims:
        a = argparse.Namespace(system="lorenz96", l96_dim=d, obs_noise=0.0, n_train=4, T_train=1000,
                               T_test=1500, lags=1)
        Xtr, Ytr, S, So, U, sd = St.synthetic_seed(a, np.random.default_rng(7))
        Xte, Yte = St.lagged(So, U, 1)
        pool = P.Pool(Xtr, Ytr); rng = np.random.default_rng(3)
        qi = rng.choice(np.arange(100, len(Xte)), args.d2_queries, replace=False)
        dims = list(range(min(d, args.d2_out_dims)))            # a subset of output dims (cost)
        for H in args.d2_H:
            for strat in ["retrieval", "random"]:
                et, eg, spread = [], [], []
                for q in qi:
                    idx = pool.nearest(H, Xte[q]) if strat == "retrieval" else rng.choice(len(Xtr), H, replace=False)
                    Xc, Yc = Xtr[idx], Ytr[idx]
                    spread.append(float(np.mean(Xc.std(0) / (Xtr.std(0) + 1e-12))))
                    for j in dims:
                        pt = tabpfn_fit_predict(Xc, Yc[:, j], Xte[q][None], 0)[0]
                        k = St.fit_gp_kernels(Xc, Yc[:, j:j + 1], min(len(Xc), 500), np.random.default_rng(1), 2)[0]
                        pg = St.FixedKernelGP(*k).fit(Xc, Yc[:, j]).predict(Xte[q][None])[0]
                        et.append((pt - Yte[q, j]) / sd[j]); eg.append((pg - Yte[q, j]) / sd[j])
                r = dict(diag="D2", dim=d, H=H, strategy=strat, tabpfn=float(np.sqrt(np.mean(np.square(et)))),
                         gp=float(np.sqrt(np.mean(np.square(eg)))), context_spread=float(np.mean(spread)))
                out.append(r); print(r, flush=True)


def d3(args, out):
    for system in args.d3_systems:
        a = argparse.Namespace(system=system, data_dir=args.data_dir, t0_min=100, horizon=100, n_train=10, lags=1)
        Xtr, Ytr, S, So, U, sd = St.d4rl_seed(a, np.random.default_rng(7))
        Xte, Yte = St.lagged(So, U, 1)
        pool = P.Pool(Xtr, Ytr); rng = np.random.default_rng(3)
        qi = rng.choice(len(Xte), min(args.d3_queries, len(Xte)), replace=False)
        abrupt = np.linalg.norm(Yte[qi] / sd, axis=1)            # size of the true (normalised) change
        kern = {}
        et, eg = np.zeros((len(qi), Ytr.shape[1])), np.zeros((len(qi), Ytr.shape[1]))
        for i, q in enumerate(qi):
            idx = pool.nearest(args.d3_H, Xte[q]); Xc, Yc = Xtr[idx], Ytr[idx]
            for j in range(Ytr.shape[1]):
                if j not in kern:                                    # one GP kernel per dim, fitted on the pool
                    kern[j] = St.fit_gp_kernels(Xtr, Ytr[:, j:j + 1], 500, np.random.default_rng(1), 5)[0]
                et[i, j] = (tabpfn_fit_predict(Xc, Yc[:, j], Xte[q][None], 0)[0] - Yte[q, j]) / sd[j]
                eg[i, j] = (St.FixedKernelGP(*kern[j]).fit(Xc, Yc[:, j]).predict(Xte[q][None])[0] - Yte[q, j]) / sd[j]
        edges = np.quantile(abrupt, [0, 0.5, 0.9, 1.0])
        for lo, hi, lab in [(edges[0], edges[1], "smooth (lower 50%)"), (edges[1], edges[2], "50-90%"),
                            (edges[2], edges[3] + 1e-9, "most abrupt 10%")]:
            m = (abrupt >= lo) & (abrupt < hi)
            r = dict(diag="D3", system=system, bin=lab, n=int(m.sum()),
                     tabpfn=float(np.sqrt(np.mean(et[m] ** 2))), gp=float(np.sqrt(np.mean(eg[m] ** 2))))
            out.append(r); print(r, flush=True)
        # D4 on the same kernels
        ls_hit, nl_hit = [], []
        for j, (k, *_) in kern.items():
            ls = np.atleast_1d(k.k1.k2.length_scale); nl = k.k2.noise_level
            ls_hit.append(float(np.mean(np.isclose(ls, 1e-2, rtol=1e-3) | np.isclose(ls, 1e3, rtol=1e-3))))
            nl_hit.append(float(np.isclose(nl, 1e-8, rtol=1e-2) or np.isclose(nl, 10, rtol=1e-2)))
        out.append(dict(diag="D4", system=system, n_dims=len(kern),
                        frac_length_scales_at_bound=float(np.mean(ls_hit)),
                        frac_dims_noise_at_bound=float(np.mean(nl_hit))))
        print(out[-1], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="+", default=["d1", "d2", "d3"])
    p.add_argument("--d1_n", nargs="+", type=int, default=[50, 100, 500, 2000])
    p.add_argument("--d2_dims", nargs="+", type=int, default=[10, 20])
    p.add_argument("--d2_H", nargs="+", type=int, default=[50, 100, 500, 2000])
    p.add_argument("--d2_queries", type=int, default=40)
    p.add_argument("--d2_out_dims", type=int, default=3)
    p.add_argument("--d3_systems", nargs="+", default=["hopper", "walker2d"])
    p.add_argument("--d3_queries", type=int, default=200)
    p.add_argument("--d3_H", type=int, default=100)
    p.add_argument("--data_dir", default="/scratch/naa9497/icwm/data/d4rl")
    p.add_argument("--out", default="diag_results.json")
    args = p.parse_args()
    out = []
    for part in args.parts:
        {"d1": d1, "d2": d2, "d3": d3}[part](args, out)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
