"""Phase 2 (exact) — Calibrate lambda via bisection on the dual gradient.

Optimality condition:  E_x[ E_{tilde_pi}[c(x,y)] ] = tau
where tilde_pi(y|x) ∝ exp( (r(x,y) - lambda*c(x,y)) / beta )

Bisects on lambda until the tilted expected cost equals tau.

Bootstrap mode (--bootstrap --n_bootstrap K):
  Runs K trials, each sampling n_prompts with replacement.
  Reports lambda* per trial, then mean +/- std.

Usage:
    python scripts/calibrate_dual_bisect.py [--beta 0.1] [--tau -1.0] ...
    python scripts/calibrate_dual_bisect.py --bootstrap --n_bootstrap 200 ...

All defaults are read from args/calibrate_dual.yaml; CLI args override yaml values.
"""

import os
import json
import time
import argparse
import yaml
import numpy as np
from scipy.optimize import brentq
from datasets import Dataset


def build_rc_arrays(dataset):
    prompt_ids = dataset["prompt_id"]
    n_prompts  = max(prompt_ids) + 1
    r_groups   = [[] for _ in range(n_prompts)]
    c_groups   = [[] for _ in range(n_prompts)]
    for pid, r, c in zip(prompt_ids, dataset["reward"], dataset["cost"]):
        r_groups[pid].append(r)
        c_groups[pid].append(c)
    R = np.array(r_groups, dtype=np.float64)   # (n_prompts, n_completions)
    C = np.array(c_groups, dtype=np.float64)
    return R, C


def tilted_expected_cost(lam, R, C, beta):
    """E_x[ E_{tilde_pi}[c] ] — softmax-weighted cost under the tilted distribution."""
    V  = (R - lam * C) / beta
    W  = np.exp(V - V.max(axis=1, keepdims=True))
    W /= W.sum(axis=1, keepdims=True)
    return (W * C).sum(axis=1).mean()


def solve_lambda(R, C, beta, tau, lam_max, tol):
    """Return lambda* for a given R, C array (may be a bootstrap subsample)."""
    f     = lambda lam: tilted_expected_cost(lam, R, C, beta) - tau
    ec_lo = tilted_expected_cost(0.0,     R, C, beta)
    ec_hi = tilted_expected_cost(lam_max, R, C, beta)

    if tau > ec_lo:
        return 0.0
    if tau < ec_hi:
        raise ValueError(
            f"tau={tau} is infeasible: min achievable tilted cost at lam_max={lam_max} "
            f"is {ec_hi:.4f}. Increase lam_max or choose tau > {ec_hi:.4f}."
        )
    return brentq(f, 0.0, lam_max, xtol=tol, full_output=False)


if __name__ == "__main__":

    with open("./args/calibrate_dual.yaml", "r") as f:
        args = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",    type=str,   default=None)
    parser.add_argument("--out_dir",     type=str,   default=None)
    parser.add_argument("--beta",        type=float, default=None)
    parser.add_argument("--tau",         type=float, default=None)
    parser.add_argument("--lam_max",     type=float, default=None)
    parser.add_argument("--tol",         type=float, default=None)
    parser.add_argument("--bootstrap",   action="store_true", default=False, help="Enable bootstrap mode")
    parser.add_argument("--n_bootstrap", type=int,   default=None, help="Number of bootstrap trials")
    parser.add_argument("--seed",        type=int,   default=None, help="Random seed for bootstrap")
    parse_args = parser.parse_args()

    for a in vars(parse_args):
        val = getattr(parse_args, a)
        if val is not None and val is not False:
            print(f"Overriding {a} from {args.get(a)} to {val}...\n")
            args[a] = val
    if parse_args.bootstrap:
        args["bootstrap"] = True

    os.makedirs(args["out_dir"], exist_ok=True)

    dataset = Dataset.load_from_disk(args["eval_dir"])
    print(f"Loaded {len(dataset)} rows from {args['eval_dir']}", flush=True)

    R, C = build_rc_arrays(dataset)
    n_prompts, n_completions = R.shape
    print(f"Shaped into ({n_prompts} prompts, {n_completions} completions)", flush=True)

    per_prompt_min = C.min(axis=1).mean()
    print(f"\nE[C]={C.mean():.4f}  E_x[min_y c]={per_prompt_min:.4f}")
    print(f"Feasible tau range: ({per_prompt_min:.4f}, {C.mean():.4f})  |  current tau={args['tau']}\n")

    beta    = float(args["beta"])
    tau     = float(args["tau"])
    lam_max = float(args.get("lam_max") or 1e4)
    tol     = float(args.get("tol") or 1e-6)

    # ---- full-data solve ----
    t0    = time.perf_counter()
    ec_lo = tilted_expected_cost(0.0,     R, C, beta)
    ec_hi = tilted_expected_cost(lam_max, R, C, beta)
    print(f"Tilted cost at lam=0:         {ec_lo:.4f}  (= E[C])")
    print(f"Tilted cost at lam={lam_max}: {ec_hi:.4f}  (lower bound)")

    lam_star   = solve_lambda(R, C, beta, tau, lam_max, tol)
    ec         = tilted_expected_cost(lam_star, R, C, beta)
    solve_time = time.perf_counter() - t0
    print(f"\nlambda* = {lam_star:.8f}")
    print(f"E_tilde[c] at lambda* = {ec:.8f}  (target tau={tau})")
    print(f"Bisection runtime: {solve_time:.4f}s")

    out = {
        "lambda_star":   lam_star,
        "tilted_cost":   ec,
        "beta":          beta,
        "tau":           tau,
        "n_prompts":     n_prompts,
        "n_completions": n_completions,
    }

    # ---- bootstrap ----
    if args.get("bootstrap"):
        n_bootstrap = int(args.get("n_bootstrap") or 200)
        seed        = int(args.get("seed") or 0)
        rng         = np.random.default_rng(seed)

        print(f"\nRunning {n_bootstrap} bootstrap trials (seed={seed})...", flush=True)
        t_boot = time.perf_counter()
        boot_lambdas = []
        n_skipped = 0
        for t in range(n_bootstrap):
            idx = rng.integers(0, n_prompts, size=n_prompts)   # sample with replacement
            try:
                lam_t = solve_lambda(R[idx], C[idx], beta, tau, lam_max, tol)
                boot_lambdas.append(lam_t)
            except ValueError:
                n_skipped += 1
                continue
            if (t + 1) % 50 == 0:
                print(f"  trial {t+1}/{n_bootstrap}  lam={lam_t:.6f}  skipped={n_skipped}", flush=True)

        if n_skipped > 0:
            print(f"\nWarning: {n_skipped}/{n_bootstrap} trials skipped (infeasible subsample)")

        boot_lambdas = np.array(boot_lambdas)
        boot_mean    = boot_lambdas.mean()
        boot_std     = boot_lambdas.std()

        boot_time = time.perf_counter() - t_boot
        print(f"\nBootstrap results ({n_bootstrap} trials, runtime: {boot_time:.2f}s):")
        print(f"  mean lambda* = {boot_mean:.6f}")
        print(f"  std          = {boot_std:.6f}")
        print(f"  95% CI       = [{np.percentile(boot_lambdas, 2.5):.6f}, {np.percentile(boot_lambdas, 97.5):.6f}]")
        print(f"  95th percentile: {np.percentile(boot_lambdas, 95.0):.6f}")
        print(f"  99th percentile: {np.percentile(boot_lambdas, 99.0):.6f}")
        print(f"  max lambda*  = {boot_lambdas.max():.6f}")

        out["bootstrap"] = {
            "n_trials":    n_bootstrap,
            "n_skipped":   n_skipped,
            "n_used":      len(boot_lambdas),
            "mean":        boot_mean,
            "std":         boot_std,
            "ci_95_lo":    float(np.percentile(boot_lambdas, 2.5)),
            "ci_95_hi":    float(np.percentile(boot_lambdas, 97.5)),
            "all_lambdas": boot_lambdas.tolist(),
        }

    out_path = os.path.join(args["out_dir"], "lambda_star.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)
