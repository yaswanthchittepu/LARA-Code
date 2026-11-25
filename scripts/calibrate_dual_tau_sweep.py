"""Sweep over tau values, running bootstrap bisection for each.

For each tau, lambda* is set to the 97.5th percentile of bootstrap lambda values
(conservative estimate that accounts for finite-sample variance).

Output: JSON with per-tau results including lambda*, full bootstrap stats.

Usage:
    python scripts/calibrate_dual_tau_sweep.py
    python scripts/calibrate_dual_tau_sweep.py --tau_list -1 -2 -3 -4 --n_bootstrap 500
"""

import os
import sys
import json
import time
import argparse
import yaml
import numpy as np
from tqdm import tqdm

# import helpers from the single-tau script (functions are module-level)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from calibrate_dual_bisect import build_rc_arrays, tilted_expected_cost, solve_lambda

from datasets import Dataset


def bootstrap_tau(R, C, beta, tau, lam_max, tol, n_bootstrap, rng):
    """Run bootstrap bisection for one tau. Returns dict of stats."""
    n_prompts = R.shape[0]
    boot_lambdas = []
    n_skipped = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        try:
            lam_t = solve_lambda(R[idx], C[idx], beta, tau, lam_max, tol)
            boot_lambdas.append(lam_t)
        except ValueError:
            n_skipped += 1

    if not boot_lambdas:
        raise ValueError(f"All {n_bootstrap} bootstrap trials infeasible for tau={tau}")

    arr = np.array(boot_lambdas)
    p25, p75 = np.percentile(arr, [25, 75])

    return {
        "n_trials":    n_bootstrap,
        "n_skipped":   n_skipped,
        "n_used":      len(arr),
        "mean":        float(arr.mean()),
        "std":         float(arr.std()),
        "min":         float(arr.min()),
        "max":         float(arr.max()),
        "p25":         float(p25),
        "p75":         float(p75),
        "iqr":         float(p75 - p25),
        "ci_95_lo":    float(np.percentile(arr, 2.5)),
        "ci_95_hi":    float(np.percentile(arr, 97.5)),
        "p95":         float(np.percentile(arr, 95.0)),
        "p97_5":       float(np.percentile(arr, 97.5)),
        "p99":         float(np.percentile(arr, 99.0)),
        "all_lambdas": arr.tolist(),
    }


if __name__ == "__main__":
    with open("./args/calibrate_dual_sweep.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",    type=str,   default=None)
    parser.add_argument("--out_dir",     type=str,   default=None)
    parser.add_argument("--tau_list",    type=float, nargs="+", default=None)
    parser.add_argument("--beta",        type=float, default=None)
    parser.add_argument("--lam_max",     type=float, default=None)
    parser.add_argument("--tol",         type=float, default=None)
    parser.add_argument("--n_bootstrap", type=int,   default=None)
    parser.add_argument("--seed",        type=int,   default=None)
    cli = parser.parse_args()

    for key in vars(cli):
        val = getattr(cli, key)
        if val is not None:
            cfg[key] = val

    eval_dir    = cfg["eval_dir"]
    out_dir     = cfg["out_dir"]
    tau_list    = cfg["tau_list"]
    beta        = float(cfg["beta"])
    lam_max     = float(cfg.get("lam_max", 1e4))
    tol         = float(cfg.get("tol", 1e-6))
    n_bootstrap = int(cfg.get("n_bootstrap", 200))
    seed        = int(cfg.get("seed", 0))

    os.makedirs(out_dir, exist_ok=True)

    dataset = Dataset.load_from_disk(eval_dir)
    print(f"Loaded {len(dataset)} rows from {eval_dir}", flush=True)

    R, C = build_rc_arrays(dataset)
    n_prompts, n_completions = R.shape
    print(f"Shaped into ({n_prompts} prompts, {n_completions} completions)")

    ec_unconstr = C.mean()
    per_prompt_min = C.min(axis=1).mean()
    print(f"\nE[C]={ec_unconstr:.4f}  E_x[min_y c]={per_prompt_min:.4f}")
    print(f"Feasible tau range: ({per_prompt_min:.4f}, {ec_unconstr:.4f})")
    print(f"\nTau sweep: {tau_list}")
    print(f"Bootstrap trials: {n_bootstrap}  seed: {seed}\n")

    rng = np.random.default_rng(seed)
    results = []

    for tau in tqdm(tau_list, desc="tau sweep"):
        t0 = time.perf_counter()
        print(f"--- tau={tau} ---", flush=True)

        # exact full-data solve
        try:
            lam_exact = solve_lambda(R, C, beta, tau, lam_max, tol)
            ec_check  = tilted_expected_cost(lam_exact, R, C, beta)
        except ValueError as e:
            print(f"  Skipping (infeasible): {e}")
            results.append({"tau": tau, "feasible": False})
            continue

        print(f"  Exact lambda* = {lam_exact:.6f}  (tilted cost = {ec_check:.6f})")

        # bootstrap
        try:
            boot = bootstrap_tau(R, C, beta, tau, lam_max, tol, n_bootstrap, rng)
        except ValueError as e:
            print(f"  All bootstrap trials infeasible: {e}")
            results.append({"tau": tau, "feasible": False, "reason": str(e),
                            "lambda_exact": lam_exact, "tilted_cost_at_lam_exact": ec_check})
            continue
        lam_star = boot["p97_5"]   # conservative: 97.5th percentile
        ec_at_lam_star = tilted_expected_cost(lam_star, R, C, beta)

        elapsed = time.perf_counter() - t0
        print(f"  Bootstrap p97.5 = {lam_star:.6f}  mean={boot['mean']:.6f}  "
              f"std={boot['std']:.6f}  skipped={boot['n_skipped']}  ({elapsed:.1f}s)")
        print(f"  E[c] at lam*={ec_at_lam_star:.6f}  diff from tau={ec_at_lam_star - tau:.6f}")

        results.append({
            "tau":                      tau,
            "feasible":                 True,
            "lambda_exact":             lam_exact,
            "tilted_cost_at_lam_exact":  ec_check,
            "cost_slack_at_lam_exact":   ec_check - tau,
            "lambda_star":              lam_star,         # 97.5th percentile — use this for deployment
            "tilted_cost_at_lam_star":  ec_at_lam_star,
            "cost_slack_at_lam_star":   ec_at_lam_star - tau,  # negative = more conservative than tau
            "beta":                     beta,
            "n_prompts":                n_prompts,
            "n_completions":            n_completions,
            "bootstrap":                boot,
        })

    # summary table
    print("\n=== Summary ===")
    print(f"{'tau':>8}  {'lam_exact':>12}  {'E[c]@lam_exact':>16}  {'slack@lam_exact':>16}  {'lam* (p97.5)':>14}  {'E[c]@lam*':>12}  {'slack@lam*':>12}  {'std':>8}  {'IQR':>8}")
    for r in results:
        if not r["feasible"]:
            print(f"{r['tau']:>8.2f}  {'INFEASIBLE':>88}")
        else:
            b = r["bootstrap"]
            print(f"{r['tau']:>8.2f}  {r['lambda_exact']:>12.6f}  {r['tilted_cost_at_lam_exact']:>16.6f}  "
                  f"{r['cost_slack_at_lam_exact']:>16.6f}  "
                  f"{r['lambda_star']:>14.6f}  {r['tilted_cost_at_lam_star']:>12.6f}  "
                  f"{r['cost_slack_at_lam_star']:>12.6f}  {b['std']:>8.6f}  {b['iqr']:>8.6f}")

    out = {
        "eval_dir":     eval_dir,
        "beta":         beta,
        "n_bootstrap":  n_bootstrap,
        "seed":         seed,
        "results":      results,
    }
    out_path = os.path.join(out_dir, "lambda_star_sweep.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")
