"""Best-of-N selection using the Lagrangian reward r - lambda * c.

Loads a scored HF dataset (prompt_id, prompt, completion, reward, cost),
then for each lambda in LAMBDAS selects one completion per prompt via:

  --mode argmax   : pick argmax(r - lambda*c)  [BoN]
  --mode tilted   : sample from softmax((r - lambda*c) / beta)
                    This is a Monte Carlo draw from pi*(y|x), so lambda*
                    from dual calibration formally guarantees E[c] = tau.

Also prints a summary table of E[r], E[c], E[r_net] across lambdas.

Usage:
    python scripts/bon_select.py [--mode argmax|tilted] [--beta 0.1] ...
"""

import os
import json
import yaml
import argparse
import numpy as np
from datasets import Dataset


def build_arrays(dataset):
    """Group flat dataset into (n_prompts, n_completions) arrays."""
    prompt_ids  = dataset["prompt_id"]
    n_prompts   = max(prompt_ids) + 1

    prompts     = [None] * n_prompts
    completions = [[] for _ in range(n_prompts)]
    rewards     = [[] for _ in range(n_prompts)]
    costs       = [[] for _ in range(n_prompts)]

    for pid, prompt, comp, r, c in zip(
        prompt_ids, dataset["prompt"], dataset["completion"],
        dataset["reward"], dataset["cost"]
    ):
        prompts[pid] = prompt
        completions[pid].append(comp)
        rewards[pid].append(r)
        costs[pid].append(c)

    R = np.array(rewards, dtype=np.float64)   # (n_prompts, n_completions)
    C = np.array(costs,   dtype=np.float64)
    return prompts, completions, R, C


def select(R_net, mode, beta, tilted_mode, rng):
    """Return selected indices (n_prompts,) given net rewards."""
    if mode == "argmax":
        return R_net.argmax(axis=1)
    else:  # tilted
        V   = R_net / beta
        V  -= V.max(axis=1, keepdims=True)          # numerical stability
        W   = np.exp(V)
        W  /= W.sum(axis=1, keepdims=True)          # softmax weights ~ pi*
        if tilted_mode == "greedy":
            return W.argmax(axis=1)                 # argmax of pi* weights
        else:  # stochastic
            cumW = W.cumsum(axis=1)
            u    = rng.random((R_net.shape[0], 1))
            return (u > cumW).sum(axis=1)           # sample from pi*


if __name__ == "__main__":

    with open("./args/bon_select.yaml", "r") as f:
        args = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",     type=str,   default=None, help="Path to scored HF dataset")
    parser.add_argument("--base_out_dir", type=str,   default=None, help="Base dir; one subdir per lambda")
    parser.add_argument("--lambdas",      type=float, nargs="+", default=None, help="Lambda values to sweep")
    parser.add_argument("--mode",         type=str,   default=None, choices=["argmax", "tilted"],
                        help="argmax=BoN, tilted=select from pi* weights (requires --beta)")
    parser.add_argument("--tilted_mode",  type=str,   default=None, choices=["greedy", "stochastic"],
                        help="(tilted only) greedy=argmax of pi* weights, stochastic=sample from pi*")
    parser.add_argument("--beta",         type=float, default=None, help="Temperature for tilted mode")
    parser.add_argument("--seed",         type=int,   default=None, help="Random seed for stochastic tilted")
    parser.add_argument("--calib_json",   type=str,   default=None,
                        help="lambda_star.json from calibrate_dual_bisect.py; its lambda* is added to the sweep")
    parse_args = parser.parse_args()

    for a in vars(parse_args):
        val = getattr(parse_args, a)
        if val is not None:
            print(f"Overriding {a}: {args.get(a)} -> {val}")
            args[a] = val

    dataset = Dataset.load_from_disk(args["eval_dir"])
    print(f"Loaded {len(dataset)} rows from {args['eval_dir']}", flush=True)

    prompts, completions, R, C = build_arrays(dataset)
    n_prompts, n_completions = R.shape
    print(f"Shaped into ({n_prompts} prompts, {n_completions} completions)\n")

    mode         = args.get("mode") or "argmax"
    tilted_mode  = args.get("tilted_mode") or "greedy"
    beta         = float(args.get("beta") or 1.0)
    seed         = int(args.get("seed") or 0)
    lambdas      = list(args.get("lambdas") or [])
    lam_star     = None
    if args.get("calib_json"):
        with open(args["calib_json"]) as f:
            lam_star = float(json.load(f)["lambda_star"])
        if lam_star not in lambdas:
            lambdas.append(lam_star)
        lambdas = sorted(lambdas)
        print(f"Added calibrated lambda*={lam_star} to the sweep\n")
    if not lambdas:
        parser.error("no lambdas given: pass --lambdas and/or --calib_json")
    rng          = np.random.default_rng(seed)

    print(f"mode={mode}" + (f"  tilted_mode={tilted_mode}" if mode == "tilted" else "") + f"  beta={beta}\n")
    print(f"{'lambda':>10}  {'E[r]':>10}  {'E[c]':>10}  {'E[r_net]':>10}")
    print("-" * 46)

    summary_points = []
    for lam in lambdas:
        R_net = R - lam * C                                        # (n_prompts, n_completions)
        idx   = select(R_net, mode, beta, tilted_mode, rng)         # (n_prompts,)

        chosen_completions = [completions[i][idx[i]] for i in range(n_prompts)]
        chosen_r    = R[np.arange(n_prompts), idx]
        chosen_c    = C[np.arange(n_prompts), idx]
        chosen_rnet = R_net[np.arange(n_prompts), idx]

        print(f"{lam:>10.4f}  {chosen_r.mean():>10.4f}  {chosen_c.mean():>10.4f}  {chosen_rnet.mean():>10.4f}")
        summary_points.append({
            "lam": lam,
            "E_r": float(chosen_r.mean()), "E_c": float(chosen_c.mean()), "E_r_net": float(chosen_rnet.mean()),
        })

        out_dir = os.path.join(args["base_out_dir"], f"lam-{lam}")
        os.makedirs(out_dir, exist_ok=True)
        out_dataset = Dataset.from_dict({
            "prompt":     prompts,
            "completion": chosen_completions,
            "reward":     chosen_r.tolist(),
            "cost":       chosen_c.tolist(),
            "r_net":      chosen_rnet.tolist(),
        })
        out_dataset.save_to_disk(out_dir)

    # Machine-readable summary (consumed by scripts/make_prf_data.py)
    summary_path = os.path.join(args["base_out_dir"], "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "eval_dir": args["eval_dir"], "mode": mode, "tilted_mode": tilted_mode, "beta": beta,
            "n_prompts": int(n_prompts), "n_completions": int(n_completions),
            "lam_star": lam_star,
            "points": summary_points,
        }, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
