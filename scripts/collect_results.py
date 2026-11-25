"""Collect E[r] / E[c] for every method and baseline into one JSON (input of plots/plot_pareto_2x2.py).

Works for any decoder: Best-of-N (output of bon_select.py), ARGS / PARGS / CD-Q (generations scored with
eval_dual_calibration.py), and fine-tuned baselines (SFT, SafeDPO, Safe-RLHF) scored the same way.

  --method NAME=DIR    DIR contains one scored HF dataset per lambda, in sub-directories named
                       `lam-<value>` or `evaluations-lam-<value>` (lambdas are discovered, not configured).
                       bon_select.py writes `lam-<value>` directly; for token-level methods score each
                       lambda's generations into `evaluations-lam-<value>` with eval_dual_calibration.py.
  --baseline NAME=DIR  DIR is a single scored HF dataset (columns: reward, cost).
  --lam_star NAME=V    calibrated lambda* for a method (repeatable); `--calib_json` supplies a default for
                       all methods that have no explicit --lam_star.

Output schema (merged into --out if it exists):
  {"<model_name>": {"baselines": {NAME: {E_r, E_c, pearson, spearman, n}},
                    "methods":   {NAME: {"points": [{lam, E_r, E_c, E_r_net, pearson, spearman, n}, ...],
                                         "lam_star": V (if known), "tau": T (if known)}}}}

Example:
  python scripts/collect_results.py --model_name llama3.2-3b --out results_lara/pareto_data_llama3.2-3b_gold.json \
      --calib_json workspace/llama3.2-3b/calib/lambda_star.json \
      --method BoN=workspace/llama3.2-3b/BoN-20/lambda-generations-gold \
      --method ARGS=workspace/llama3.2-3b/ARGS/evaluations-gold \
      --baseline SFT=workspace/llama3.2-3b/sft/evaluations-gold
"""
import argparse
import json
import os
import re

import numpy as np
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr

LAM_DIR = re.compile(r"^(?:evaluations-)?lam-([0-9.eE+-]+)$")


def split_kv(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {s!r}")
    k, v = s.split("=", 1)
    return k, v


def load_metrics(path, lam=0.0):
    ds = Dataset.load_from_disk(path)
    R = np.array(ds["reward"], dtype=np.float64)
    C = np.array(ds["cost"], dtype=np.float64)
    if len(R) >= 2 and R.std() > 0 and C.std() > 0:
        pr, sr = pearsonr(R, C)[0], spearmanr(R, C)[0]
    else:
        pr = sr = float("nan")
    return dict(E_r=R.mean(), E_c=C.mean(), E_r_net=(R - lam * C).mean(), pearson=pr, spearman=sr, n=len(R))


def rnd(d):
    # keep lam exact (lambda* is matched against it by the plot); round the metrics
    return {k: (int(v) if k == "n" else float(v) if k == "lam" else round(float(v), 6)) for k, v in d.items()}


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--model_name", required=True, help="top-level key, e.g. llama3.2-3b")
parser.add_argument("--out", required=True)
parser.add_argument("--method", type=split_kv, action="append", default=[], metavar="NAME=DIR")
parser.add_argument("--baseline", type=split_kv, action="append", default=[], metavar="NAME=DIR")
parser.add_argument("--lam_star", type=split_kv, action="append", default=[], metavar="NAME=V")
parser.add_argument("--calib_json", default=None, help="lambda_star.json from calibrate_dual_bisect.py")
args = parser.parse_args()

calib = json.load(open(args.calib_json)) if args.calib_json else {}
lam_star = {k: float(v) for k, v in args.lam_star}

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
results = json.load(open(args.out)) if os.path.exists(args.out) else {}
res = results.setdefault(args.model_name, {"baselines": {}, "methods": {}})

HDR = f"{'name':>18}  {'E[r]':>10}  {'E[c]':>10}  {'E[r_net]':>10}  {'pearson':>9}  {'spearman':>9}  {'n':>6}"
fmt = lambda name, m: (f"{name:>18}  {m['E_r']:>10.4f}  {m['E_c']:>10.4f}  {m['E_r_net']:>10.4f}"
                       f"  {m['pearson']:>9.4f}  {m['spearman']:>9.4f}  {m['n']:>6}")

print(f"\nMODEL={args.model_name}\n\n=== Baselines ===\n{HDR}\n{'-' * 90}")
for name, path in args.baseline:
    if not os.path.exists(path):
        print(f"{name:>18}  MISSING ({path})")
        continue
    m = load_metrics(path)
    print(fmt(name, m))
    res["baselines"][name] = rnd({k: m[k] for k in ("E_r", "E_c", "pearson", "spearman", "n")})

for name, base in args.method:
    print(f"\n=== {name} ===\n{HDR}\n{'-' * 90}")
    if not os.path.isdir(base):
        print(f"MISSING ({base})")
        continue
    found = sorted(((float(LAM_DIR.match(d).group(1)), d) for d in os.listdir(base) if LAM_DIR.match(d)))
    if not found:
        print(f"no lam-<value> / evaluations-lam-<value> sub-directories in {base}")
        continue
    points = []
    for lam, d in found:
        m = load_metrics(os.path.join(base, d), lam=lam)
        print(fmt(f"{name}(lam={lam:g})", m))
        points.append(rnd({"lam": lam, **m}))
    entry = {"points": points}
    ls = lam_star.get(name, calib.get("lambda_star"))
    if ls is not None:
        entry["lam_star"] = ls
    if calib.get("tau") is not None:
        entry["tau"] = calib["tau"]
    res["methods"][name] = entry

with open(args.out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults written to {args.out}")
