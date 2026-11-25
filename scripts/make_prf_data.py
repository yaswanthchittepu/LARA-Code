"""Build the JSON read by plots/plot_prf.py from pipeline outputs.

Combines
  * the per-lambda E[r]/E[c] summary written by scripts/bon_select.py  (<base_out_dir>/summary.json)
  * the calibrated lambda* / tau written by scripts/calibrate_dual_bisect.py (lambda_star.json)

Output schema (merged into --out if it already exists):
    {"<model_name>": {"tau": float, "lam_star": float,
                      "points": [{"lam": ..., "E_r": ..., "E_c": ...}, ...]}}

Usage:
    python scripts/make_prf_data.py \
        --bon_summary workspace/llama3.1-8b/lmbda/lambda-generations-gold/summary.json \
        --calib_json  workspace/llama3.1-8b/lmbda/calibration/lambda_star.json \
        --model_name Llama --out results_lara/prf_data_llama3.1-8b.json
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--bon_summary", required=True, help="summary.json from bon_select.py")
parser.add_argument("--calib_json", default=None, help="lambda_star.json from calibrate_dual_bisect.py")
parser.add_argument("--tau", type=float, default=None, help="override/supply tau")
parser.add_argument("--lam_star", type=float, default=None, help="override/supply lambda* (e.g. the bootstrap percentile)")
parser.add_argument("--model_name", default="Llama", help="top-level key (must match plot_prf.py --model)")
parser.add_argument("--out", required=True)
a = parser.parse_args()

with open(a.bon_summary) as f:
    points = [{"lam": p["lam"], "E_r": p["E_r"], "E_c": p["E_c"]} for p in json.load(f)["points"]]

calib = {}
if a.calib_json:
    with open(a.calib_json) as f:
        calib = json.load(f)
tau = a.tau if a.tau is not None else calib.get("tau")
lam_star = a.lam_star if a.lam_star is not None else calib.get("lambda_star")
if tau is None or lam_star is None:
    parser.error("need tau and lambda*: pass --calib_json or both --tau and --lam_star")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
data = json.load(open(a.out)) if os.path.exists(a.out) else {}
data[a.model_name] = {"tau": tau, "lam_star": lam_star, "points": sorted(points, key=lambda p: p["lam"])}
with open(a.out, "w") as f:
    json.dump(data, f, indent=2)
print(f"Wrote {a.out}: model={a.model_name} tau={tau} lam_star={lam_star} points={len(points)}")
