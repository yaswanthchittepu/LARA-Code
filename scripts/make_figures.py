"""Make the paper-style figures from a results directory (no constants to edit).

Expected layout under --root (see scripts/sweep_lambdas.py, scripts/bon_select.py):
  <root>/calib/lambda_star.json                        calibrate_dual_bisect.py
  <root>/BoN/lam-<lam>/ , <root>/BoN/summary.json      bon_select.py
  <root>/{ARGS,PARGS,CDQ}/evaluations-lam-<lam>/       sweep_lambdas.py
  <root>/<METHOD>/gpt_rating-lam-<lam>/rate_*.json     sweep_lambdas.py --rate     (optional)
  <root>/baselines/{SFT,SafeDPO,Safe-RLHF}/            scored dataset per baseline  (optional)
  <root>/baselines/<NAME>/gpt_rating/rate_*.json       GPT ratings of a baseline    (optional)

Figures written to --out_dir (default <root>/figures):
  pareto_<mode>.{pdf,png}      E[r] vs E[c]            (plots/plot_pareto_2x2.py)
  prf_<model>.{pdf,png}        BoN E[r]/E[c] over lambda (plots/plot_prf.py)         if BoN/summary.json and calib/lambda_star.json exist
  pareto_gpt_<mode>.{pdf,png}  GPT helpfulness vs harmlessness (plots/plot_gpt_ratings.py)  if ratings exist

    python scripts/make_figures.py --root results/llama3.2-3b --model_name llama3.2-3b --eval_mode gold
"""
import argparse
import glob
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ["BoN", "ARGS", "PARGS", "CDQ"]

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                             epilog="\n".join(__doc__.splitlines()[1:]))
ap.add_argument("--root", required=True)
ap.add_argument("--model_name", required=True, choices=["llama3.2-3b", "qwen2.5-3b"],
                help="the plot scripts are set up for the paper's two models")
ap.add_argument("--eval_mode", default="gold", choices=["gold", "trained"], help="label for the output file name")
ap.add_argument("--out_dir", default=None)
a = ap.parse_args()

root = os.path.abspath(a.root)
out = os.path.abspath(a.out_dir or os.path.join(root, "figures"))
os.makedirs(out, exist_ok=True)
mode = "llama" if a.model_name.startswith("llama") else "qwen"
calib = os.path.join(root, "calib", "lambda_star.json")
calib_args = ["--calib_json", calib] if os.path.exists(calib) else []
py = sys.executable


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=out, check=True)


S = lambda *p: os.path.join(REPO, *p)
present = [m for m in METHODS if os.path.isdir(os.path.join(root, m))]
bdir = os.path.join(root, "baselines")
baselines = sorted(d for d in os.listdir(bdir) if os.path.exists(os.path.join(bdir, d, "dataset_info.json"))) if os.path.isdir(bdir) else []
print(f"methods: {present}\nbaselines: {baselines}\nlambda* file: {'yes' if calib_args else 'no (stars omitted)'}")
if not present:
    sys.exit(f"no method directories ({METHODS}) under {root}")

# 1. E[r] vs E[c]
data_json = os.path.join(out, f"pareto_data_{a.model_name}_{a.eval_mode}.json")
if os.path.exists(data_json):
    os.remove(data_json)
cmd = [py, S("scripts", "collect_results.py"), "--model_name", a.model_name, "--out", data_json, *calib_args]
for m in present:
    cmd += ["--method", f"{m}={os.path.join(root, m)}"]
for b in baselines:
    cmd += ["--baseline", f"{b}={os.path.join(bdir, b)}"]
run(cmd)
run([py, S("plots", "plot_pareto_2x2.py"), "--eval_mode", a.eval_mode, "--mode", mode, "--json_dir", out,
     "--out_name", f"pareto_{mode}"])

# 2. BoN E[r]/E[c] across lambda
summary = os.path.join(root, "BoN", "summary.json")
if os.path.exists(summary) and calib_args:
    prf = os.path.join(out, f"prf_data_{a.model_name}.json")
    run([py, S("scripts", "make_prf_data.py"), "--bon_summary", summary, "--model_name", mode.capitalize(), "--out", prf,
         *calib_args])
    run([py, S("plots", "plot_prf.py"), "--json_file", prf, "--model", mode.capitalize(), "--size", a.model_name.split("-")[-1].upper()])

# 3. GPT ratings
rated = [m for m in present if glob.glob(os.path.join(root, m, "gpt_rating-lam-*"))]
if rated:
    gpt_json = os.path.join(out, f"gpt_data_{a.model_name}.json")
    if os.path.exists(gpt_json):
        os.remove(gpt_json)
    cmd = [py, S("scripts", "collect_gpt_ratings.py"), "--model_name", a.model_name, "--out", gpt_json, *calib_args]
    for m in rated:
        cmd += ["--method", f"{m}={os.path.join(root, m)}"]
    for b in baselines:
        gp = os.path.join(bdir, b, "gpt_rating")
        if os.path.isdir(gp):
            cmd += ["--baseline", f"{b}={gp}"]
    run(cmd)
    run([py, S("plots", "plot_gpt_ratings.py"), "--mode", mode, "--json_dir", out, "--out_name", f"pareto_gpt_{mode}"])
else:
    print("no GPT ratings found; skipping the GPT figure")
print(f"\nFigures in {out}")
