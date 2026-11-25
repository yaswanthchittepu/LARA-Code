"""Collect mean GPT helpfulness / harmlessness (rate_gpt.py outputs) into one JSON for plots/plot_gpt_ratings.py.

  --method NAME=DIR    DIR contains `gpt_rating-lam-<lam>/rate_*.json` sub-directories (lambdas are discovered;
                       scripts/sweep_lambdas.py --rate writes this layout)
  --baseline NAME=P    P is a rate_*.json file, or a directory containing one
  --lam_star NAME=V / --calib_json   as in collect_results.py

Output: {"<model>": {"baselines": {NAME: {helpfulness, harmlessness, n}},
                     "methods":   {NAME: {"points": [{lam, helpfulness, harmlessness, n}], "lam_star": V}}}}
"""
import argparse
import glob
import json
import os
import re

import numpy as np

LAM_DIR = re.compile(r"^gpt_rating-lam-([0-9.eE+-]+)$")


def split_kv(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {s!r}")
    return tuple(s.split("=", 1))


def load_scores(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "rate_*.json")))
        if not files:
            raise FileNotFoundError(f"no rate_*.json in {path}")
        path = files[0]
    data = json.load(open(path))
    rows = data["results"] if "results" in data else data
    help_ = [r["helpfulness_score"] for r in rows if r.get("helpfulness_score", -1) >= 0]
    harm = [r["harmlessness_score"] for r in rows if r.get("harmlessness_score", -1) >= 0]
    return dict(helpfulness=float(np.mean(help_)), harmlessness=float(np.mean(harm)), n=len(rows))


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--model_name", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--method", type=split_kv, action="append", default=[])
ap.add_argument("--baseline", type=split_kv, action="append", default=[])
ap.add_argument("--lam_star", type=split_kv, action="append", default=[])
ap.add_argument("--calib_json", default=None)
a = ap.parse_args()

calib = json.load(open(a.calib_json)) if a.calib_json else {}
lam_star = {k: float(v) for k, v in a.lam_star}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
res_all = json.load(open(a.out)) if os.path.exists(a.out) else {}
res = res_all.setdefault(a.model_name, {"baselines": {}, "methods": {}})

for name, path in a.baseline:
    try:
        res["baselines"][name] = load_scores(path)
        print(f"baseline {name}: {res['baselines'][name]}")
    except Exception as e:
        print(f"baseline {name}: skipped ({e})")
for name, base in a.method:
    found = sorted((float(LAM_DIR.match(d).group(1)), d) for d in (os.listdir(base) if os.path.isdir(base) else [])
                   if LAM_DIR.match(d))
    pts = []
    for lam, d in found:
        try:
            pts.append({"lam": lam, **load_scores(os.path.join(base, d))})
            print(f"{name}(lam={lam:g}): {pts[-1]}")
        except Exception as e:
            print(f"{name}(lam={lam:g}): skipped ({e})")
    if pts:
        entry = {"points": pts}
        ls = lam_star.get(name, calib.get("lambda_star"))
        if ls is not None:
            entry["lam_star"] = ls
        res["methods"][name] = entry
json.dump(res_all, open(a.out, "w"), indent=2)
print(f"Results written to {a.out}")
