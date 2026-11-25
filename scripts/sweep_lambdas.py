"""Run one decoder over a list of lambdas (plus the calibrated lambda*) and score the outputs.

Output layout under --root (what scripts/make_figures.py expects):
  BoN    : <root>/BoN/lam-<lam>/                   (scored datasets, written by bon_select.py; needs --eval_dir)
  ARGS / PARGS / CDQ :
           <root>/<METHOD>/generations-lam-<lam>/    (generate_args_pargs.py / generate_cdq.py)
           <root>/<METHOD>/evaluations-lam-<lam>/    (eval_dual_calibration.py -> reward/cost columns)
  --rate : <root>/<METHOD>/gpt_rating-lam-<lam>/rate_<id>.json   (rate_gpt.py; costs OpenAI credits)

Finished outputs are skipped, so an interrupted sweep can simply be re-run (use --force to redo).
Everything after the script name is passed through, e.g. model paths, which otherwise come from args/*.yaml.

Examples:
  python scripts/sweep_lambdas.py --method ARGS --root results/llama3.2-3b --lambdas 0.0 2.0 \
      --calib_json results/llama3.2-3b/calib/lambda_star.json \
      --gen_args "--sft_path X --rm_path Y --cm_path Z --tokenizer T --n_prompts 1024" \
      --eval_args "--rm_path Y --cm_path Z --tokenizer T"
  python scripts/sweep_lambdas.py --method BoN --root results/llama3.2-3b --lambdas 0.0 2.0 \
      --calib_json results/llama3.2-3b/calib/lambda_star.json --eval_dir <scored candidates, N per prompt>
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = {"ARGS": "generate_args_pargs.py", "PARGS": "generate_args_pargs.py", "CDQ": "generate_cdq.py"}

p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog="\n".join(__doc__.splitlines()[1:]))
p.add_argument("--method", required=True, type=lambda s: {"BON": "BoN", "CD-Q": "CDQ"}.get(s.upper(), s.upper()),
               help="BoN | ARGS | PARGS | CDQ")
p.add_argument("--root", required=True)
p.add_argument("--lambdas", type=float, nargs="+", required=True)
p.add_argument("--calib_json", default=None, help="lambda_star.json; its lambda* is added to the sweep")
p.add_argument("--gen_args", default="", help="extra args for the generation script (quoted string)")
p.add_argument("--eval_args", default="", help="extra args for eval_dual_calibration.py (quoted string)")
p.add_argument("--eval_dir", default=None, help="BoN only: scored candidates (prompt_id, prompt, completion, reward, cost)")
p.add_argument("--rate", action="store_true", help="also rate the outputs with GPT (rate_gpt.py)")
p.add_argument("--rate_max_samples", type=int, default=None, help="rate only this many samples per lambda")
p.add_argument("--force", action="store_true")
p.add_argument("--dry_run", action="store_true")
a = p.parse_args()
if a.method != "BoN" and a.method not in GEN:
    p.error(f"--method must be BoN, ARGS, PARGS or CDQ (got {a.method})")

lams = list(a.lambdas)
if a.calib_json:
    lam_star = float(json.load(open(a.calib_json))["lambda_star"])
    lams.append(lam_star)
    print(f"lambda* = {lam_star} added to the sweep")
lams = sorted(set(lams))
M, root = a.method, os.path.abspath(a.root)


def run(cmd):
    print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if not a.dry_run:
        subprocess.run(cmd, cwd=REPO, check=True)


def done(path):
    return os.path.exists(path) and not a.force


py = sys.executable
S = lambda name: os.path.join(REPO, "scripts", name)

if M == "BoN":
    if not a.eval_dir:
        p.error("--method BoN needs --eval_dir")
    base = os.path.join(root, "BoN")
    cmd = [py, S("bon_select.py"), "--eval_dir", a.eval_dir, "--base_out_dir", base, "--mode", "argmax",
           "--lambdas", *[repr(l) for l in lams]]
    if not all(done(os.path.join(base, f"lam-{l}")) for l in lams):
        run(cmd)
    gen_dir = lambda l: os.path.join(base, f"lam-{l}")
else:
    for lam in lams:
        gen = os.path.join(root, M, f"generations-lam-{lam}")
        ev = os.path.join(root, M, f"evaluations-lam-{lam}")
        if not done(gen):
            run([py, S(GEN[M]), "--lam", repr(lam), "--out_dir", gen, *shlex.split(a.gen_args)])
        if not done(ev):
            run([py, S("eval_dual_calibration.py"), "--gen_dir", gen, "--out_dir", ev, *shlex.split(a.eval_args)])
    gen_dir = lambda l: os.path.join(root, M, f"generations-lam-{l}")

if a.rate:
    for lam in lams:
        out = os.path.join(root, M, f"gpt_rating-lam-{lam}")
        if done(os.path.join(out, f"rate_{M}-lam-{lam}.json")):
            continue
        cmd = [py, S("rate_gpt.py"), "--dataset_path", gen_dir(lam), "--output_dir", out, "--model_id", f"{M}-lam-{lam}"]
        if a.rate_max_samples:
            cmd += ["--max_samples", str(a.rate_max_samples)]
        run(cmd)
print(f"\nDone: {M} at lambdas {lams} under {root}")
