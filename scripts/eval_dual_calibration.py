"""Phase 1b — Score generated completions with a reward model and cost model.

Run with the safe-rlhf conda environment.

Input:  HuggingFace Dataset on disk (from generate_dual_calibration.py) with columns:
            prompt_id  — integer index of the originating prompt
            prompt     — the formatted prompt string
            completion — one generated completion

Output: same dataset with two extra columns added:
            reward     — scalar reward score (float)
            cost       — scalar cost score (float)

Usage:
    python scripts/eval_dual_calibration.py \
        --gen_dir /path/to/generations \
        --out_dir /path/to/scored_output

All defaults are read from args/eval_dual_calibration.yaml,
CLI args override yaml values.
"""

import os, sys
import argparse
import yaml

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from safe_rlhf.models import AutoModelForScore


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def score_dataset(model, tokenizer, texts, batch_size, max_length):
    """Run model over a list of strings and return end_scores as a list of floats."""
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        scores.extend(out.end_scores.squeeze(-1).tolist())
    return scores


if __name__ == "__main__":

    with open("./args/eval_dual_calibration.yaml", "r") as f:
        args = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--rm_path", type=str, default=None)
    parser.add_argument("--cm_path", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None, help="Shared tokenizer; overridden by --rm_tokenizer/--cm_tokenizer")
    parser.add_argument("--rm_tokenizer", type=str, default=None, help="Tokenizer for RM; defaults to --tokenizer if not set")
    parser.add_argument("--cm_tokenizer", type=str, default=None, help="Tokenizer for CM; defaults to --tokenizer if not set")
    parser.add_argument("--gen_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parse_args = parser.parse_args()

    for a in vars(parse_args):
        val = getattr(parse_args, a)
        if val == "NULL":
            print(f"Overriding {a} from {args[a]} to None...\n")
            args[a] = None
        elif val is not None:
            print(f"Overriding {a} from {args.get(a)} to {val}...\n")
            args[a] = val

    os.makedirs(args["out_dir"], exist_ok=True)

    # ---- Load generated dataset ----
    gen_dataset = Dataset.load_from_disk(args["gen_dir"])
    print(f"Loaded {len(gen_dataset)} rows from {args['gen_dir']}", flush=True)

    # Concatenate prompt + completion into a single string for scoring
    texts = [p + c for p, c in zip(gen_dataset["prompt"], gen_dataset["completion"])]

    shared_tok = args.get("tokenizer")

    def load_tokenizer(model_specific, fallback_model_path):
        path = model_specific or shared_tok or fallback_model_path
        tok = AutoTokenizer.from_pretrained(path, cache_dir=args["cache_dir"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    # ---- Reward model ----
    print("Loading reward model...", flush=True)
    rm_tokenizer = load_tokenizer(args.get("rm_tokenizer"), args["rm_path"])
    rm = AutoModelForScore.from_pretrained(
        args["rm_path"],
        score_type="reward",
        cache_dir=args["cache_dir"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    rm.eval()
    rewards = score_dataset(rm, rm_tokenizer, texts, args["batch_size"], args["max_length"])
    del rm
    torch.cuda.empty_cache()

    # ---- Cost model ----
    print("Loading cost model...", flush=True)
    cm_tokenizer = load_tokenizer(args.get("cm_tokenizer"), args["cm_path"])
    cm = AutoModelForScore.from_pretrained(
        args["cm_path"],
        score_type="cost",
        cache_dir=args["cache_dir"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    cm.eval()
    costs = score_dataset(cm, cm_tokenizer, texts, args["batch_size"], args["max_length"])
    del cm
    torch.cuda.empty_cache()

    # ---- Save ----
    scored_dataset = gen_dataset.add_column("reward", rewards).add_column("cost", costs)
    scored_dataset.save_to_disk(args["out_dir"])
    print(f"Saved {len(scored_dataset)} scored rows to {args['out_dir']}", flush=True)
