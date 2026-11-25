"""ARGS / PARGS decoding for safe-RLHF.

At each token step, the LLM proposes top-k candidates. Each is scored:
    score_i = log_prob(token_i) + w * (r(prompt + y_{1:i-1} + token_i)
                                       - lam * c(prompt + y_{1:i-1} + token_i))

Pass full RM/CM for ARGS, partial RM/CM for PARGS.
Both support greedy (argmax) or stochastic (Categorical) selection via --mode.

Usage:
    python scripts/generate_args_pargs.py --lam 0.5 --mode greedy ...

All defaults are read from args/generate_args_pargs.yaml; CLI args override yaml values.
"""

import os
import argparse
import yaml

import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from safe_rlhf.models import AutoModelForScore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:"


def score_batch(model, tokenizer, texts, max_length):
    """Score a batch of strings. Returns end_scores shape (B,)."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(DEVICE)
    with torch.no_grad():
        return model(**enc).end_scores.squeeze(-1)  # (B,)


def decode_one(
    prompt, llm, llm_tokenizer,
    rm, cm, score_tokenizer,
    lam, w, topk, max_new_tokens, max_score_length, mode,
):
    prompt_ids = llm_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated  = []

    for _ in range(max_new_tokens):
        input_ids = torch.cat(
            [prompt_ids, torch.tensor([generated], device=DEVICE)], dim=-1
        ) if generated else prompt_ids

        with torch.no_grad():
            logits = llm(input_ids).logits[0, -1]          # (vocab,)
        log_probs = F.log_softmax(logits, dim=-1)
        topk_lp, topk_ids = log_probs.topk(topk)           # (k,)

        # score each candidate: prompt + y_{1:i-1} + token_i
        candidate_texts = [
            prompt + llm_tokenizer.decode(generated + [tok.item()], skip_special_tokens=False)
            for tok in topk_ids
        ]
        rewards    = score_batch(rm, score_tokenizer, candidate_texts, max_score_length)
        costs      = score_batch(cm, score_tokenizer, candidate_texts, max_score_length)
        lagrangian = rewards - lam * costs                  # (k,)

        scores = topk_lp + w * lagrangian                  # (k,)

        if mode == "greedy":
            chosen = scores.argmax()
        else:  # stochastic
            chosen = torch.distributions.Categorical(logits=scores).sample()

        next_token = topk_ids[chosen].item()
        generated.append(next_token)

        if next_token == llm_tokenizer.eos_token_id:
            break

    return llm_tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":

    with open("./args/generate_args_pargs.yaml", "r") as f:
        args = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_path",         type=str,   default=None, help="Path to the SFT (reference) LLM")
    parser.add_argument("--rm_path",          type=str,   default=None, help="Path to reward model (full for ARGS, partial for PARGS)")
    parser.add_argument("--cm_path",          type=str,   default=None, help="Path to cost model (full for ARGS, partial for PARGS)")
    parser.add_argument("--tokenizer",        type=str,   default=None, help="Tokenizer for the LLM")
    parser.add_argument("--score_tokenizer",  type=str,   default=None, help="Tokenizer for RM/CM; defaults to --tokenizer if not set")
    parser.add_argument("--dataset_path",     type=str,   default=None, help="HuggingFace dataset path or local json file path")
    parser.add_argument("--dataset_split",    type=str,   default=None, help="Dataset split to generate on (e.g. test); ignored when dataset_path is a json file")
    parser.add_argument("--out_dir",          type=str,   default=None, help="Directory to save the output HF dataset")
    parser.add_argument("--mode",             type=str,   default=None, choices=["greedy", "stochastic"], help="greedy=argmax, stochastic=sample from score distribution")
    parser.add_argument("--lam",              type=float, default=None, help="lambda* from dual calibration — weight on cost penalty")
    parser.add_argument("--w",                type=float, default=None, help="Weight on the Lagrangian (r - lam*c) relative to LLM log-prob")
    parser.add_argument("--topk",             type=int,   default=None, help="Number of top-k candidate tokens scored per step")
    parser.add_argument("--max_new_tokens",   type=int,   default=None, help="Maximum number of tokens to generate per prompt")
    parser.add_argument("--max_score_length", type=int,   default=None, help="Max token length passed to RM/CM")
    parser.add_argument("--n_prompts",        type=int,   default=None, help="Number of prompts to generate for; null = full dataset")
    parser.add_argument("--seed",             type=int,   default=None, help="Random seed for dataset shuffling")
    parser.add_argument("--cache_dir",        type=str,   default=None, help="HuggingFace cache directory")
    parse_args = parser.parse_args()

    for a in vars(parse_args):
        val = getattr(parse_args, a)
        if val is not None:
            print(f"Overriding {a}: {args.get(a)} -> {val}")
            args[a] = val

    if args.get("lam") is None:

        parser.error("--lam is required: the calibrated lambda* (or any lambda in your sweep)")

    lam = float(args["lam"])
    os.makedirs(args["out_dir"], exist_ok=True)
    print(f"\nlam={lam}  mode={args['mode']}  topk={args['topk']}  w={args['w']}\n")

    print("Loading LLM...", flush=True)
    llm_tokenizer = AutoTokenizer.from_pretrained(args["tokenizer"], cache_dir=args["cache_dir"])
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        args["sft_path"], cache_dir=args["cache_dir"],
        torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

    print("Loading RM...", flush=True)
    rm = AutoModelForScore.from_pretrained(
        args["rm_path"], score_type="reward", cache_dir=args["cache_dir"],
        torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

    print("Loading CM...", flush=True)
    cm = AutoModelForScore.from_pretrained(
        args["cm_path"], score_type="cost", cache_dir=args["cache_dir"],
        torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

    score_tokenizer = AutoTokenizer.from_pretrained(
        args.get("score_tokenizer") or args["tokenizer"], cache_dir=args["cache_dir"]
    )
    if score_tokenizer.pad_token is None:
        score_tokenizer.pad_token = score_tokenizer.eos_token

    if args["dataset_path"].endswith(".json"):
        import json
        with open(args["dataset_path"]) as f:
            data = Dataset.from_list(json.load(f))
    else:
        data = load_dataset(args["dataset_path"], cache_dir=args["cache_dir"])[args["dataset_split"]]
        if args.get("n_prompts"):
            data = data.shuffle(seed=args.get("seed") or 42).select(range(args["n_prompts"]))
    print(f"Generating for {len(data)} prompts\n", flush=True)

    results = []
    for row in tqdm(data):
        prompt     = PROMPT_TEMPLATE.format(prompt=row["prompt"])
        completion = decode_one(
            prompt=prompt, llm=llm, llm_tokenizer=llm_tokenizer,
            rm=rm, cm=cm, score_tokenizer=score_tokenizer,
            lam=lam, w=float(args["w"]), topk=int(args["topk"]),
            max_new_tokens=int(args["max_new_tokens"]),
            max_score_length=int(args["max_score_length"]),
            mode=args["mode"],
        )
        results.append({"prompt": prompt, "completion": completion})

    out_dataset = Dataset.from_list(results)
    out_dataset.save_to_disk(args["out_dir"])
    print(f"Saved {len(out_dataset)} generations to {args['out_dir']}", flush=True)
