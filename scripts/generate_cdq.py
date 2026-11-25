"""CDQ decoding for safe-RLHF.

At each token step, the LLM proposes top-k candidates. Each is scored using
value models (V_r, V_c) that predict long-horizon value for every next token
in a single forward pass:

    score_i = log_prob(token_i) + w * (V_r(token_i | prefix) - lam * V_c(token_i | prefix))

V_r and V_c are separate value models trained with CDQ (Bellman MSE) using the
reward and cost models respectively. Unlike ARGS/PARGS, no RM/CM forward passes
are needed at inference — the value head gives all vocab values in one shot.

Usage:
    python scripts/generate_cdq.py --lam 0.5 --mode greedy ...

All defaults are read from args/generate_cdq.yaml; CLI args override yaml values.
"""

import os
import argparse
import yaml

import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from safe_rlhf.models import AutoModelForValue

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:"


def decode_one(
    prompt, llm, tokenizer,
    v_r, v_c,
    lam, w, topk, max_new_tokens, mode,
):
    """Generate one completion using CDQ token-level decoding.

    At each step:
      1. LLM forward pass -> log probs over full vocab
      2. Take top-k token indices
      3. V_r / V_c forward pass on current prefix -> values[:, -1, topk_ids]
      4. score_i = log_prob_i + w * (V_r_i - lam * V_c_i)
      5. greedy (argmax) or stochastic (Categorical) selection
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated  = []

    for _ in range(max_new_tokens):
        input_ids = torch.cat(
            [prompt_ids, torch.tensor([generated], device=DEVICE)], dim=-1
        ) if generated else prompt_ids

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        with torch.no_grad():
            # LLM: next-token log probs
            logits    = llm(input_ids).logits[0, -1]          # (vocab,)
            log_probs = F.log_softmax(logits, dim=-1)
            topk_lp, topk_ids = log_probs.topk(topk)          # (k,)

            # Value models: V for all vocab tokens at last position
            vr_all = v_r(input_ids, attention_mask=attention_mask).values[0, -1, :]  # (vocab,)
            vc_all = v_c(input_ids, attention_mask=attention_mask).values[0, -1, :]  # (vocab,)

        # Index into top-k tokens
        vr_topk = vr_all[topk_ids]   # (k,)
        vc_topk = vc_all[topk_ids]   # (k,)

        scores = topk_lp + w * (vr_topk - lam * vc_topk)     # (k,)

        if mode == "greedy":
            chosen = scores.argmax()
        else:  # stochastic
            chosen = torch.distributions.Categorical(logits=scores).sample()

        next_token = topk_ids[chosen].item()
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":

    with open("./args/generate_cdq.yaml", "r") as f:
        args = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_path",       type=str,   default=None, help="Path to the SFT (reference) LLM")
    parser.add_argument("--vr_path",        type=str,   default=None, help="Path to reward value model (V_r)")
    parser.add_argument("--vc_path",        type=str,   default=None, help="Path to cost value model (V_c)")
    parser.add_argument("--tokenizer",      type=str,   default=None, help="Tokenizer shared by LLM and value models")
    parser.add_argument("--dataset_path",   type=str,   default=None, help="HuggingFace dataset path")
    parser.add_argument("--dataset_split",  type=str,   default=None, help="Dataset split to generate on (e.g. test)")
    parser.add_argument("--out_dir",        type=str,   default=None, help="Directory to save the output HF dataset")
    parser.add_argument("--mode",           type=str,   default=None, choices=["greedy", "stochastic"], help="greedy=argmax, stochastic=sample from score distribution")
    parser.add_argument("--lam",            type=float, default=None, help="lambda* from dual calibration — weight on cost value penalty")
    parser.add_argument("--w",              type=float, default=None, help="Weight on (V_r - lam*V_c) relative to LLM log-prob")
    parser.add_argument("--topk",           type=int,   default=None, help="Number of top-k candidate tokens scored per step")
    parser.add_argument("--max_new_tokens", type=int,   default=None, help="Maximum number of tokens to generate per prompt")
    parser.add_argument("--n_prompts",      type=int,   default=None, help="Number of prompts to generate for; null = full dataset")
    parser.add_argument("--seed",           type=int,   default=None, help="Random seed for dataset shuffling")
    parser.add_argument("--cache_dir",      type=str,   default=None, help="HuggingFace cache directory")
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
    tokenizer = AutoTokenizer.from_pretrained(args["tokenizer"], cache_dir=args["cache_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        args["sft_path"], cache_dir=args["cache_dir"],
        torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

    print("Loading V_r...", flush=True)
    v_r = AutoModelForValue.from_pretrained(
        args["vr_path"], cache_dir=args["cache_dir"],
        torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

    print("Loading V_c...", flush=True)
    v_c = AutoModelForValue.from_pretrained(
        args["vc_path"], cache_dir=args["cache_dir"],
        torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

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
            prompt=prompt, llm=llm, tokenizer=tokenizer,
            v_r=v_r, v_c=v_c,
            lam=lam, w=float(args["w"]), topk=int(args["topk"]),
            max_new_tokens=int(args["max_new_tokens"]),
            mode=args["mode"],
        )
        results.append({"prompt": prompt, "completion": completion})

    out_dataset = Dataset.from_list(results)
    out_dataset.save_to_disk(args["out_dir"])
    print(f"Saved {len(out_dataset)} generations to {args['out_dir']}", flush=True)
