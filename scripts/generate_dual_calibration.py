"""Phase 1a — Generate N completions per prompt using vLLM.

Run with the llm_gen conda environment (has vLLM).

Output: HuggingFace Dataset saved to disk with columns:
    prompt     — the formatted prompt string
    completion — one completion (flattened: N rows per prompt)

Usage:
    python scripts/generate_dual_calibration.py \
        --model_path /path/to/sft_model \
        --out_dir /path/to/output

All defaults are read from args/generate_dual_calibration.yaml,
CLI args override yaml values.
"""

from collections import defaultdict
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
import os, sys, time, argparse, yaml


PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:"


if __name__ == "__main__":

    with open('./args/generate_dual_calibration.yaml', 'r') as file:
        args = yaml.safe_load(file)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--n_completions", type=int, default=None, help="Number of completions to sample per prompt")
    parser.add_argument("--max_completion_length", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--calib_size", type=int, default=None, help="Number of prompts to use for calibration")
    parser.add_argument("--calib_seed", type=int, default=None, help="Seed for shuffling the calibration dataset (default: 4321)")
    parser.add_argument("--debug_small_data", action="store_true", default=False)
    parse_args = parser.parse_args()

    for a in vars(parse_args):
        val = getattr(parse_args, a)
        if val == "NULL":
            print(f"Overriding {a} from {args[a]} to None...\n")
            args[a] = None
        elif val is not None and val is not False:
            print(f"Overriding {a} from {args.get(a)} to {val}...\n")
            args[a] = val

    os.makedirs(args["out_dir"], exist_ok=True)

    # ---- Load prompts ----
    if args["dataset_path"].endswith(".json"):
        import json
        with open(args["dataset_path"]) as f:
            data = Dataset.from_list(json.load(f))
    else:
        data = load_dataset(args["dataset_path"], cache_dir=args["cache_dir"])[args["dataset_split"]]

    if args.get("calib_size") is not None:
        seed = args.get("calib_seed") or 4321
        data = data.shuffle(seed=seed).select(range(args["calib_size"]))
        print(f"Calibration subset: {len(data)} prompts (seed={seed})", flush=True)

    if args.get("debug_small_data"):
        data = data.select(range(10))

    # Format prompts — adjust template if your model uses a different chat format
    data = data.map(
        lambda batch: {
            "prompt": [PROMPT_TEMPLATE.format(prompt=p) for p in batch["prompt"]]
        },
        batched=True,
        remove_columns=[c for c in data.column_names if c != "prompt"],
    )

    # ---- vLLM generation ----
    n = args["n_completions"]

    sampling_params = SamplingParams(
        n=n,                              # N completions per prompt
        presence_penalty=0,
        frequency_penalty=0,
        repetition_penalty=1.0,
        temperature=1.2,
        top_p=1.0,
        top_k=50,
        ignore_eos=False,
        max_tokens=args["max_completion_length"],
        min_tokens=0,
        detokenize=True,
    )

    model = LLM(
        model=args["model_path"],
        tokenizer=args["tokenizer"],
        download_dir=args["cache_dir"],
    )

    print(f"Generating {n} completions for {len(data)} prompts ({len(data) * n} total)...", flush=True)
    start_t = time.time()
    outputs = model.generate(data["prompt"], sampling_params)
    print(f"Generation done in {(time.time() - start_t) / 60:.2f} minutes", flush=True)

    # ---- Flatten to (prompt_id, prompt, completion) rows ----
    gen_data = defaultdict(list)
    for prompt_id, output in enumerate(outputs):
        for completion in output.outputs:          # N completions per prompt
            gen_data["prompt_id"].append(prompt_id)
            gen_data["prompt"].append(output.prompt)
            gen_data["completion"].append(completion.text)

    gen_dataset = Dataset.from_dict(gen_data)
    gen_dataset.save_to_disk(args["out_dir"])
    print(f"Saved {len(gen_dataset)} rows to {args['out_dir']}", flush=True)
