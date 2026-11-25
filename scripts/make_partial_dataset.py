"""Build the partial-sequence preference dataset used to train PARGS's reward / cost models.

Subsamples N examples from PKU-SafeRLHF and expands each preference pair into prefixes of both responses at
every --stride tokens (plus the full pair), keeping the original preference / safety labels, so that a
reward / cost model trained on it can score *incomplete* sequences during token-level decoding.

Output (HF datasets on disk):  <out_dir>/train , <out_dir>/test
These are what the dataset names `PKU-SafeRLHF-Partial/train` and `PKU-SafeRLHF-Partial/test` load
(default location ./data/safe_rlhf_partial). Train the partial RM / CM with:

    python scripts/make_partial_dataset.py --tokenizer meta-llama/Llama-3.2-3B-Instruct
    bash scripts/reward-model.sh --model_name_or_path <sft> --output_dir <out> \\
        --train_datasets PKU-SafeRLHF-Partial/train --eval_datasets PKU-SafeRLHF-Partial/test
    bash scripts/cost-model.sh   (same flags)

Use the tokenizer of the model you will train on (it decides where the prefixes are cut).
The defaults (20k train examples, stride 10) give roughly 200k+ training pairs.
"""
import argparse
import os

import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[1:]))
parser.add_argument("--tokenizer", required=True, help="HF name or path of the tokenizer used to cut prefixes")
parser.add_argument("--out_dir", default="./data/safe_rlhf_partial")
parser.add_argument("--dataset_path", default="PKU-Alignment/PKU-SafeRLHF")
parser.add_argument("--n_train", type=int, default=20000, help="train examples to subsample")
parser.add_argument("--n_test", type=int, default=1000, help="test examples to subsample")
parser.add_argument("--stride", type=int, default=10, help="emit one prefix every STRIDE tokens")
parser.add_argument("--min_length", type=int, default=20, help="skip pairs whose longer response has <= this many tokens")
parser.add_argument("--max_length", type=int, default=512, help="drop pairs whose prompt+response exceeds this many tokens")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_proc", type=int, default=4, help="processes for the length filter")
parser.add_argument("--cache_dir", default=None)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
os.makedirs(args.out_dir, exist_ok=True)

raw = load_dataset(args.dataset_path, cache_dir=args.cache_dir)
raw_train = raw["train"].shuffle(seed=args.seed).select(range(min(args.n_train, len(raw["train"]))))
raw_test = raw["test"].shuffle(seed=args.seed).select(range(min(args.n_test, len(raw["test"]))))
print(f"Subsampled - train: {len(raw_train)}, test: {len(raw_test)}")

generation_lengths = []


def expand_to_partials(raw_dataset):
    rows = []
    for row in tqdm(raw_dataset):
        prompt, response_0, response_1 = row["prompt"], row["response_0"], row["response_1"]
        ids_0 = tokenizer.encode(response_0, add_special_tokens=False)
        ids_1 = tokenizer.encode(response_1, add_special_tokens=False)
        max_len = max(len(ids_0), len(ids_1))
        if max_len <= args.min_length:
            continue
        generation_lengths.append(max_len)

        def make_row(r0, r1, gen_len):
            return {
                "prompt": prompt,
                "response_0": r0,
                "response_1": r1,
                "better_response_id": row["better_response_id"],
                "safer_response_id": row["safer_response_id"],
                "is_response_0_safe": row["is_response_0_safe"],
                "is_response_1_safe": row["is_response_1_safe"],
                "generation_length": gen_len,
            }

        for i in range(args.stride, max_len, args.stride):
            rows.append(make_row(tokenizer.decode(ids_0[:i], skip_special_tokens=True),
                                 tokenizer.decode(ids_1[:i], skip_special_tokens=True), i))
        rows.append(make_row(response_0, response_1, max_len))  # always include the full pair
    return rows


print("Processing train split...")
train_rows = expand_to_partials(raw_train)
print("Processing test split...")
test_rows = expand_to_partials(raw_test)
if not train_rows or not test_rows:
    raise SystemExit("no examples left after the min_length filter; lower --min_length or raise --n_train/--n_test")
train_dataset = Dataset.from_list(train_rows).shuffle(seed=args.seed)
test_dataset = Dataset.from_list(test_rows).shuffle(seed=args.seed)
print(f"Before filtering - train: {len(train_dataset)}, test: {len(test_dataset)}")


def within_max_length(example):
    return all(len(tokenizer(example["prompt"] + example[k])["input_ids"]) <= args.max_length
               for k in ("response_0", "response_1"))


train_dataset = train_dataset.filter(within_max_length, num_proc=args.num_proc)
test_dataset = test_dataset.filter(within_max_length, num_proc=args.num_proc)
print(f"After filtering  - train: {len(train_dataset)}, test: {len(test_dataset)}")
print(f"Mean generation length: {np.mean(generation_lengths):.1f} +- {np.std(generation_lengths):.1f}")

train_dataset.save_to_disk(os.path.join(args.out_dir, "train"))
test_dataset.save_to_disk(os.path.join(args.out_dir, "test"))
print(f"Saved to {args.out_dir}")
