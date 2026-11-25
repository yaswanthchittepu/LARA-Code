# LARA: Safe Inference-Time Alignment via Lagrangian Reward Augmentation

Code for **"Safe Inference-Time Alignment via Lagrangian Reward Augmentation"**
(Chittepu, Joshi, Chintala, Niekum) — [arXiv:2607.02781](https://arxiv.org/abs/2607.02781).

LARA calibrates a dual variable `λ*` on a small calibration set and uses the augmented reward
`r − λ*·c` as a drop-in scoring signal for inference-time alignment (Best-of-N, ARGS, PARGS, CD-Q).

Built on [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf) (Apache-2.0).

## Setup

```bash
conda env create -f conda-recipe.yaml   # or: pip install -r requirements.txt
pip install -e .
export WANDB_API_KEY=...    # training only
export HF_TOKEN=...         # gated models (e.g. Llama)
export OPENAI_API_KEY=...   # scripts/rate_gpt.py only
```

Generation (`scripts/generate_dual_calibration.py`) needs [vLLM](https://github.com/vllm-project/vllm); install it in a separate environment.
Run all commands from the repo root. Defaults for each script are in `args/*.yaml`; command-line flags override them.

## Usage

**1. SFT, reward and cost models**
```bash
bash scripts/sft.sh --model_name_or_path <base> --output_dir workspace/<model>/sft
bash scripts/reward-model.sh --model_name_or_path workspace/<model>/sft/pytorch_model.bin --output_dir workspace/<model>/rm
bash scripts/cost-model.sh   --model_name_or_path workspace/<model>/sft/pytorch_model.bin --output_dir workspace/<model>/cm
```
Alternatively use `PKU-Alignment/beaver-7b-unified-{reward,cost}` (the default in `args/eval_dual_calibration.yaml`).

*PARGS only* — train the reward/cost models on partial responses:
```bash
python scripts/make_partial_dataset.py --tokenizer <model-or-tokenizer>      # -> data/safe_rlhf_partial/{train,test}
bash scripts/reward-model.sh --model_name_or_path <sft> --output_dir workspace/<model>/partial-rm \
    --train_datasets PKU-SafeRLHF-Partial/train --eval_datasets PKU-SafeRLHF-Partial/test
bash scripts/cost-model.sh   --model_name_or_path <sft> --output_dir workspace/<model>/partial-cm \
    --train_datasets PKU-SafeRLHF-Partial/train --eval_datasets PKU-SafeRLHF-Partial/test
```
(Other locations: `PKU-SafeRLHF-Partial/train:1.0:<path>`.)

**2. CD-Q value models** (only for CD-Q): `bash scripts/cdq.sh --model_name_or_path <sft> --reward_model_name_or_path <rm> --output_dir <out>` (use the cost model for the cost value model).

**3. Calibration data**
```bash
python scripts/generate_dual_calibration.py     # sample K completions per prompt from the SFT model (vLLM)
python scripts/eval_dual_calibration.py         # score them with the reward and cost models
```

**4. Calibrate λ\***
```bash
python scripts/calibrate_dual_bisect.py --tau -4.0 --out_dir results/<model>/calib [--bootstrap --n_bootstrap 10000]
python scripts/calibrate_dual_tau_sweep.py      # λ* for a list of thresholds
```

**5. Decode over a λ sweep (λ\* is added automatically)** and score the outputs
```bash
R=results/<model>
python scripts/sweep_lambdas.py --method BoN --root $R --lambdas 0.0 0.6 2.0 \
    --calib_json $R/calib/lambda_star.json --eval_dir <scored candidates, N per prompt>
python scripts/sweep_lambdas.py --method ARGS --root $R --lambdas 0.0 0.6 2.0 --calib_json $R/calib/lambda_star.json \
    --gen_args "--sft_path <sft> --rm_path <rm> --cm_path <cm> --tokenizer <tok>" \
    --eval_args "--rm_path <rm> --cm_path <cm> --tokenizer <tok>"
```
λ values have no defaults: pass `--lambdas` (and `--calib_json` to include λ\*); `generate_args_pargs.py` / `generate_cdq.py` require `--lam`.
`--method` is `BoN`, `ARGS`, `PARGS` (partial RM/CM) or `CDQ` (`generate_cdq.py` arguments). Add `--rate` to also rate outputs with GPT (`rate_gpt.py`).

**6. Baselines** — Safe-RLHF (`scripts/ppo-lag.sh`), SafeDPO (`deepspeed --module safe_rlhf.algorithms.safedpo`), SFT. Score each baseline's completions with `scripts/eval_dual_calibration.py` and save them to `$R/baselines/<SFT|SafeDPO|Safe-RLHF>/`.

**7. Figures**
```bash
python scripts/make_figures.py --root $R --model_name llama3.2-3b --eval_mode gold    # -> $R/figures/
```
`--model_name` is `llama3.2-3b` or `qwen2.5-3b`. The individual steps are `collect_results.py`, `collect_gpt_ratings.py`, `make_prf_data.py` and `plots/*.py`.

### Results layout

```
$R/calib/lambda_star.json                      calibrate_dual_bisect.py
$R/BoN/lam-<λ>/, $R/BoN/summary.json           sweep_lambdas.py --method BoN
$R/{ARGS,PARGS,CDQ}/evaluations-lam-<λ>/       sweep_lambdas.py
$R/<METHOD>/gpt_rating-lam-<λ>/rate_*.json     sweep_lambdas.py --rate
$R/baselines/<NAME>/                           scored baseline datasets
```
λ\* is marked on a curve only if the method was run at λ\*.

## Citation

```bibtex
@article{chittepu2026lara,
  title   = {Safe Inference-Time Alignment via Lagrangian Reward Augmentation},
  author  = {Chittepu, Yaswanth and Joshi, Ativ and Chintala, Sohini and Niekum, Scott},
  journal = {arXiv preprint arXiv:2607.02781},
  year    = {2026}
}
```

## License

Apache-2.0 (see `LICENSE`); portions derived from PKU-Alignment/safe-rlhf.
