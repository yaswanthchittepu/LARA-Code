#!/usr/bin/env bash
#
# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

ACTOR_MODEL_NAME_OR_PATH="PKU-Alignment/alpaca-7b-reproduced"
REWARD_MODEL_NAME_OR_PATH="${ROOT_DIR}/output/rm"
COST_MODEL_NAME_OR_PATH="${ROOT_DIR}/output/cm"
unset {REWARD,COST}_CRITIC_MODEL_NAME_OR_PATH
OUTPUT_DIR="${ROOT_DIR}/output/ppo-lag"
unset HOSTFILE
ZERO_STAGE=3
OFFLOAD="none"
LOG_RUN_NAME="ppo-lag-1gpu"
THRESHOLD=0.
CVAR_ALPHA=
CVAR_WINDOW=0
SEED=42
ACTOR_LR=5e-6
TRAIN_DATASETS="PKU-SafeRLHF/train:0.15"
USE_PTX=True
SAVE_INTERVAL=300
KEEP_LAST_CKPT=False
while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--actor_model_name_or_path)
			ACTOR_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--actor_model_name_or_path=*)
			ACTOR_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--reward_model_name_or_path)
			REWARD_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--reward_model_name_or_path=*)
			REWARD_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--reward_critic_model_name_or_path)
			REWARD_CRITIC_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--reward_critic_model_name_or_path=*)
			REWARD_CRITIC_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--cost_model_name_or_path)
			COST_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--cost_model_name_or_path=*)
			COST_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--cost_critic_model_name_or_path)
			COST_CRITIC_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--cost_critic_model_name_or_path=*)
			COST_CRITIC_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--output_dir)
			OUTPUT_DIR="$1"
			shift
			;;
		--output_dir=*)
			OUTPUT_DIR="${arg#*=}"
			;;
		--hostfile)
			HOSTFILE="$1"
			shift
			;;
		--hostfile=*)
			HOSTFILE="${arg#*=}"
			;;
		--zero_stage)
			ZERO_STAGE="$1"
			shift
			;;
		--zero_stage=*)
			ZERO_STAGE="${arg#*=}"
			;;
		--offload)
			OFFLOAD="$1"
			shift
			;;
		--offload=*)
			OFFLOAD="${arg#*=}"
			;;
		--log_run_name)
			LOG_RUN_NAME="$1"
			shift
			;;
		--log_run_name=*)
			LOG_RUN_NAME="${arg#*=}"
			;;
		--threshold)
			THRESHOLD="$1"
			shift
			;;
		--threshold=*)
			THRESHOLD="${arg#*=}"
			;;
		--cvar_alpha)
			CVAR_ALPHA="$1"
			shift
			;;
		--cvar_alpha=*)
			CVAR_ALPHA="${arg#*=}"
			;;
		--cvar_window)
			CVAR_WINDOW="$1"
			shift
			;;
		--cvar_window=*)
			CVAR_WINDOW="${arg#*=}"
			;;
		--seed)
			SEED="$1"
			shift
			;;
		--seed=*)
			SEED="${arg#*=}"
			;;
		--actor_lr)
			ACTOR_LR="$1"
			shift
			;;
		--actor_lr=*)
			ACTOR_LR="${arg#*=}"
			;;
		--train_datasets)
			TRAIN_DATASETS="$1"
			shift
			;;
		--train_datasets=*)
			TRAIN_DATASETS="${arg#*=}"
			;;
		--use_ptx)
			USE_PTX="$1"
			shift
			;;
		--use_ptx=*)
			USE_PTX="${arg#*=}"
			;;
		--save_interval)
			SAVE_INTERVAL="$1"
			shift
			;;
		--save_interval=*)
			SAVE_INTERVAL="${arg#*=}"
			;;
		--keep_last_checkpoint_only)
			KEEP_LAST_CKPT="$1"
			shift
			;;
		--keep_last_checkpoint_only=*)
			KEEP_LAST_CKPT="${arg#*=}"
			;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

if [[ -z "${REWARD_CRITIC_MODEL_NAME_OR_PATH+x}" ]]; then
	REWARD_CRITIC_MODEL_NAME_OR_PATH="${REWARD_MODEL_NAME_OR_PATH}"
fi
if [[ -z "${COST_CRITIC_MODEL_NAME_OR_PATH+x}" ]]; then
	COST_CRITIC_MODEL_NAME_OR_PATH="${COST_MODEL_NAME_OR_PATH}"
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

if [[ -z "${WANDB_API_KEY}" ]]; then
	export WANDB_MODE="offline"
fi

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

DEEPSPEED_ARGS=()
if [[ -n "${HOSTFILE+x}" ]]; then
	DEEPSPEED_ARGS+=("--hostfile" "${HOSTFILE}")
fi
DEEPSPEED_ARGS+=("--master_port" "${MASTER_PORT}")

# --ptx_datasets is only added when USE_PTX=True: rl_trainer.py gates the entire extra pretraining-mix
# forward/backward pass on `ptx_datasets is not None`, so omitting the flag (not just zeroing --ptx_coeff)
# both disables the loss term and skips the wasted compute.
CVAR_ARGS=()
if [[ -n "${CVAR_ALPHA}" ]]; then
	CVAR_ARGS+=("--cvar_alpha" "${CVAR_ALPHA}" "--cvar_window" "${CVAR_WINDOW}")
fi

PTX_ARGS=()
if [[ "${USE_PTX}" == "True" ]]; then
	PTX_ARGS+=("--ptx_datasets" "alpaca" "--ptx_coeff" "8.0")
fi

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

deepspeed "${DEEPSPEED_ARGS[@]}" \
	--master_port "${MASTER_PORT}" \
	--module safe_rlhf.algorithms.ppo_lag \
	--train_datasets "${TRAIN_DATASETS}" \
	"${PTX_ARGS[@]}" \
	--actor_model_name_or_path "${ACTOR_MODEL_NAME_OR_PATH}" \
	--reward_model_name_or_path "${REWARD_MODEL_NAME_OR_PATH}" \
	--reward_critic_model_name_or_path "${REWARD_CRITIC_MODEL_NAME_OR_PATH}" \
	--cost_model_name_or_path "${COST_MODEL_NAME_OR_PATH}" \
	--cost_critic_model_name_or_path "${COST_CRITIC_MODEL_NAME_OR_PATH}" \
	--max_length 512 \
	--temperature 1.2 \
	--num_return_sequences 1 \
	--repetition_penalty 1.2 \
	--trust_remote_code True \
	--epochs 1 \
	--update_iters 1 \
	--per_device_prompt_batch_size 16 \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 2 \
	--actor_lr "${ACTOR_LR}" \
	--actor_weight_decay 0 \
	--actor_lr_scheduler_type cosine \
	--actor_lr_warmup_ratio 0.03 \
	--actor_gradient_checkpointing \
	--critic_lr 5e-6 \
	--critic_weight_decay 0.1 \
	--critic_lr_scheduler_type constant \
	--critic_lr_warmup_ratio 0.03 \
	--critic_gradient_checkpointing \
	--normalize_reward False \
	--normalize_cost False \
	--seed "${SEED}" \
	--threshold "${THRESHOLD}" \
	"${CVAR_ARGS[@]}" \
	--lambda_init 1.0 \
	--lambda_lr 0.01 \
	--lambda_max 100.0 \
	--lambda_update_delay_steps 0 \
	--episode_cost_window_size 128 \
	--kl_coeff 0.1 \
	--clip_range_ratio 0.1 \
	--clip_range_score 50.0 \
	--clip_range_value 5.0 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project Safe-RLHF-PPO \
	--log_run_name "${LOG_RUN_NAME}" \
	--save_interval "${SAVE_INTERVAL}" \
	--keep_last_checkpoint_only "${KEEP_LAST_CKPT}" \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True \
	# --eval_strategy steps \
	# --eval_interval 75 \
	# --need_eval \
	# --eval_split_ratio 0.1 \
	# --eval_datasets safety_test/val \
	# --eval_datasets PKU-SafeRLHF/train \
