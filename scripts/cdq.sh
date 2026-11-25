#!/usr/bin/env bash

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-3B-Instruct"
REF_MODEL_NAME_OR_PATH=""
REWARD_MODEL_NAME_OR_PATH=""
OUTPUT_DIR="${ROOT_DIR}/output/cdq"
unset HOSTFILE
ZERO_STAGE=3
OFFLOAD="none"
LOG_RUN_NAME="default_cdq_run"
TRAIN_DATASETS=""
EVAL_DATASETS=""

while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model_name_or_path)
			MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--model_name_or_path=*)
			MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--ref_model_name_or_path)
			REF_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--ref_model_name_or_path=*)
			REF_MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--reward_model_name_or_path)
			REWARD_MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--reward_model_name_or_path=*)
			REWARD_MODEL_NAME_OR_PATH="${arg#*=}"
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
		--train_datasets)
			TRAIN_DATASETS="$1"
			shift
			;;
		--train_datasets=*)
			TRAIN_DATASETS="${arg#*=}"
			;;
		--eval_datasets)
			EVAL_DATASETS="$1"
			shift
			;;
		--eval_datasets=*)
			EVAL_DATASETS="${arg#*=}"
			;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

if [[ -z "${REWARD_MODEL_NAME_OR_PATH}" ]]; then
	echo "--reward_model_name_or_path is required." >&2
	exit 1
fi

if [[ -z "${TRAIN_DATASETS}" ]]; then
	echo "--train_datasets is required." >&2
	exit 1
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

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

deepspeed "${DEEPSPEED_ARGS[@]}" \
	--module safe_rlhf.values.cdq \
	--train_datasets "${TRAIN_DATASETS}" \
	${EVAL_DATASETS:+--eval_datasets "${EVAL_DATASETS}"} \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	${REF_MODEL_NAME_OR_PATH:+--ref_model_name_or_path "${REF_MODEL_NAME_OR_PATH}"} \
	--reward_model_name_or_path "${REWARD_MODEL_NAME_OR_PATH}" \
	--max_length 1024 \
	--trust_remote_code True \
	--epochs 2 \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 4 \
	--gradient_checkpointing \
	--lr 2e-5 \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.1 \
	--seed 42 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project Safe-RLHF-CDQ \
	--log_run_name "${LOG_RUN_NAME}" \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True \
	--save_interval 10000 \
	# --need_eval \
	# --eval_strategy epoch \
