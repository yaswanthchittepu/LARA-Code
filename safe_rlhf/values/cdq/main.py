"""Main training script for the CD-Q value network."""

from __future__ import annotations

import argparse

import deepspeed
import torch
import torch.distributed as dist
from transformers import SchedulerType
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available

from safe_rlhf.configs import get_deepspeed_eval_config, get_deepspeed_train_config
from safe_rlhf.datasets import parse_dataset
from safe_rlhf.logger import set_logger_level
from safe_rlhf.utils import seed_everything, str2bool
from safe_rlhf.values.cdq.trainer import CDQTrainer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='deepspeed --module safe_rlhf.values.cdq',
        description='Train a CD-Q value network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help='Checkpoint to initialise the value network backbone from.',
    )
    model_parser.add_argument(
        '--ref_model_name_or_path',
        type=str,
        default=None,
        help='Reference model pi_ref. Defaults to --model_name_or_path if omitted.',
    )
    model_parser.add_argument(
        '--reward_model_name_or_path',
        type=str,
        required=True,
        help='Frozen reward model used to score (prompt, response) pairs.',
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length.',
    )
    model_parser.add_argument(
        '--trust_remote_code',
        type=str2bool,
        default=False,
    )

    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--train_datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
        required=True,
    )
    dataset_parser.add_argument(
        '--eval_datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
    )

    # Training
    training_parser = parser.add_argument_group('training')
    training_parser.add_argument('--epochs', type=int, default=1)
    training_parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    training_parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    training_parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    training_parser.add_argument('--gradient_checkpointing', action='store_true')
    training_parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4)
    training_parser.add_argument(
        '--lr_scheduler_type',
        type=SchedulerType,
        default='cosine',
        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial',
                 'constant', 'constant_with_warmup'],
    )
    training_parser.add_argument('--lr_warmup_ratio', type=float, default=0.0)
    training_parser.add_argument('--weight_decay', type=float, default=1e-6)
    training_parser.add_argument('--seed', type=int, default=42)
    training_parser.add_argument('--fp16', type=str2bool, default=False)
    training_parser.add_argument('--bf16', type=str2bool, default=False)
    training_parser.add_argument('--tf32', type=str2bool, default=None)

    # Evaluation
    eval_parser = parser.add_argument_group('evaluation')
    eval_parser.add_argument('--eval_strategy', type=str, default='epoch',
                             choices=['epoch', 'steps'])
    eval_parser.add_argument('--eval_interval', type=int, default=1000000)
    eval_parser.add_argument('--need_eval', action='store_true', default=False)
    eval_parser.add_argument('--eval_split_ratio', type=float, default=None)

    # Logging
    logging_parser = parser.add_argument_group('logging')
    logging_parser.add_argument('--output_dir', type=str, default=None)
    logging_parser.add_argument('--log_type', type=str, default='wandb',
                                choices=['wandb', 'tensorboard'])
    logging_parser.add_argument('--log_dir', type=str, default=None)
    logging_parser.add_argument('--log_project', type=str, default=None)
    logging_parser.add_argument('--log_run_name', type=str, default=None)
    logging_parser.add_argument('--save_16bit', action='store_true')
    logging_parser.add_argument('--save_interval', type=int, default=1000000)

    # DeepSpeed
    deepspeed_parser = parser.add_argument_group('deepspeed')
    deepspeed_parser.add_argument('--local_rank', type=int, default=-1)
    deepspeed_parser.add_argument('--zero_stage', type=int, default=0,
                                  choices=[0, 1, 2, 3])
    deepspeed_parser.add_argument(
        '--offload', type=str, default='none',
        choices=['none', 'parameter', 'optimizer', 'all'],
    )
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    if args.local_rank == -1:
        parser.error('`local_rank` not set, please use DeepSpeed launcher.')
    if args.fp16 and args.bf16:
        parser.error('Cannot use both bf16 and fp16 precision.')
    if args.bf16 and not is_torch_bf16_gpu_available():
        parser.error('bf16 precision is not supported on this GPU.')
    if args.tf32 is not None and is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32

    return args


def main() -> None:
    args = parse_arguments()

    deepspeed.init_distributed()

    args.global_rank = dist.get_rank()
    args.device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(args.device)
    seed_everything(args.seed)
    set_logger_level()

    dist.barrier()

    ds_train_config = get_deepspeed_train_config(
        micro_batch_size_per_gpu=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        stage=args.zero_stage,
        offload=args.offload,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    ds_eval_config = get_deepspeed_eval_config(
        stage=args.zero_stage,
        offload=args.offload,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    trainer = CDQTrainer(args, ds_train_config, ds_eval_config)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    main()
