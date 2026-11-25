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
"""Trainer base class for supervised training."""

from __future__ import annotations

import abc
import os, sys
import argparse
from typing import Any, ClassVar

import deepspeed
import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.configs import ADAM_BETAS
from safe_rlhf.datasets import TokenizedDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers.base import TrainerBase
from safe_rlhf.utils import get_optimizer_grouped_parameters, is_main_process, to_device, get_all_reduce_mean


class SupervisedTrainer(TrainerBase):
    """Trainer base class for supervised training.

    Abstract methods:
        loss: Compute supervised training loss.
        train_step: Perform a single training step.
    """

    TRAINING_TYPE: ClassVar[str] = 'supervised'
    DATASET_TYPE: ClassVar[type[TokenizedDataset]]
    MODEL_TYPE = AutoModelForCausalLM

    model: deepspeed.DeepSpeedEngine
    ds_config: dict[str, Any]

    extra_model_kwargs: dict[str, Any] | None = None
    extra_tokenizer_kwargs: dict[str, Any] | None = None

    def __init__(self, args: argparse.Namespace, ds_config: dict[str, Any]) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_config = ds_config
        self.global_step = 0

        self.init_models()
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_config is not None and self.ds_config['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_config)

        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='right',
            auto_model_type=self.MODEL_TYPE,
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs=self.extra_model_kwargs,
            auto_tokenizer_kwargs=self.extra_tokenizer_kwargs,
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        
        train_dataset = self.DATASET_TYPE(
            self.args.train_datasets,
            tokenizer=self.tokenizer,
        )

        if self.args.need_eval:
            if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
                train_dataset, eval_dataset = train_dataset.split_train_test(
                    split_ratio=self.args.eval_split_ratio,
                )
            elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
                eval_dataset = self.DATASET_TYPE(
                    self.args.eval_datasets,
                    tokenizer=self.tokenizer,
                )
            else:
                raise ValueError('Either `eval_datasets` or `eval_split_ratio` should be provided.')

            self.eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=True),
                batch_size=self.args.per_device_eval_batch_size,
            )
        else:
            self.eval_dataloader = None
        
        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=self.args.per_device_train_batch_size,
        )

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.args.num_update_steps_per_epoch = (
            len(self.train_dataloader) + self.args.gradient_accumulation_steps - 1
        ) // self.args.gradient_accumulation_steps
        self.args.total_training_steps = self.args.epochs * self.args.num_update_steps_per_epoch

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            self.model,
            self.args.weight_decay,
        )
        if (
            self.ds_config['zero_optimization'].get('offload_optimizer', {}).get('device', 'none')
            != 'none'
        ):
            optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                betas=ADAM_BETAS,
            )
        else:
            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                betas=ADAM_BETAS,
            )

        num_warmup_steps = int(self.args.lr_warmup_ratio * self.args.total_training_steps)
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.args.total_training_steps,
        )

        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            args=self.args,
            config=self.ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
        )

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @abc.abstractmethod
    def loss(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute supervised training loss."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform a single training step."""
        raise NotImplementedError

    def eval(self):
        """Evaluate the Model"""
        with torch.no_grad():
            correct = 0
            total = 0
            for i, batch_ in tqdm(enumerate(self.eval_dataloader)):
                batch = to_device(batch_, self.args.device)
                batch_size = batch['better_input_ids'].size(0)
                sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
                    self.model.module,
                    input_ids=torch.cat([batch['better_input_ids'], batch['worse_input_ids']], dim=0),
                    attention_mask=torch.cat([batch['better_attention_mask'], batch['worse_attention_mask']], dim=0),
                )
                (
                    chosen_log_probs,  # size = (B, L - 1)
                    rejected_log_probs,  # size = (B, L - 1)
                ) = sequence_log_probs.chunk(chunks=2, dim=0)
                ref_sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
                    self.reference_model.module,
                    input_ids=torch.cat([batch['better_input_ids'], batch['worse_input_ids']], dim=0),
                    attention_mask=torch.cat([batch['better_attention_mask'], batch['worse_attention_mask']], dim=0),
                )
                (
                    ref_chosen_log_probs,  # size = (B, L - 1)
                    ref_rejected_log_probs,  # size = (B, L - 1)
                ) = ref_sequence_log_probs.chunk(chunks=2, dim=0)

                chosen_sample_log_ratios, rejected_sample_log_ratios = [], []
                for i in range(batch_size):
                    
                    chosen_end_index = batch['better_attention_mask'][i].nonzero()[-1].squeeze().item()
                    rejected_end_index = batch['worse_attention_mask'][i].nonzero()[-1].squeeze().item()


                    if(torch.all(torch.eq(batch['better_input_ids'][i], batch['worse_input_ids'][i]),).item()):
                        continue
                    
                    diverge_index = (
                        (batch['better_input_ids'][i] != batch['worse_input_ids'][i]).nonzero()[0].squeeze().item()
                    )
                    assert 0 <= diverge_index <= chosen_end_index, 'pref diverge index is out of range!'
                    assert 0 <= diverge_index <= rejected_end_index, 'pref diverge index is out of range!'
                    
                    chosen_seq_slice = slice(diverge_index, chosen_end_index + 1)
                    rejected_seq_slice = slice(diverge_index, rejected_end_index + 1)

                    # size = ()
                    chosen_log_prob_ = chosen_log_probs[i, chosen_seq_slice].sum(dim=-1)    
                    rejected_log_prob_ = rejected_log_probs[i, rejected_seq_slice].sum(dim=-1)
                    ref_chosen_log_prob_ = ref_chosen_log_probs[i, chosen_seq_slice].sum(dim=-1)
                    ref_rejected_log_prob_ = ref_rejected_log_probs[i, rejected_seq_slice].sum(dim=-1)
                    chosen_log_ratio = chosen_log_prob_ - ref_chosen_log_prob_
                    rejected_log_ratio = rejected_log_prob_ - ref_rejected_log_prob_
                    chosen_sample_log_ratios.append(chosen_log_ratio.detach())
                    rejected_sample_log_ratios.append(rejected_log_ratio.detach())
                
                chosen_sample_log_ratios_tensor = torch.stack(chosen_sample_log_ratios)  # size = (B,)
                rejected_sample_log_ratios_tensor = torch.stack(rejected_sample_log_ratios)
                correct += (chosen_sample_log_ratios_tensor > rejected_sample_log_ratios_tensor).float().sum()
                total += batch_size
            accuracy = (correct / total)
            accuracy = get_all_reduce_mean(accuracy)

        return {'test/accuracy': accuracy.item()}

    @torch.no_grad()
    def eval_margins(self, dir_name='',file_name='margins_eval.npz'):
        # return {}
        
        ##################################################################
        
        self.model.eval()
        pref_margins_res = None
        gt_margins_res = None
        
        for i, batch in enumerate(tqdm(self.eval_dataloader)):
            info = self.loss(**to_device(batch, self.args.device))

            # Prepare an empty tensor to hold gathered values
            pref_margins_ = [torch.zeros_like(info['reward_margin']) for _ in range(dist.get_world_size())]
            # Gather costs from all devices
            dist.all_gather(pref_margins_, info['reward_margin'])
            # Concatenate the gathered tensors
            pref_margins_all = torch.cat(pref_margins_, dim=0).to(self.args.device)

            gt = (batch['margin']).to(self.args.device)
            gt_margins_ = [torch.zeros_like(gt) for _ in range(dist.get_world_size())]
            # Gather costs from all devices
            dist.all_gather(gt_margins_, gt)
            # Concatenate the gathered tensors
            gt_margins_all = torch.cat(gt_margins_, dim=0).to(self.args.device)

            torch.cuda.empty_cache()
            if(pref_margins_res is None):
                pref_margins_res = pref_margins_all.flatten().detach().cpu()
            else:
                pref_margins_res = torch.cat((pref_margins_res, pref_margins_all.flatten().detach().cpu()))
            if(gt_margins_res is None):
                gt_margins_res = gt_margins_all.flatten().detach().cpu()
            else:
                gt_margins_res = torch.cat((gt_margins_res, gt_margins_all.flatten().detach().cpu()))
        
        pref_np = pref_margins_res.float().numpy()
        gt_np = gt_margins_res.float().numpy()
        # SAVE_DIR_NP = './margin_saves_pop2/llama3.1-8b'
        os.makedirs(dir_name, exist_ok=True)
        # SAVE_FILE = os.path.join(SAVE_DIR_NP, f"margins_eval_{name}.npz" if(name) else f"margins_eval.npz")
        np.savez(os.path.join(dir_name, file_name), pref=pref_np, gt=gt_np)
        
        #######################################################################

        # BINS=25

        # plt.figure(figsize=(8, 5))
        # plt.hist(pref_np, bins=BINS, alpha=0.6, label="Eval Margins", density=True)
        # plt.hist(gt_np, bins=BINS, alpha=0.6, label="GT Margins", density=True)

        # plt.title("Normalized Histogram of Margins")
        # plt.xlabel("Margin")
        # plt.ylabel("Density")
        # plt.legend()
        # # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f'hist_margins_eval_{name}.png' if(name) else f'hist_margins_eval.png')

        # fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        # # Preferred margins histogram
        # axs[0].hist(pref_np, bins=BINS, density=True, color='red', alpha=0.7)
        # axs[0].set_title("Preferred Margins")
        # axs[0].set_xlabel("Margin")
        # axs[0].set_ylabel("Density")
        # # axs[0].grid(True)

        # axs[1].hist(gt_np, bins=BINS, density=True, color='blue', alpha=0.7)
        # axs[1].set_title("GT Margins")
        # axs[1].set_xlabel("Margin")

        # plt.suptitle("Normalized Histograms of Margins", fontsize=14)
        # plt.tight_layout()
        # plt.savefig(f'hist_margins_subplot_eval_{name}.png' if(name) else f'hist_margins_subplot_eval.png')



    def train(self) -> None:
        """Train the model."""

        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.args.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.args.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        if self.args.need_eval:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.logger.log(self.eval(), step=0)

        for epoch in range(self.args.epochs):
            self.model.train()

            for batch in self.train_dataloader:
                info = self.train_step(**to_device(batch, self.args.device))
                torch.cuda.empty_cache()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.args.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)

                if self.global_step % self.args.save_interval == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    self.model.save_checkpoint(self.args.output_dir, tag=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.args.need_eval
                    and self.args.eval_strategy == 'steps'
                    and self.global_step % self.args.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)

            if self.args.need_eval and self.args.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)

            self.model.tput_timer.update_epoch_count()

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for model."""
        if mode:
            self.model.train()
            if self.args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.model.eval()
            if self.args.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()
