"""CD-Q Value Network Trainer.

Trains a token-level value function V_theta using the Bellman MSE objective.
Architecture follows FARMA: value head outputs vocab_size logits per position,
enabling efficient Bellman targets via a single dot product with pi_ref probs.

Training data: (prompt, response) pairs scored by a frozen reward model.

Loss at response position t:
    pred   = V_theta(x_t | x[:t-1])          -- value at actual token taken
    target = E_{z~pi_ref}[V_theta(z | x[:t])] -- Bellman bootstrap (stop-grad)
           = reward                            -- at EOS position
    L_t = 0.5 * (pred - sg(target))^2
"""

from __future__ import annotations

import argparse
from typing import Any
import sys

import deepspeed
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.configs import IGNORE_INDEX
from safe_rlhf.datasets import SupervisedDataset
from safe_rlhf.models import AutoModelForScore, AutoModelForValue
from safe_rlhf.models.score_model import ScoreModelOutput
from safe_rlhf.models.value_model import ValueModelOutput
from safe_rlhf.trainers import SupervisedTrainer
from safe_rlhf.utils import get_all_reduce_mean, is_main_process, to_device
from safe_rlhf.models import load_pretrained_models


class CDQTrainer(SupervisedTrainer):
    """Trainer for the CD-Q value network."""

    TRAINING_TYPE = 'cdq'
    DATASET_TYPE = SupervisedDataset
    MODEL_TYPE = AutoModelForValue

    model: deepspeed.DeepSpeedEngine          # V_theta (trainable)
    ref_model: deepspeed.DeepSpeedEngine      # pi_ref  (frozen CausalLM)
    reward_model: deepspeed.DeepSpeedEngine   # r(x,y)  (frozen ScoreModel)

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        super().__init__(args, ds_train_config)

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def init_models(self) -> None:
        """Load value model (trainable), reference CausalLM, and reward model."""
        if (
            self.ds_train_config is not None
            and self.ds_train_config['zero_optimization']['stage'] == 3
        ):
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_config)

        if (
            self.ds_eval_config is not None
            and self.ds_eval_config['zero_optimization']['stage'] == 3
        ):
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_config)

        # Value network V_theta — trained from scratch (new head) on top of
        # a pretrained LM backbone.  Initialised from model_name_or_path.
        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='right',
            auto_model_type=AutoModelForValue,
            trust_remote_code=self.args.trust_remote_code,
        )

        # Reference model pi_ref (frozen).  Defaults to same checkpoint as
        # value model unless --ref_model_name_or_path is given.
        ref_path = getattr(self.args, 'ref_model_name_or_path', None) or self.args.model_name_or_path
        self.ref_model, _ = load_pretrained_models(
            ref_path,
            model_max_length=self.args.max_length,
            padding_side='right',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )

        # Reward model r(x,y) (frozen).
        self.reward_model, _ = load_pretrained_models(
            self.args.reward_model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='right',
            auto_model_type=AutoModelForScore,
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs={'score_type': 'reward', 'do_normalize': False},
        )

    # ------------------------------------------------------------------
    # Engine initialisation
    # ------------------------------------------------------------------

    def init_engines(self) -> None:
        """DeepSpeed: value model trains; ref/reward models are eval-only."""
        # SupervisedTrainer.init_engines() wraps self.model with a training engine.
        super().init_engines()

        self.ref_model, *_ = deepspeed.initialize(
            model=self.ref_model,
            config=self.ds_eval_config,
        )
        self.reward_model, *_ = deepspeed.initialize(
            model=self.reward_model,
            config=self.ds_eval_config,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        input_ids: torch.LongTensor,       # (B, L)
        attention_mask: torch.BoolTensor,  # (B, L)
        labels: torch.LongTensor,          # (B, L)  — IGNORE_INDEX for prompt/pad
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Bellman MSE loss for the CD-Q value network.

        For each response position t in [1, L-1]:

            pred_t   = V_theta(x_t | x[:t-1])
                     = values[:, t-1, input_ids[:, t]]

            target_t = E_{z ~ pi_ref(·|x[:t])} [sg(V_theta(z | x[:t]))]
                     = (ref_probs[:, t, :] * sg(values[:, t, :])).sum(-1)

            At EOS:  target_t = r(x, y)   (from reward model)

            loss     = mean_t 0.5 * (pred_t - sg(target_t))^2
        """
        B, L = input_ids.size()

        # ---- frozen passes ----
        with torch.no_grad():
            # Reward model: scalar reward + last-token index for each sequence
            reward_out: ScoreModelOutput = self.reward_model(
                input_ids,
                attention_mask=attention_mask,
            )
            rewards = reward_out.end_scores.squeeze(-1).float()  # (B,)
            end_index = reward_out.end_index                     # (B,)

            # Reference model: next-token probabilities at every position
            ref_logits = self.ref_model(
                input_ids,
                attention_mask=attention_mask,
            ).logits.float()                                     # (B, L, vocab_size)
            ref_probs = F.softmax(ref_logits, dim=-1)            # (B, L, vocab_size)

        # ---- trainable value forward pass ----
        value_out: ValueModelOutput = self.model(
            input_ids,
            attention_mask=attention_mask,
        )
        values = value_out.values  # (B, L, vocab_size)

        # ---- vectorised Bellman targets ----
        # pred_all[:, i] = V(x_{i+1} | x[:i])  for i in 0..L-2
        pred_all = values[:, :-1, :].gather(
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)                                            # (B, L-1)

        # bellman_bootstrap[:, i] = E_{z~pi_ref(·|x[:i+1])}[sg(V(z|x[:i+1]))]
        bellman_bootstrap = (
            ref_probs[:, 1:, :] * values[:, 1:, :].detach()
        ).sum(-1)                                                # (B, L-1)

        # Start from bootstrap targets; override EOS positions with reward
        targets = bellman_bootstrap                              # (B, L-1)
        for b in range(B):
            eos = end_index[b].item()
            if eos >= 1:
                # position eos in input_ids → index eos-1 in the L-1 view
                targets = targets.clone()
                targets[b, eos - 1] = rewards[b]

        # Mask: only loss on response tokens (labels[:, 1:] != IGNORE_INDEX)
        response_mask = (labels[:, 1:] != IGNORE_INDEX).float()  # (B, L-1)

        loss_per_token = 0.5 * (pred_all - targets.detach()) ** 2  # (B, L-1)
        loss = (loss_per_token * response_mask).sum() / response_mask.sum().clamp(min=1)

        return {
            'loss': loss,
        }

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def train_step(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.LongTensor,
        **kwargs,
    ) -> dict[str, Any]:
        loss_dict = self.loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        loss = get_all_reduce_mean(loss)

        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        if self.eval_dataloader is None:
            return {}

        self.set_eval()
        total_loss = torch.tensor(0.0, device=self.args.device)
        n_batches = 0

        for batch in self.eval_dataloader:
            batch = to_device(batch, self.args.device)
            loss_dict = self.loss(**batch)
            total_loss += loss_dict['loss']
            n_batches += 1

        if n_batches == 0:
            self.set_train()
            return {}

        avg_loss = get_all_reduce_mean(total_loss / n_batches)
        self.set_train()
        return {'eval/loss': avg_loss.item()}
