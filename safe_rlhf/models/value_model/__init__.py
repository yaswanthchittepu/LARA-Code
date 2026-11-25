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
"""Auto-models for value models (CD-Q value network)."""

from __future__ import annotations

import functools
import importlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import transformers.models.auto as auto_module
from transformers import PretrainedConfig
from transformers.models.auto.auto_factory import (
    _BaseAutoModelClass,
    _LazyAutoMapping,
    auto_class_update,
    getattribute_from_module,
)
from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING_NAMES,
    model_type_to_module_name,
)
from transformers.utils.generic import ModelOutput


class _LazyAutoMappingInSafeRLHF(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type: str, attr: str) -> Any:
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f'.{module_name}',
                'safe_rlhf.models.value_model',
            )
        return getattribute_from_module(self._modules[module_name], attr)


MODEL_FOR_VALUE_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        ('llama', 'LlamaForValue'),
        ('qwen2', 'Qwen2ForValue'),
    ],
)
MODEL_FOR_VALUE_MAPPING: OrderedDict[str, Any] = _LazyAutoMappingInSafeRLHF(
    CONFIG_MAPPING_NAMES,
    MODEL_FOR_VALUE_MAPPING_NAMES,
)


@functools.partial(auto_class_update, head_doc='value model')
class AutoModelForValue(_BaseAutoModelClass):
    _model_mapping: OrderedDict[str, Any] = MODEL_FOR_VALUE_MAPPING


setattr(auto_module, 'MODEL_FOR_VALUE_MAPPING', MODEL_FOR_VALUE_MAPPING)  # noqa: B010
setattr(auto_module, AutoModelForValue.__name__, AutoModelForValue)


@dataclass
class ValueModelOutput(ModelOutput):
    """
    Output of the value model.

    Args:
        values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, vocab_size)`):
            Per-token value estimates V_theta(z | x, y[:i]) for all z in the vocabulary.
            At position i, values[:, i, :] gives V(z | x, y[:i]) for all z.
        end_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, vocab_size)`):
            Value estimates at the last non-padding position.
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_dim)`):
            Sequence of hidden-states at the output of the last layer of the model.
        end_index (`torch.LongTensor` of shape `(batch_size,)`):
            Indices of the last non-padding token.
    """

    values: torch.FloatTensor | None = None           # (B, L, vocab_size)
    end_values: torch.FloatTensor | None = None       # (B, vocab_size)
    last_hidden_state: torch.FloatTensor | None = None  # (B, L, E)
    end_index: torch.LongTensor | None = None         # (B,)


class ValueModelMixin:
    """
    Mixin for value models used in CD-Q.

    The value head projects hidden states to a vocab-sized output so that a single
    forward pass on prefix [x, y[:i]] yields V_theta(z | x, y[:i]) for all z in D.
    This enables efficient Bellman target computation via a dot product with pi_ref
    probabilities, without querying V for each vocabulary token separately.
    """

    value_head: nn.Linear
    _is_value_head_initialized: bool = False

    def init_value_head(self, config: PretrainedConfig, hidden_size: int, **kwargs: Any) -> None:
        """Initialize the vocab-sized value head."""
        if self._is_value_head_initialized:
            return

        vocab_size = getattr(config, 'vocab_size', None)
        if vocab_size is None:
            raise ValueError('config must have a vocab_size attribute to initialize the value head.')

        # bias=False mirrors standard LM head convention
        self.value_head = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)

        self._is_value_head_initialized = True

    def get_values(
        self,
        last_hidden_state: torch.FloatTensor,       # (B, L, E)
        attention_mask: torch.BoolTensor | None = None,  # (B, L)
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | ValueModelOutput:
        """
        Compute per-token value estimates over the full vocabulary.

        Returns values of shape (B, L, vocab_size) where values[:, i, :] =
        V_theta(z | x, y[:i]) for all vocabulary tokens z.
        """
        B, L, E = last_hidden_state.size()

        if attention_mask is None:
            if B > 1:
                raise ValueError("'attention_mask' is required when batch size > 1.")
            attention_mask = last_hidden_state.new_ones(B, L, dtype=torch.bool)

        values = self.value_head(last_hidden_state).float()  # (B, L, vocab_size)

        end_index = torch.cat([m.nonzero()[-1] for m in attention_mask])  # (B,)
        end_values = torch.gather(  # (B, 1, vocab_size)
            values,
            dim=1,
            index=(
                end_index.to(values.device)
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .expand(-1, -1, values.size(-1))
            ),
        ).squeeze(dim=1)  # (B, vocab_size)

        if not return_dict:
            return values, end_values

        return ValueModelOutput(
            values=values,                          # (B, L, vocab_size)
            end_values=end_values,                  # (B, vocab_size)
            last_hidden_state=last_hidden_state,    # (B, L, E)
            end_index=end_index,                    # (B,)
        )
