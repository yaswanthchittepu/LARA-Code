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

from __future__ import annotations

from typing import Callable, Hashable
from typing_extensions import TypedDict  # Python 3.10+

import torch
from torch.utils.data import Dataset, Subset

from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, left_padding


__all__ = [
    'PromptOnlyDataset',
    'PromptOnlyCollator',
    'PromptOnlySample',
    'PromptOnlyBatch',
]


class PromptOnlySample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)


class PromptOnlyBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)


class PromptOnlyDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> PromptOnlySample:
        prompt = format_prompt(input=raw_sample['input'], eos_token=self.tokenizer.eos_token)
        input_ids = self.tokenize(prompt)
        return {
            'input_ids': input_ids,  # size = (L,)
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PromptOnlyCollator(self.tokenizer.pad_token_id)

    def _merge_raw_datasets(self, seed: int | None = None) -> Dataset[RawSample]:
        """Merge multiple raw datasets into one dataset and remove duplicates."""

        def to_hashable(raw_sample: RawSample) -> Hashable:
            input = raw_sample['input']  # pylint: disable=redefined-builtin
            return input if isinstance(input, str) else tuple(input)

        merged = super()._merge_raw_datasets(seed)
        inputs = {to_hashable(merged[i]): i for i in range(len(merged))}
        return Subset(merged, sorted(inputs.values()))


class PromptOnlyCollator(CollatorBase):
    def __call__(self, samples: list[PromptOnlySample]) -> PromptOnlyBatch:
        input_ids = [sample['input_ids'] for sample in samples]
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]

        input_ids = left_padding(input_ids, padding_value=self.pad_token_id)
        attention_mask = left_padding(attention_mask, padding_value=0)
        return {
            'input_ids': input_ids,  # size = (B, L)
            'attention_mask': attention_mask,  # size = (B, L)
        }


class PromptOnlyDataset_GSM8K(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> PromptOnlySample:
        # prompt = format_prompt(input=raw_sample['input'], eos_token=self.tokenizer.eos_token)
        prefix_few_shot = """
                Given below are a few exemplars. Please answer the Question and the Question only. End your answer with: 'The answer is <>' \n
                Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
                A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.
                Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
                A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
                Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
                A: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.
                Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
                A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.
                Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
                A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.
                Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
                A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.
                Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
                A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.
                Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
                A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.
            """
        prompt = prefix_few_shot+f"\nQ: {raw_sample['input']} \n A: "
            
        input_ids = self.tokenize(prompt)
        answer_tokenized = self.tokenize(raw_sample['answer'])
        return {
            'input_ids': input_ids,  # size = (L,)
            'answer_ids': answer_tokenized,
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PromptOnlyCollator_GSM8K(self.tokenizer.pad_token_id)

    def _merge_raw_datasets(self, seed: int | None = None) -> Dataset[RawSample]:
        """Merge multiple raw datasets into one dataset and remove duplicates."""

        def to_hashable(raw_sample: RawSample) -> Hashable:
            input = raw_sample['input']  # pylint: disable=redefined-builtin
            return input if isinstance(input, str) else tuple(input)

        merged = super()._merge_raw_datasets(seed)
        inputs = {to_hashable(merged[i]): i for i in range(len(merged))}
        return Subset(merged, sorted(inputs.values()))


class PromptOnlyCollator_GSM8K(CollatorBase):
    def __call__(self, samples: list[PromptOnlySample]) -> PromptOnlyBatch:
        input_ids = [sample['input_ids'] for sample in samples]
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]
        answer_ids = [sample['answer_ids'] for sample in samples]

        input_ids = left_padding(input_ids, padding_value=self.pad_token_id)
        attention_mask = left_padding(attention_mask, padding_value=0)
        answer_ids = left_padding(answer_ids, padding_value=self.pad_token_id)
        return {
            'input_ids': input_ids,  # size = (B, L)
            'attention_mask': attention_mask,  # size = (B, L)
            'answer_ids': answer_ids,
        }
