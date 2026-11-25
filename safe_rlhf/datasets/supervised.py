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

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch

from safe_rlhf.configs import IGNORE_INDEX, PROMPT_ASSISTANT, PROMPT_BEGIN, PROMPT_USER
from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding


__all__ = [
    'SupervisedDataset',
    'SupervisedCollator',
    'SupervisedSample',
    'SupervisedBatch',
]


class SupervisedSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)


class SupervisedBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)

class SupervisedDataset_GSM8K(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> SupervisedSample:
        if raw_sample.get('input') is None and raw_sample.get('dialogue') is None:
            raise ValueError('Either `input` or `dialogue` must be provided.')
        if raw_sample.get('input') is not None and raw_sample.get('dialogue') is not None:
            raise ValueError('At most one of `input` and `dialogue` can be provided.')

        if raw_sample.get('input') is not None:
            input = raw_sample['input']  # pylint: disable=redefined-builtin
            if not isinstance(input, str):
                raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str.')

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
            prompt = prefix_few_shot+f"\nQ: {input} \n A: "
            text = prompt + raw_sample['answer']+ self.tokenizer.eos_token

            input_ids = self.tokenize(text)
            labels = input_ids.clone()
            # Mask non-assistant input
            labels[: len(self.tokenize(prompt))] = IGNORE_INDEX
            if(raw_sample.get('reward') is not None):
                return {'input_ids': input_ids, 'labels': labels, 'reward': raw_sample.get('reward')}
            else:
                return {'input_ids': input_ids, 'labels': labels}

        dialogue = raw_sample['dialogue']  # is not None
        text = PROMPT_BEGIN
        offsets = [0]
        input_ids = torch.empty(0, dtype=torch.long)
        for i, line in enumerate(dialogue):
            if i % 2 == 0:
                # User input
                text += PROMPT_USER.format(input=line) + PROMPT_ASSISTANT
            else:
                # Assistant input
                text += line + self.tokenizer.eos_token
            input_ids = self.tokenize(text)
            offsets.append(len(input_ids))

        labels = input_ids.clone()
        # Mask non-assistant input
        for begin, end in zip(offsets[::2], offsets[1::2]):
            labels[begin:end] = IGNORE_INDEX

        if(raw_sample.get('reward') is not None):
            return {
                'input_ids': input_ids,  # size = (L,)
                'labels': labels,  # size = (L,)
                'reward': raw_sample.get('reward')
            }
        else:
            return {
                'input_ids': input_ids,  # size = (L,)
                'labels': labels,  # size = (L,)
            }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator_GSM8K(self.tokenizer.pad_token_id)

class SupervisedCollator_GSM8K(CollatorBase):
    def __call__(self, samples: list[SupervisedSample]) -> SupervisedBatch:
        input_ids = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        )
        labels = right_padding(
            [sample['labels'] for sample in samples],
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids.ne(self.pad_token_id)
        rewards = torch.tensor([sample['reward'] for sample in samples])
        return {
            'input_ids': input_ids,  # size = (B, L)
            'labels': labels,  # size = (B, L)
            'attention_mask': attention_mask,  # size = (B, L)
            'reward': rewards
        }

class SupervisedDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> SupervisedSample:
        if raw_sample.get('input') is None and raw_sample.get('dialogue') is None:
            raise ValueError('Either `input` or `dialogue` must be provided.')
        if raw_sample.get('input') is not None and raw_sample.get('dialogue') is not None:
            raise ValueError('At most one of `input` and `dialogue` can be provided.')

        if raw_sample.get('input') is not None:
            input = raw_sample['input']  # pylint: disable=redefined-builtin
            if not isinstance(input, str):
                raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str.')

            prompt = format_prompt(input=input, eos_token=self.tokenizer.eos_token)
            answer = raw_sample['answer']
            text = prompt + answer + self.tokenizer.eos_token

            input_ids = self.tokenize(text)
            labels = input_ids.clone()
            # Mask non-assistant input
            labels[: len(self.tokenize(prompt))] = IGNORE_INDEX
            return {'input_ids': input_ids, 'labels': labels}

        dialogue = raw_sample['dialogue']  # is not None
        text = PROMPT_BEGIN
        offsets = [0]
        input_ids = torch.empty(0, dtype=torch.long)
        for i, line in enumerate(dialogue):
            if i % 2 == 0:
                # User input
                text += PROMPT_USER.format(input=line) + PROMPT_ASSISTANT
            else:
                # Assistant input
                text += line + self.tokenizer.eos_token
            input_ids = self.tokenize(text)
            offsets.append(len(input_ids))

        labels = input_ids.clone()
        # Mask non-assistant input
        for begin, end in zip(offsets[::2], offsets[1::2]):
            labels[begin:end] = IGNORE_INDEX

        return {
            'input_ids': input_ids,  # size = (L,)
            'labels': labels,  # size = (L,)
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id)


class SupervisedCollator(CollatorBase):
    def __call__(self, samples: list[SupervisedSample]) -> SupervisedBatch:
        input_ids = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        )
        labels = right_padding(
            [sample['labels'] for sample in samples],
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids.ne(self.pad_token_id)
        return {
            'input_ids': input_ids,  # size = (B, L)
            'labels': labels,  # size = (B, L)
            'attention_mask': attention_mask,  # size = (B, L)
        }
