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
"""Safe-RLHF preference datasets."""

from __future__ import annotations

from typing import ClassVar

import random
from datasets import Dataset, load_dataset, load_from_disk
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = [
    'SafeRLHFDataset',
    'SafeDPODataset',
    'SafeRLHFTrainDataset',
    'SafeRLHFTestDataset',
    'SafeRLHF30KTrainDataset',
    'SafeRLHF30KTestDataset',
    'SafeRLHF10KTrainDataset',
    'SafeRLHFPartialDataset',
    'SafeRLHFPartialTrainDataset',
    'SafeRLHFPartialTestDataset',
    'SafeRLHFSupervisedDataset',
    'SafeRLHFSupervisedTrainDataset',
    'SafeRLHFSupervisedTestDataset',
]

class SafeRLHFDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT)

        # For train
        # seed_value = random.randint(0, 1000000)
        # downsample_size = 1000
        # old_size = self.data.num_rows
        # print(f"Shuffling the dataset with seed: {seed_value}")
        # self.data = self.data.shuffle(seed=seed_value)
        # self.data = self.data.select(range(downsample_size))
        # print(f"Truncated the Dataset from size {old_size} to {self.data.num_rows}")

        # For Eval
        # downsample_size = 100
        # old_size = self.data.num_rows
        # print(f"Shuffling the dataset with seed: {42}")
        # self.data = self.data.shuffle(seed=42)
        # self.data = self.data.select(range(downsample_size))
        # print(f"Truncated the Dataset from size {old_size} to {self.data.num_rows}")

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'],
            answer=data['response_0'],
            other_answer=data['response_1'],
            better=int(data['better_response_id']) == 0,
            safer=int(data['safer_response_id']) == 0,
            is_safe=bool(data['is_response_0_safe']),
            is_other_safe=bool(data['is_response_1_safe']),
        )

    def __len__(self) -> int:
        return len(self.data)


class SafeRLHFTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'train'


class SafeRLHFTestDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF/test',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'test'


class SafeRLHF30KTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-30K/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-30K/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-30K'
    SPLIT: str = 'train'


class SafeRLHF30KTestDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-30K/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-30K/test',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-30K'
    SPLIT: str = 'test'


class SafeRLHF10KTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-10K/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-10K/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-10K'
    SPLIT: str = 'train'

    # def __init__(self, path: str | None = None) -> None:
    #     self.data = load_dataset(path or self.PATH, split=self.SPLIT)
    #     self.data = self.data.shuffle(seed=42)
    #     self.data = self.data.select(range(100))

class SafeDPODataset(SafeRLHFDataset):
    NAME: str = 'SafeDPO/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF/SafeDPO/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        super().__init__(path)
        self.data = self.data.filter(
            lambda x: bool(x['is_response_0_safe']) or bool(x['is_response_1_safe'])
        )

    def __getitem__(self, index: int) -> RawSample:
        
        data = self.data[index]
        is_response_0_safe = bool(data['is_response_0_safe'])
        is_response_1_safe = bool(data['is_response_1_safe'])

        # If one is safe and the other isn't, safety overrides helpfulness preference
        if is_response_0_safe and not is_response_1_safe:
            better = True   # response_0 is safe, response_1 is not → prefer response_0
        elif is_response_1_safe and not is_response_0_safe:
            better = False  # response_1 is safe, response_0 is not → prefer response_1
        else:
            # Both safe: fall back to helpfulness label
            better = int(data['better_response_id']) == 0

        # better = True => First responses is better in our augmented dataset
        # better = False => Second response is better in our augmented dataset
        # safer is useless, just a placeholder
        # is_safe and is_other_safe can be used for margin term. If no margin, these are also useless
        # For margin if is_safe == true and is_other_safe == true, set margin to 0
        # if is_safe xor is_other_safe = true, set margin to delta
        return RawSample(
            input=data['prompt'],
            answer=data['response_0'],
            other_answer=data['response_1'],
            better=better,
            safer=int(data['safer_response_id']) == 0,
            is_safe=is_response_0_safe,
            is_other_safe=is_response_1_safe,
        )


class SafeRLHFPartialDataset(RawDataset):
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_from_disk(path or self.PATH)
        self.data = self.data.filter(lambda x: x['response_0'] != x['response_1'])

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'],
            answer=data['response_0'],
            other_answer=data['response_1'],
            better=int(data['better_response_id']) == 0,
            safer=int(data['safer_response_id']) == 0,
            is_safe=bool(data['is_response_0_safe']),
            is_other_safe=bool(data['is_response_1_safe']),
        )

    def __len__(self) -> int:
        return len(self.data)


class SafeRLHFPartialTrainDataset(SafeRLHFPartialDataset):
    NAME: str = 'PKU-SafeRLHF-Partial/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-Partial/train',)
    PATH: str = './data/safe_rlhf_partial/train'


class SafeRLHFPartialTestDataset(SafeRLHFPartialDataset):
    NAME: str = 'PKU-SafeRLHF-Partial/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-Partial/test',)
    PATH: str = './data/safe_rlhf_partial/test'


class SafeRLHFSupervisedDataset(RawDataset):
    """SafeRLHF preference data expanded into supervised (prompt, response) pairs.

    Each preference pair (prompt, response_0, response_1) yields two datapoints:
        index 2*i   → (prompt, response_0)
        index 2*i+1 → (prompt, response_1)

    Used to train the CD-Q value network where both responses are treated as
    independent supervised samples to be scored by the reward model at training time.
    """

    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        raw = load_dataset(path or self.PATH, split=self.SPLIT)
        # Unpack each preference pair into two (prompt, response) rows,
        # then shuffle so the two responses from the same pair are not adjacent.
        prompts, responses = [], []
        for item in raw:
            prompts.append(item['prompt'])
            responses.append(item['response_0'])
            prompts.append(item['prompt'])
            responses.append(item['response_1'])
        self.data = Dataset.from_dict({'prompt': prompts, 'response': responses})
        self.data = self.data.shuffle(seed=42)

    def __getitem__(self, index: int) -> RawSample:
        item = self.data[index]
        return RawSample(input=item['prompt'], answer=item['response'])

    def __len__(self) -> int:
        return len(self.data)


class SafeRLHFSupervisedTrainDataset(SafeRLHFSupervisedDataset):
    NAME: str = 'PKU-SafeRLHF-Supervised/train'
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'train'


class SafeRLHFSupervisedTestDataset(SafeRLHFSupervisedDataset):
    NAME: str = 'PKU-SafeRLHF-Supervised/test'
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'test'
