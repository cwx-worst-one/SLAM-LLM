# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import random
from itertools import islice

import numpy as np
import torch
from collections import defaultdict
import json


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
            )
        self.num_replicas = num_replicas
        self.rank = rank
        
    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)
         
    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas
            

class GroupedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True, task_group_path=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        if task_group_path is not None:
            with open(task_group_path) as f:
                self.grouped_indices = json.load(f)
                self.grouped_indices = {k: list(map(int, v)) for k, v in self.grouped_indices.items()}
        else:
            self.grouped_indices = self._group_indices_by_task()

    def _group_indices_by_task(self):
        grouped = defaultdict(list)
        for i, item in enumerate(self.dataset.data_list):
            task = item["task_type"]
            grouped[task].append(i)
        return grouped

    def __iter__(self):
        all_batches = []
        for task_type, indices in self.grouped_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        total = 0
        for indices in self.grouped_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size  # ceil
        return total


class DistributedGroupedBatchSampler(torch.utils.data.Sampler):
    def __init__(self,
                 dataset,
                 batch_size: int,
                 num_replicas: int,
                 rank: int,
                 drop_last: bool = True,
                 shuffle: bool = True,
                 seed: int = 42,
                 task_group_path: str = None):
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.base_sampler = GroupedBatchSampler(
            dataset, batch_size, drop_last, shuffle, task_group_path
        )
        self.grouped_indices = self.base_sampler.grouped_indices

    def __iter__(self):
        rnd = random.Random(self.seed)
        tasks = list(self.grouped_indices.keys())
        if self.shuffle:
            rnd.shuffle(tasks)

        all_step_groups = []
        for task in tasks:
            idxs = list(self.grouped_indices[task])
            if self.shuffle:
                rnd.shuffle(idxs)

            task_batches = [
                idxs[i : i + self.batch_size]
                for i in range(0, len(idxs), self.batch_size)
                if (len(idxs[i : i + self.batch_size]) == self.batch_size or not self.drop_last)
            ]

            m = len(task_batches) // self.num_replicas * self.num_replicas
            task_batches = task_batches[:m]

            for i in range(0, m, self.num_replicas):
                group = task_batches[i : i + self.num_replicas]
                all_step_groups.append(group)

        if self.shuffle:
            rnd.shuffle(all_step_groups)

        for group in all_step_groups:
            yield group[self.rank]

    def __len__(self):
        total = 0
        for task, idxs in self.grouped_indices.items():
            b = len(idxs) // self.batch_size
            if not self.drop_last and len(idxs) % self.batch_size:
                b += 1
            total += (b // self.num_replicas)
        return total