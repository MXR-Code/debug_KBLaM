import json
import numpy as np
import torch
from typing import List
import random


class Dataloader():
    def __init__(self,
                 dataset_name="synthetic.json",
                 num_train=120000,
                 batch_size=10,
                 knowledge_base_size=None,
                 use_extended_question_and_answer=False,
                 use_multi_entity=False,
                 use_data_augmentation=False):
        data = json.load(open(dataset_name))
        self.data = data
        self.train_data = data[:num_train]
        self.test_data = data[num_train:]
        self.batch_size = batch_size

        self.knowledge_base_size = knowledge_base_size

        self.use_extended_question_and_answer = use_extended_question_and_answer
        self.use_multi_entity = use_multi_entity
        self.use_data_augmentation = use_data_augmentation

    def train_dataloader(self, epoch):
        batch_data = random.sample(self.train_data, self.batch_size)
        context_size = self.context_size_scheduler(epoch=epoch, kb_size=self.knowledge_base_size)
        context_data = random.sample(self.train_data, context_size)
        return batch_data, context_data

    def context_size_scheduler(self, epoch: int, kb_size: list[int] | int | str) -> int:
        """Determines the KB size for the current training step.
        The KB size can be a fixed number, a list of numbers or a "dynamic" value.
        If no KB size is provided, the KB size is dynamicly increased every 100 steps."""

        dynamic_range = (10, 200)
        if kb_size == "dynamic":
            return np.random.randint(dynamic_range[0], dynamic_range[1])

        if isinstance(kb_size, list):
            return np.random.randint(kb_size[0], kb_size[1])

        increase_kb_size_every = 100
        if not kb_size:
            round = (epoch) // increase_kb_size_every
            return 4 * (round + 1)

        return kb_size







