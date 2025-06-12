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
        self.dataset_name=dataset_name
        self.data = data
        self.num_train = num_train
        self.num_valid = 1000
        self.batch_size = batch_size
        self.knowledge_base_size = knowledge_base_size

        self.use_extended_question_and_answer = use_extended_question_and_answer
        self.use_multi_entity = use_multi_entity
        self.use_data_augmentation = use_data_augmentation

        self.train_data = self.data[:self.num_train]
        self.valid_data = self.data[self.num_train:self.num_train + self.num_valid]
        self.test_data = self.data[self.num_train + self.num_valid:]
        self.num_test = len(self.test_data)

        self.num_train_batch = None
        self.train_data_index = None
        self.batch_split_train_data_index = None
        self.batch_train_data = None
        self.context_train_data = None
        self.shuffle_train_data()

        self.num_valid_batch = None
        self.valid_data_index = None
        self.batch_split_valid_data_index = None
        self.batch_valid_data = None
        self.context_valid_data = None
        self.shuffle_valid_data()

        self.num_test_batch = None
        self.test_data_index = None
        self.batch_split_test_data_index = None
        self.batch_test_data = None
        self.context_test_data = None
        self.shuffle_test_data()

    def shuffle_train_data(self):
        self.num_train_batch = self.num_train // self.batch_size
        if self.num_train_batch * self.batch_size < self.num_train:
            self.num_train_batch += 1
        self.train_data_index = np.arange(start=0, stop=self.num_train, step=1, dtype=int)
        np.random.shuffle(self.train_data_index)
        self.batch_split_train_data_index = np.array_split(ary=self.train_data_index,
                                                           indices_or_sections=self.num_train_batch)

    def train_dataloader(self, epoch, batch_index):
        data_index = self.batch_split_train_data_index[batch_index]
        self.batch_train_data = [self.train_data[i] for i in data_index]

        context_size = self.context_size_scheduler(epoch=epoch, kb_size=self.knowledge_base_size)
        self.context_train_data = random.sample(self.train_data, context_size)
        return self.batch_train_data, self.context_train_data

    def shuffle_valid_data(self):
        self.num_valid_batch = self.num_valid // self.batch_size
        if self.num_valid_batch * self.batch_size < self.num_valid:
            self.num_valid_batch += 1
        self.valid_data_index = np.arange(start=0, stop=self.num_valid, step=1, dtype=int)
        np.random.shuffle(self.valid_data_index)
        self.batch_split_valid_data_index = np.array_split(ary=self.valid_data_index,
                                                           indices_or_sections=self.num_valid_batch)

    def valid_dataloader(self, epoch, batch_index):
        data_index = self.batch_split_valid_data_index[batch_index]
        self.batch_valid_data = [self.valid_data[i] for i in data_index]

        context_size = self.context_size_scheduler(epoch=epoch, kb_size=self.knowledge_base_size)
        self.context_valid_data = random.sample(self.valid_data, context_size)
        return self.batch_valid_data, self.context_valid_data

    def shuffle_test_data(self):
        self.num_test_batch = self.num_test // self.batch_size
        if self.num_test_batch * self.batch_size < self.num_test:
            self.num_test_batch += 1
        self.test_data_index = np.arange(start=0, stop=self.num_test, step=1, dtype=int)
        np.random.shuffle(self.test_data_index)
        self.batch_split_test_data_index = np.array_split(ary=self.test_data_index,
                                                           indices_or_sections=self.num_test_batch)

    def test_dataloader(self, epoch, batch_index):
        data_index = self.batch_split_test_data_index[batch_index]
        self.batch_test_data = [self.test_data[i] for i in data_index]

        context_size = self.context_size_scheduler(epoch=epoch, kb_size=self.knowledge_base_size)
        self.context_test_data = random.sample(self.test_data, context_size)
        return self.batch_test_data, self.context_test_data


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
