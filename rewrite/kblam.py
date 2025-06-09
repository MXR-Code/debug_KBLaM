from torch.nn import Module
from transformers import FeatureExtractionMixin
import sentence_transformers
import os
import sys
from pathlib import Path
from azure.identity import AuthenticationRecord
from azure.identity import DeviceCodeCredential
from azure.identity import TokenCachePersistenceOptions
from azure.identity import get_bearer_token_provider
from openai import AzureOpenAI
import torch
import json
import numpy as np
import torch
from typing import List
import random


class KBLaM(Module):
    def __init__(self,
                 llm_type='llama',
                 tokenizer=None,
                 sentence_encoder=None,
                 llm=None,
                 key_adapter=None,
                 value_adapter=None,
                 use_extended_question_and_answer=False,
                 use_data_augmentation=False):
        super().__init__()
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.sentence_encoder = sentence_encoder
        self.llm = llm
        self.key_adapter = key_adapter
        self.value_adapter = value_adapter

        self.use_extended_question_and_answer = use_extended_question_and_answer
        self.use_data_augmentation = use_data_augmentation

    def forward(self, batch_data, context_data):
        input_index, attention_mask, true_label = self.tokenize(batch_data=batch_data)
        kb_embed = self.retriever(batch_data=batch_data, context_data=context_data)

        out = self.llm.forward(input_ids=input_index, attention_mask=attention_mask, kb_kvs=kb_embed,
                               output_attentions=False)
        logits = out["logits"]

        return logits

    def tokenize(self, batch_data):
        with torch.autograd.no_grad():
            batch_format_QA = []
            for data in batch_data:
                Q, A = self.select_question_and_answer(data=data)
                if self.llm_type == 'llama':
                    format_QA = self.format_question_and_answer_llama(Q=Q, A=A)
                batch_format_QA.append(format_QA)

            batch_format_QA = self.tokenizer(batch_format_QA, return_tensors="pt", padding=True)

            input_index = batch_format_QA["input_ids"]
            attention_mask = batch_format_QA["attention_mask"]

            if self.llm_type == 'llama':
                true_label = self.create_label_llama(input_ids=input_index, input_strs=batch_format_QA)

        return input_index, attention_mask, true_label

    def retriever(self, batch_data, context_data):
        batch_size = len(batch_data)
        # 1
        key_embed_list, value_embed_list = [], []
        for data in batch_data:
            key_text = data["key_string"]
            key_embed = self.sentence_encoder.forward(sentence=key_text)
            key_embed = self.key_adapter(key_embed)
            key_embed_list.append(key_embed)

            value_text = data["description"]
            value_embed = self.sentence_encoder.forward(sentence=value_text)
            value_embed = self.value_adapter(value_embed)
            value_embed_list.append(value_embed)

        batch_key_embed = torch.stack(key_embed_list)
        batch_value_embed = torch.stack(value_embed_list)

        if len(batch_key_embed.shape) == 2:
            batch_size, embed_dim = batch_key_embed.shape
            batch_size, embed_dim = batch_value_embed.shape
            batch_key_embed = batch_key_embed.unsqueeze(1)
            batch_value_embed = batch_value_embed.unsqueeze(1)

        # 2
        context_key_embed_list, context_value_embed_list = [], []
        for data in context_data:
            key_text = data["key_string"]
            key_embed = self.sentence_encoder.forward(sentence=key_text)
            key_embed = self.key_adapter(key_embed)
            context_key_embed_list.append(key_embed)

            value_text = data["description"]
            value_embed = self.sentence_encoder.forward(sentence=value_text)
            value_embed = self.value_adapter(value_embed)
            context_value_embed_list.append(value_embed)

        context_key_embed = torch.stack(context_key_embed_list)
        context_value_embed = torch.stack(context_value_embed_list)

        context_size, embed_dim = context_key_embed.shape
        context_size, embed_dim = context_value_embed.shape
        context_key_embed = context_key_embed.unsqueeze(0).expand(batch_size, context_size, embed_dim)
        context_value_embed = context_value_embed.unsqueeze(0).expand(batch_size, context_size, embed_dim)

        true_kb_copy = 1
        batch_key_embed = [batch_key_embed] * true_kb_copy + [context_key_embed]
        batch_value_embed = [batch_value_embed] * true_kb_copy + [context_value_embed]
        batch_key_embed = torch.concat(tensors=batch_key_embed, dim=1)
        batch_value_embed = torch.concat(tensors=batch_value_embed, dim=1)

        batch_size, seq_len, embed_dim = batch_key_embed.shape
        seq_len = batch_size // batch_size + context_size

        kb_embedding = (batch_key_embed, batch_value_embed)

        return kb_embedding

    def select_question_and_answer(self, data):
        Q, A = None, None
        if self.use_extended_question_and_answer:
            Q = data["extended_Q"]
            A = data["extended_A"]

        elif self.use_data_augmentation:
            data = data
            templates = ["What {} does {} have?",
                         "What is the {} of {}?",
                         "Tell me about the {} of {}.",
                         "Can you let me know the {} of {}?",
                         "Can you inform me about the {} of {}?",
                         "Describe the {} of {}.",
                         "What details can you share about the {} of {}?",
                         "What kind of {} does {} have?",
                         "Provide details on the {} of {}.",
                         "What features does the {} of {} include?",
                         "Can you elaborate on the {} of {}?",
                         "How would you describe the {} of {}?",
                         "What can you tell me about the {} characteristics of {}?",
                         "Can you explain the {} of {}?",
                         "What insights can you provide about the {} of {}?",
                         "What should I know about the {} of {}?"]
            dtype = data["description_type"]
            name = data["name"]
            tid = np.random.randint(0, len(templates))
            Q = templates[tid].format(dtype, name)
            A = "I am sorry I cannot find relevant information in the KB."

        else:
            Q = data["Q"]
            A = data["A"]

        assert Q is not None and A is not None
        return Q, A

    def format_question_and_answer_llama(self, Q: str, A: str):
        text = "<|start_header_id|>user<|end_header_id|> "
        text += Q
        text += "<|eot_id|>"
        text += "<|start_header_id|>assistant<|end_header_id|>"
        text += A
        text += "<|eot_id|>"
        return text

    def create_label_llama(self, input_ids: torch.Tensor, input_strs: List[str]):
        # Not sure this is correct. This method simply masks the <|start_header_id|>user<|end_header_id|>
        # then leaves the rest in the labels
        # Possibly what they want is to mask out the query.
        # To do that swap the index from the tokenizer below from 1 to 2
        answer_indices = torch.argmax(
            (input_ids == self.tokenizer("<|start_header_id|>assistant<|end_header_id|>")["input_ids"][1]).long(),
            -1,
        )
        answer_mask = torch.ones_like(input_ids)
        for b in range(len(input_strs)):
            answer_mask[b, : (answer_indices[b].item() + 2)] = 0
        labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
        return labels
