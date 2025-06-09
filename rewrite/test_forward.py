import json
import logging
import os
import pathlib
import wandb
import numpy as np
import torch
from transformers import AutoTokenizer
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from accelerate import Accelerator

from dataloader import Dataloader
from transformers.models.llama.modeling_llama import LlamaConfig
from llama_kblam import LlamaForCausalLM_KBLaM
from sentence_transformer import SentenceEncoder
from adapter import KeyAdapter, ValueAdapter
from kblam import KBLaM

device = "cpu"
torch.manual_seed(seed=1)
np.random.seed(seed=1)

torch.cuda.empty_cache()

dataloader = Dataloader()

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
                                          trust_remote_code=True,
                                          token=)
tokenizer.pad_token = tokenizer.eos_token

llama_config = LlamaConfig(num_hidden_layers=1, vocab_size=128009 + 1000)
llm = LlamaForCausalLM_KBLaM(config=llama_config)

kb_token_layer_frequency = 3
out_dim = llm.config.hidden_size * (llm.config.num_hidden_layers // kb_token_layer_frequency + 1)
sentence_encoder = SentenceEncoder(model_name="sentence-transformers/all-mpnet-base-v2", device=device)
key_adapter = KeyAdapter(in_dim=sentence_encoder.out_dim, out_dim=out_dim)
value_adapter = ValueAdapter(in_dim=sentence_encoder.out_dim, out_dim=out_dim)

kblam = KBLaM(tokenizer=tokenizer,
              sentence_encoder=sentence_encoder,
              key_adapter=key_adapter,
              value_adapter=value_adapter,
              llm=llm)

for epoch in range(0, 10):
    batch_data, context_data = dataloader.train_dataloader(epoch=epoch)
    logits = kblam.forward(batch_data=batch_data, context_data=context_data)
