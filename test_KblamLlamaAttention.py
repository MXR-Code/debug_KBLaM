import math
import os

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
from model_kblam_config import KBLaMConfig


class KblamLlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            assert False

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.score_shift = nn.Parameter(torch.zeros(self.num_heads, 1) - 3)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.q_proj_new = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)

        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def prune_key_value(self, query, kb_keys, kb_values, topk_size=20):
        assert query.requires_grad is False, "This function should only be used at test time"
        batch_size, num_heads, kb_len, head_dim = kb_keys.shape

        # Batchsize, num_heads, query_size, key_size
        attn_weights = torch.matmul(query, kb_keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if topk_size >= kb_len:
            return kb_keys, kb_values, attn_weights

        with torch.autograd.no_grad():
            top_idx = attn_weights.sum((1, 2)).topk(min(kb_len, topk_size), -1)[1]
            # top_idx = attn_weights.sum(1).topk(topk_size, -1)[1]
            top_idx = top_idx.view(batch_size, -1, topk_size, 1).expand(batch_size, num_heads, topk_size, head_dim)
            kb_keys = kb_keys.gather(-2, top_idx)
            kb_values = kb_values.gather(-2, top_idx)

        attn_weights = attn_weights[..., :topk_size]

        return kb_keys, kb_values, attn_weights

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                kb_kvs: Optional[tuple] = None,
                kb_config: Optional[KBLaMConfig] = None,
                save_attention_weights: bool = True,
                attention_save_loc: Optional[str] = None,
                attention_file_base_name: Optional[str] = None,
                **kwargs,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if save_attention_weights:
            assert attention_save_loc is not None, "Please provide a location to save the attention weights"
            assert attention_file_base_name is not None, "Please provide a base name for the attention weights"

        batch_size, q_len, hidden_dim = hidden_states.size()

        if self.config.pretraining_tp > 1: # 预训练张量并行度 (Pretraining Tensor Parallelism)
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp

            split_size = (self.num_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(split_size=split_size, dim=0)
            key_slices = self.k_proj.weight.split(split_size=key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(split_size=key_value_slicing, dim=0)

            query_states = []
            for i in range(self.config.pretraining_tp):
                state = F.linear(input=hidden_states, weight=query_slices[i])
                query_states.append(state)
            query_states = torch.cat(query_states, dim=-1)

            key_states = []
            for i in range(self.config.pretraining_tp):
                state = F.linear(input=hidden_states, weight=key_slices[i])
                key_states.append(state)
            key_states = torch.cat(key_states, dim=-1)

            value_states = []
            for i in range(self.config.pretraining_tp):
                state = F.linear(input=hidden_states, weight=value_slices[i])
                value_states.append(state)
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            query_states_2 = self.q_proj_new(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        query_states = query_states.transpose(dim0=1, dim1=2)

        query_states_2 = query_states_2.view(batch_size, q_len, self.num_heads, self.head_dim)
        query_states_2 = query_states_2.transpose(dim0=1, dim1=2)

        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
        key_states = key_states.transpose(dim0=1, dim1=2)

        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.transpose(dim0=1, dim1=2)

        cos, sin = self.rotary_emb.forward(x=value_states, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(q=query_states, k=key_states, cos=cos, sin=sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin,
                            "cos": cos,
                            "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states=key_states,
                                                             value_states=value_states,
                                                             layer_idx=self.layer_idx,
                                                             cache_kwargs=cache_kwargs)

        key_states = repeat_kv(hidden_states=key_states, n_rep=self.num_key_value_groups)
        value_states = repeat_kv(hidden_states=value_states, n_rep=self.num_key_value_groups)

        kb_layer_frequency = kb_config.kb_layer_frequency
        dynamic_sparsify = kb_config.dynamic_sparsify
        topk_size = kb_config.top_k_kb
        attn_weights_2 = None

        if kb_kvs is not None:
            if self.layer_idx % kb_layer_frequency == 0:
                kb_keys, kb_values = kb_kvs  # (kb_len, head_dim * num_heads * num_adapters)

                # Should be something inside the kb config [what??]
                # kb_idx = self.layer_idx // kb_layer_frequency

                if len(kb_keys.shape) == 2:  # Not batch dim
                    kb_len = kb_keys.shape[0]

                    # shape = (kb_len, 1 + self.config.num_hidden_layers // kb_layer_frequency, -1)
                    # kb_keys = kb_keys.reshape(shape)
                    # kb_keys = kb_keys[:, kb_idx]

                    # shape = (kb_len, 1 + self.config.num_hidden_layers // kb_layer_frequency, -1)
                    # kb_values = kb_values.reshape(shape)
                    # kb_values = kb_values[:, kb_idx]

                    shape = (kb_len, self.num_heads, self.head_dim)
                    kb_keys = kb_keys.view(shape).transpose(0, 1)

                    shape = (kb_len, self.num_heads, self.head_dim)
                    kb_values = kb_values.view(shape).transpose(0, 1)

                    shape = (batch_size, self.num_heads, kb_len, self.head_dim)
                    kb_keys = kb_keys.unsqueeze(0).expand(shape)

                    shape = (batch_size, self.num_heads, kb_len, self.head_dim)
                    kb_values = kb_values.unsqueeze(0).expand(shape)

                    if dynamic_sparsify:
                        kb_keys, kb_values, attn_weights_2 = self.prune_key_value(query=query_states_2,
                                                                                  kb_keys=kb_keys,
                                                                                  kb_values=kb_values,
                                                                                  topk_size=topk_size)
                    # Append the KB keys and values in the front, in front of padding
                    # batch_size, num_head, kb_len+q_len, head_dim
                    key_states = torch.concat([kb_keys, key_states], dim=2)
                    value_states = torch.concat([kb_values, value_states], dim=2)

                elif len(kb_keys.shape) == 3:  # Has a batch dim
                    kb_len = kb_keys.shape[1]

                    shape = (batch_size, kb_len, 1 + self.config.num_hidden_layers // kb_layer_frequency, -1)
                    kb_keys = kb_keys.view(shape)
                    # kb_keys = kb_keys[:, :, kb_idx]

                    shape = (batch_size, kb_len, 1 + self.config.num_hidden_layers // kb_layer_frequency, -1,)
                    kb_values = kb_values.view(shape)
                    # kb_values = kb_values[:, :, kb_idx]

                    kb_keys = kb_keys.view(batch_size, kb_len, self.num_heads, self.head_dim)
                    kb_keys = kb_keys.transpose(1, 2)

                    kb_values = kb_values.view(batch_size, kb_len, self.num_heads, self.head_dim)
                    kb_values = kb_values.transpose(1, 2)

                    if dynamic_sparsify:
                        kb_keys, kb_values, attn_weights_2 = self.prune_key_value(query=query_states_2,
                                                                                  kb_keys=kb_keys,
                                                                                  kb_values=kb_values,
                                                                                  topk_size=topk_size)
                    # Append the KB keys and values in the front, in front of padding
                    key_states = torch.concat(tensors=[kb_keys, key_states], dim=2)
                    value_states = torch.concat(tensors=[kb_values, value_states], dim=2)

                # Modify the attention matrix: Appendx a (seq_len, kb_len) block to the left
                kb_len = kb_keys.shape[2]
                kb_atten_mask = attention_mask.new_zeros(batch_size, 1, q_len, kb_len)
                padding_mask = torch.all(input=attention_mask < 0, dim=-1, keepdim=True)  # (bsz, num_heads, q_len, 1)
                kb_atten_mask = padding_mask * PADDING_VALUE + (~padding_mask) * kb_atten_mask
                attention_mask = torch.concat(tensors=[kb_atten_mask, attention_mask], dim=-1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        sep_query_head = kb_config.sep_query_head
        kb_scale_factor = kb_config.kb_scale_factor

        if sep_query_head:
            if kb_kvs is not None:
                if self.layer_idx % kb_layer_frequency == 0:
                    # If we have pruned the KB tokens, then this quantity should have been computed,
                    # if not, then we compute it here
                    if attn_weights_2 is None:
                        attn_weights_2 = torch.matmul(query_states_2, kb_keys.transpose(2, 3)) / math.sqrt(
                            self.head_dim)
                    attn_weights = attn_weights[:, :, :, kb_len:]
                    if kb_scale_factor is not None:
                        attn_weights_2 = (attn_weights_2 - np.log(kb_len) + np.log(kb_scale_factor))
                    attn_weights = torch.concat(tensors=[attn_weights_2, attn_weights], dim=-1)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

        if not attn_weights.requires_grad:
            # TODO: Make this function injectable
            if save_attention_weights:
                if q_len > 1:
                    save_path = os.path.join(attention_save_loc, f"{attention_file_base_name}_{self.layer_idx}.npy")
                    np.save(file=save_path, arr=attn_weights.to(torch.float32).cpu().detach().numpy())

        attn_weights = attn_weights.to(query_states.dtype)
        attn_weights = nn.functional.dropout(input=attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)},"
                             f" but is {attn_output.size()}")

        attn_output = attn_output.transpose(dim0=1, dim1=2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(split_size=self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(split_size=self.hidden_size // self.config.pretraining_tp, dim=1)

            attn_output = [F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)]
            attn_output = sum(attn_output)
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


PADDING_VALUE = -1e9
config = LlamaConfig(hidden_size=16,  # 保持较小维度方便测试
                     num_attention_heads=2,
                     num_key_value_heads=2,
                     max_position_embeddings=10,
                     attention_dropout=0.0,  # 关闭 dropout
                     attention_bias=False,  # 关闭 bias
                     pretraining_tp=1,  # 关键设置，避免多路径计算
                     rope_theta=10000
                     )
attention_module = KblamLlamaAttention(config, layer_idx=0)

# 简单输入数据
batch_size = 1
q_len = 4
hidden_states = torch.randn(batch_size, q_len, config.hidden_size)
position_ids = torch.arange(q_len).unsqueeze(0).expand(batch_size, q_len)
attention_mask = torch.ones(batch_size, 1, q_len, q_len)  # 全部注意力

# 创建KBLaMConfig实例
kb_config = KBLaMConfig(kb_token_layer_frequency=1,
                        dynamic_sparsify=False,
                        top_k_kb=2,
                        separate_query_head=False,
                        kb_scale_factor=None)

# 创建kb_kvs，假设kb_len=2，head_dim=8
kb_len = 2
head_dim = config.hidden_size // config.num_attention_heads
kb_keys = torch.randn(kb_len, head_dim * config.num_attention_heads)
kb_values = torch.randn(kb_len, head_dim * config.num_attention_heads)
kb_kvs = (kb_keys, kb_values)

# 模型前向传播
output, attn_weights, past_key_value = attention_module(hidden_states=hidden_states,
                                                        attention_mask=attention_mask,
                                                        position_ids=position_ids,
                                                        kb_kvs=kb_kvs,
                                                        kb_config=kb_config,
                                                        save_attention_weights=False)

print(output.shape)  # 输出张量的形状
print(attn_weights)  # 注意力权重 (应该为None,因为save_attention_weights=False)
