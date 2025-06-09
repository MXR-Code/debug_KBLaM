import os
import math
import numpy as np
from functools import partial

from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers import AutoTokenizer
from transformers import PretrainedConfig

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
from transformers.models.llama.modeling_llama import _CONFIG_FOR_DOC
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import repeat_kv

from transformers.utils import add_start_docstrings_to_model_forward
from transformers.utils import can_return_tuple
from transformers.utils import is_torch_flex_attn_available
from transformers.utils import replace_return_docstrings
from transformers.utils import is_torch_flex_attn_available

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.modeling_outputs import TokenClassifierOutput

from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import StaticCache

from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils.deprecation import deprecate_kwarg

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.integrations.flex_attention import make_flex_block_causal_mask


class KBLaMConfig(PretrainedConfig):
    def __init__(self,
                 base_model_name_or_path: str = "",
                 kb_token_layer_frequency: int = 3,
                 kb_scale_factor: int | None = None,
                 top_k_kb: int = 100,
                 dynamic_sparsify: bool = False,
                 separate_query_head: bool = False,
                 attn_implementation: str = "eager",
                 **kwargs):
        self.base_model_name_or_path = base_model_name_or_path
        self.kb_layer_frequency = kb_token_layer_frequency
        self.kb_scale_factor = kb_scale_factor
        self.top_k_kb = top_k_kb
        self.dynamic_sparsify = dynamic_sparsify
        self.seperate_query_head = separate_query_head
        self.attn_implementation = attn_implementation
        super().__init__(**kwargs)


class LlamaAttention_KBLaM(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, kb_config=None):
        super().__init__()
        # LlamaAttention Use
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(in_features=config.hidden_size,
                                out_features=config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(in_features=config.hidden_size,
                                out_features=config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(in_features=config.hidden_size,
                                out_features=config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(in_features=config.num_attention_heads * self.head_dim,
                                out_features=config.hidden_size,
                                bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # KBLaM Use
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        if (self.head_dim * self.num_heads) != self.hidden_size: assert False
        self.score_shift = nn.Parameter(torch.zeros(self.num_heads, 1) - 3)
        self.q_proj_new = nn.Linear(in_features=self.hidden_size,
                                    out_features=self.num_heads * self.head_dim,
                                    bias=config.attention_bias)

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
            top_idx = top_idx.view(batch_size, -1, topk_size, 1).expand(
                batch_size, num_heads, topk_size, head_dim
            )
            kb_keys = kb_keys.gather(-2, top_idx)
            kb_values = kb_values.gather(-2, top_idx)
        return kb_keys, kb_values, attn_weights[..., :topk_size]

    def in_tensor_parallel(self, hidden_states):
        if self.config.pretraining_tp > 1:
            pretraining_tp = self.config.pretraining_tp
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // pretraining_tp
            query_slices = self.q_proj.weight.split(split_size=(self.num_heads * self.head_dim) // pretraining_tp,
                                                    dim=0)
            key_slices = self.k_proj.weight.split(split_size=key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(split_size=key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            query_states_2 = self.q_proj_new(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        return query_states, key_states, value_states, query_states_2

    def attention_forward(self,
                          key_states,
                          value_states,
                          kb_kvs,
                          batch_size,
                          query_states_2,
                          attention_mask,
                          query_seq_len,
                          query_states,
                          save_attention_weights,
                          attention_save_loc,
                          attention_file_base_name,
                          kb_layer_frequency,
                          dynamic_sparsify,
                          topk_size,
                          seperate_query_head,
                          kb_scale_factor
                          ):
        PADDING_VALUE = torch.finfo(torch.bfloat16).min

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # KBLaM Part
        attn_weights_2 = None

        if kb_kvs is not None:
            if self.layer_idx % kb_layer_frequency == 0:
                kb_keys, kb_values = kb_kvs  # (kb_len, head_dim * num_heads * num_adapters)
                kb_idx = self.layer_idx // kb_layer_frequency  # Should be something inside the kb config
                if len(kb_keys.shape) == 2:  # Not batch dim
                    kb_len = kb_keys.shape[0]

                    shape = (kb_len, 1 + self.config.num_hidden_layers // kb_layer_frequency, -1)
                    kb_keys = kb_keys.reshape(shape)[:, kb_idx]

                    shape = (kb_len, 1 + self.config.num_hidden_layers // kb_layer_frequency, -1)
                    kb_values = kb_values.reshape(shape)[:, kb_idx]

                    kb_keys = kb_keys.view(kb_len, self.num_heads, self.head_dim).transpose(0, 1)
                    kb_values = kb_values.view(kb_len, self.num_heads, self.head_dim).transpose(0, 1)

                    kb_keys = kb_keys.unsqueeze(0).expand(batch_size, self.num_heads, kb_len, self.head_dim)
                    kb_values = kb_values.unsqueeze(0).expand(batch_size, self.num_heads, kb_len, self.head_dim)

                    if dynamic_sparsify:
                        kb_keys, kb_values, attn_weights_2 = self.prune_key_value(query=query_states_2,
                                                                                  kb_keys=kb_keys,
                                                                                  kb_values=kb_values,
                                                                                  topk_size=topk_size)
                    # Append the KB keys and values in the front, in front of padding
                    key_states = torch.concat([kb_keys, key_states], dim=2)
                    value_states = torch.concat([kb_values, value_states], dim=2)

                elif len(kb_keys.shape) == 3:  # Has a batch dim
                    kb_len = kb_keys.shape[1]

                    shape = (batch_size, kb_len, 1 + self.config.num_hidden_layers // kb_layer_frequency, -1)
                    kb_keys = kb_keys.view(shape)[:, :, kb_idx]

                    shape = (batch_size, kb_len, 1 + self.config.num_hidden_layers // kb_layer_frequency, -1)
                    kb_values = kb_values.view(shape)[:, :, kb_idx]

                    kb_keys = kb_keys.view(batch_size, kb_len, self.num_heads, self.head_dim).transpose(1, 2)
                    kb_values = kb_values.view(batch_size, kb_len, self.num_heads, self.head_dim).transpose(1, 2)

                    if dynamic_sparsify:
                        kb_keys, kb_values, attn_weights_2 = self.prune_key_value(query=query_states_2,
                                                                                  kb_keys=kb_keys,
                                                                                  kb_values=kb_values,
                                                                                  topk_size=topk_size)
                    # Append the KB keys and values in the front, in front of padding
                    key_states = torch.concat([kb_keys, key_states], dim=2)
                    value_states = torch.concat([kb_values, value_states], dim=2)

                # Modify the attention matrix: Appendx a (seq_len, kb_len) block to the left
                kb_len = kb_keys.shape[2]
                kb_atten_mask = attention_mask.new_zeros(batch_size, 1, query_seq_len, kb_len)
                padding_mask = torch.all(attention_mask < 0, -1, keepdim=True)  # (bsz, num_heads, q_len, 1)
                kb_atten_mask = padding_mask * PADDING_VALUE + (~padding_mask) * kb_atten_mask
                attention_mask = torch.concat([kb_atten_mask, attention_mask], dim=-1)

        # eager_attention_forward
        attn_weights = torch.matmul(query_states, key_states.transpose(dim0=2, dim1=3)) / math.sqrt(self.head_dim)

        # KBLaM Part
        if seperate_query_head:
            if kb_kvs is not None:
                if self.layer_idx % kb_layer_frequency == 0:
                    # If we have pruned the KB tokens, then this quantity should have been computed,
                    # if not, then we compute it here
                    if attn_weights_2 is None:
                        kb_keys = kb_keys.transpose(2, 3)
                        attn_weights_2 = torch.matmul(query_states_2, kb_keys) / math.sqrt(self.head_dim)
                    attn_weights = attn_weights[:, :, :, kb_len:]
                    if kb_scale_factor is not None:
                        attn_weights_2 = (attn_weights_2 - np.log(kb_len) + np.log(kb_scale_factor))
                    attn_weights = torch.concat(tensors=[attn_weights_2, attn_weights], dim=-1)

        # eager_attention_forward
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

        # KBLaM Part
        if not attn_weights.requires_grad:
            # TODO: Make this function injectable
            if save_attention_weights:
                if query_seq_len > 1:
                    save_path = os.path.join(attention_save_loc, f"{attention_file_base_name}_{self.layer_idx}.npy")
                    np.save(save_path, attn_weights.to(torch.float32).cpu().detach().numpy())

        # eager_attention_forward
        attn_weights = attn_weights.to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (batch_size, self.num_heads, query_seq_len, self.head_dim):
            assert False
        attn_output = attn_output.transpose(dim0=1, dim1=2).contiguous()

        return attn_output, attn_weights

    def out_tensor_parallel(self, attn_output):
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(split_size=self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(split_size=self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output

    def forward(self,
                # Llama Use
                hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                past_key_value: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor] = None,
                # KBLaM Use
                position_ids: Optional[torch.LongTensor] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                kb_kvs: Optional[tuple] = None,
                kb_layer_frequency=3,
                dynamic_sparsify=False,
                topk_size=100,
                seperate_query_head=False,
                kb_scale_factor=None,
                save_attention_weights: bool = True,
                attention_save_loc: Optional[str] = None,
                attention_file_base_name: Optional[str] = None,
                **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if save_attention_weights:
            assert attention_save_loc is not None, "Please provide a location to save the attention weights"
            assert attention_file_base_name is not None, "Please provide a base name for the attention weights"

        batch_size, query_seq_len, hidden_dim = hidden_states.size()

        query_states, key_states, value_states, query_states_2 = self.in_tensor_parallel(hidden_states=hidden_states)

        query_states = query_states.view(batch_size, query_seq_len, self.num_heads, self.head_dim)
        query_states = query_states.transpose(dim0=1, dim1=2)

        query_states_2 = query_states_2.view(batch_size, query_seq_len, self.num_heads, self.head_dim)
        query_states_2 = query_states_2.transpose(dim0=1, dim1=2)

        key_states = key_states.view(batch_size, query_seq_len, self.num_key_value_heads, self.head_dim)
        key_states = key_states.transpose(dim0=1, dim1=2)

        value_states = value_states.view(batch_size, query_seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.transpose(dim0=1, dim1=2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # eager_attention_forward
        attn_output, attn_weights = self.attention_forward(query_states=query_states,
                                                           key_states=key_states,
                                                           value_states=value_states,
                                                           query_seq_len=query_seq_len,
                                                           batch_size=batch_size,
                                                           attention_mask=attention_mask,
                                                           # KBLaM Use
                                                           query_states_2=query_states_2,
                                                           kb_kvs=kb_kvs,
                                                           save_attention_weights=save_attention_weights,
                                                           attention_save_loc=attention_save_loc,
                                                           attention_file_base_name=attention_file_base_name,
                                                           kb_layer_frequency=kb_layer_frequency,
                                                           dynamic_sparsify=dynamic_sparsify,
                                                           topk_size=topk_size,
                                                           seperate_query_head=seperate_query_head,
                                                           kb_scale_factor=kb_scale_factor)

        attn_output = attn_output.reshape(batch_size, query_seq_len, self.hidden_size)

        attn_output = self.out_tensor_parallel(attn_output=attn_output)

        if output_attentions:
            attn_weights = attn_weights
        else:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer_KBLaM(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention_KBLaM(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                # KBLaM Use
                kb_kvs: Optional[tuple] = None,
                kb_config: Optional[KBLaMConfig] = None,
                save_attention_weights: bool = False,
                attention_save_loc: Optional[str] = None,
                attention_file_base_name: Optional[str] = None,
                # Llama Use
                hidden_states: torch.Tensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                # necessary, but kept here for BC
                **kwargs: Unpack[FlashAttentionKwargs],
                ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states.clone()

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            # KBLaM Use
            kb_kvs=kb_kvs,
            kb_config=kb_config,
            save_attention_weights=save_attention_weights,
            attention_save_loc=attention_save_loc,
            attention_file_base_name=attention_file_base_name,
            # Llama Use
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states.clone()
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaModel_KBLaM(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size,
                                         embedding_dim=config.hidden_size,
                                         padding_idx=self.padding_idx)
        layers = [LlamaDecoderLayer_KBLaM(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        self.layers = nn.ModuleList(layers)
        self.norm = LlamaRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(self,
                # KBLaM Use
                kb_kvs: Optional[tuple] = None,
                return_dict: Optional[bool] = None,
                kb_config: Optional[KBLaMConfig] = None,
                save_attention_weights: bool = False,
                attention_save_loc: Optional[str] = None,
                attention_file_base_name: Optional[str] = None,
                # Llama Use
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
                ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            # logger.warning_once("use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False")
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    # KBLaM
                    kb_kvs=kb_kvs,
                    kb_config=kb_config,
                    save_attention_weights=save_attention_weights,
                    attention_save_loc=attention_save_loc,
                    attention_file_base_name=attention_file_base_name,
                    # Llama
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs)
            else:
                layer_outputs = decoder_layer(
                    # KBLaM
                    kb_kvs=kb_kvs,
                    kb_config=kb_config,
                    save_attention_weights=save_attention_weights,
                    attention_save_loc=attention_save_loc,
                    attention_file_base_name=attention_file_base_name,
                    # Llama
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs)

            if output_attentions and use_cache:
                hidden_states, all_self_attns, next_decoder_cache = layer_outputs
            elif output_attentions and ~use_cache:
                hidden_states, all_self_attns = layer_outputs
            elif ~output_attentions and use_cache:
                hidden_states, next_decoder_cache = layer_outputs
            elif ~output_attentions and ~use_cache:
                hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values if use_cache else None,
                                       hidden_states=all_hidden_states,
                                       attentions=all_self_attns)

    def _update_causal_mask(self,
                            attention_mask: torch.Tensor,
                            input_tensor: torch.Tensor,
                            cache_position: torch.Tensor,
                            past_key_values: Cache,
                            output_attentions: bool = False):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(attention_mask=attention_mask,
                                                               inputs_embeds=input_tensor,
                                                               past_key_values_length=past_seen_tokens,
                                                               is_training=self.training):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]

        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            if isinstance(attention_mask, torch.Tensor):
                target_length = attention_mask.shape[-1]
            else:
                target_length = past_seen_tokens + sequence_length + 1

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(attention_mask=attention_mask,
                                                                                 sequence_length=sequence_length,
                                                                                 target_length=target_length,
                                                                                 dtype=dtype,
                                                                                 device=device,
                                                                                 cache_position=cache_position,
                                                                                 batch_size=input_tensor.shape[0])

        condition1 = self.config._attn_implementation == "sdpa"
        condition2 = attention_mask is not None
        condition3 = attention_mask.device.type in ["cuda", "xpu"]
        condition4 = not output_attentions
        if condition1 and condition2 and condition3 and condition4:
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(attention_mask: torch.Tensor,
                                                              sequence_length: int,
                                                              target_length: int,
                                                              dtype: torch.dtype,
                                                              device: torch.device,
                                                              cache_position: torch.Tensor,
                                                              batch_size: int,
                                                              **kwargs
                                                              ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(size=(sequence_length, target_length),
                                     fill_value=min_dtype,
                                     dtype=dtype,
                                     device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                device = causal_mask.device
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask,
                                                                                                    min_dtype)

        return causal_mask


class LlamaForCausalLM_KBLaM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        # KBLaM
        if hasattr(config, "base_model_name_or_path"):
            base_model_name_or_path = config.base_model_name_or_path
            self.model = LlamaModel_KBLaM.from_pretrained(base_model_name_or_path=base_model_name_or_path,
                                                          torch_dtype=config.torch_dtype)
        else:
            self.model = LlamaModel_KBLaM(config=config)

        if config._attn_implementation == "flash_attention_2":
            raise NotImplementedError("Flash Attention 2 is not yet supported for KBLaM.")

        # Llama
        # self.model = LlamaModel_KBLaM(config=config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(in_features=config.hidden_size,
                                 out_features=config.vocab_size,
                                 bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # KBLaM
    def get_kblam_config(self):
        return self.config

    # KBLaM
    def set_kblam_config(self, config):
        self.config = config

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self,
                # KBLaM
                kb_kvs: Optional[tuple] = None,
                return_dict: Optional[bool] = None,
                kb_config: Optional[KBLaMConfig] = None,
                save_attention_weights: bool = False,
                attention_save_loc: Optional[str] = None,
                attention_file_base_name: Optional[str] = None,
                # Llama
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kwargs: Unpack[KwargsForCausalLM],
                ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model.forward(
            # KBLaM Use
            kb_kvs=kb_kvs,
            return_dict=return_dict,
            kb_config=kb_config,
            save_attention_weights=save_attention_weights,
            attention_save_loc=attention_save_loc,
            attention_file_base_name=attention_file_base_name,
            # Llama Use
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs)

        hidden_states = outputs.last_hidden_state

        # Llama
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])

        # KBLaM
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(split_size=self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(loss=loss,
                                      logits=logits,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions, )
