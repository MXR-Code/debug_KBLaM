import torch
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import *
from llama_attention_kblam import LlamaAttention_KBLaM

# 指定模型名称
model_name = "decapoda-research/llama-7b-hf"

# 加载预训练的 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载预训练的模型
model = AutoModel.from_pretrained(model_name)


class Llama_Kblam(Module):
    def __init__(self, llama_config, kblam_config,
                 huggingface_acess_token,

                 llama_name="meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__()
        self.llama =  LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=llama_name,
                                                       token=huggingface_acess_token)

        self.llama_kblam = LlamaForCausalLM(config=llama_config)

        num_layer = len(self.llama_kblam.model.layers)
        for layer_index in range(num_layer):
            kblam_attention = LlamaAttention_KBLaM(config=kblam_config, layer_idx=layer_index)
            self.llama_kblam.model.layers[layer_index].self_attn = kblam_attention

        # copy parameter from self.llama
        # self.llama_kblam.parameters() = self.llama.parameters()



