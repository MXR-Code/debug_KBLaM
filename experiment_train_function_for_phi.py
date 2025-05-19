import re
from typing import List
import torch
from model_llama3 import KblamLlamaForCausalLM
from model_phi3 import KBLaMPhi3ForCausalLM


def format_QA_phi3(Q: str, A: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n" + A + "<|end|>\n"

def create_labels_for_phi3(input_ids: torch.Tensor, input_strs: List[str], tokenizer):
    # We just want to mask out the starting token.
    # The tokenized values are left padded so we want to know where our Q/A pairs start
    # Not 100% this is correct
    answer_indices = torch.argmax((input_ids == tokenizer("<|user|>")["input_ids"][0]).long(),
                                  -1,
                                  )
    answer_mask = torch.ones_like(input_ids)
    for b in range(len(input_strs)):
        answer_mask[b, : (answer_indices[b].item() + 1)] = 0
    labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
    return labels

def get_phi3_query_head_parameters(model: KblamLlamaForCausalLM | KBLaMPhi3ForCausalLM,
                                   sep_query_head: bool,
                                   kb_token_layer_frequency: int,
                                   ):
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:
            # For phi3
            if "qkv_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight[: model.config.hidden_size, :])  # type: ignore
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params