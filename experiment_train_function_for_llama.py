import re
from typing import List
import torch
from model_llama3 import KblamLlamaForCausalLM
from model_phi3 import KBLaMPhi3ForCausalLM

def format_QA_llama(Q: str, A: str):
    return ("<|start_header_id|>user<|end_header_id|> "
            + Q
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>"
            + A
            + "<|eot_id|>"
            )

def create_labels_for_llama(input_ids: torch.Tensor, input_strs: List[str], tokenizer):
    # Not sure this is correct. This method simply masks the <|start_header_id|>user<|end_header_id|> then leaves the rest in the labels
    # Possibly what they want is to mask out the query. To do that swap the index from the tokenizer below from 1 to 2
    answer_indices = torch.argmax(
        (input_ids == tokenizer("<|start_header_id|>assistant<|end_header_id|>")["input_ids"][1]).long(),
        -1,
        )
    answer_mask = torch.ones_like(input_ids)
    for b in range(len(input_strs)):
        answer_mask[b, : (answer_indices[b].item() + 2)] = 0
    labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
    return labels

def get_llama3_query_head_parameters(model: KblamLlamaForCausalLM | KBLaMPhi3ForCausalLM,
                                     sep_query_head: bool,
                                     kb_token_layer_frequency: int,
                                     ):
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:  # TODO: this is different for each model type
            # For llama3
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight)  # type: ignore
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params