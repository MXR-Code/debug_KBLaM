from typing import Callable, Dict, List, Optional
import numpy as np
from model_kblam_kb_encoder import KBEncoder  # 导入知识库编码器模型
from utils_train import context_set_size_scheduler, get_kb_embd  # 导入训练工具
import torch

class KBRetriever:
    def __init__(self,
                 encoder: KBEncoder,
                 dataset: List[Dict],
                 key_embds: Optional[np.ndarray],
                 value_embds: Optional[np.ndarray]):
        self.encoder = encoder
        self.key_embds = key_embds
        self.value_embds = value_embds
        self.dataset = dataset

    def _use_cached_embd(self):
        if self.key_embds is not None and self.value_embds is not None:
            return True
        else:
            return False

    def get_key_embeddings(self, batch_indices, batch_size, step, kb_size):
        if self._use_cached_embd():
            train_set_key, train_set_val = get_kb_embd(kb_encoder=self.encoder,
                                                       indices=batch_indices,
                                                       kb_dict=None,
                                                       precomputed_embd=(self.key_embds, self.value_embds))
        else:
            train_set_key, train_set_val = get_kb_embd(kb_encoder=self.encoder,
                                                       indices=batch_indices,
                                                       kb_dict=self.dataset,
                                                       precomputed_embd=None)

        if len(train_set_key.shape) == 2:
            # Add comment on why we need this line
            train_set_key = train_set_key.unsqueeze(0).transpose(0, 1)
            train_set_val = train_set_val.unsqueeze(0).transpose(0, 1)

        context_set_size = context_set_size_scheduler(step, kb_size)
        context_set_index = np.random.choice(len(self.dataset), context_set_size, replace=False)  # type: ignore
        if self._use_cached_embd():
            context_set_key, context_set_val = get_kb_embd(kb_encoder=self.encoder,
                                                           indices=context_set_index,
                                                           kb_dict=None,
                                                           precomputed_embd=(self.key_embds, self.value_embds))
        else:
            context_set_key, context_set_val = get_kb_embd(kb_encoder=self.encoder,
                                                           indices=context_set_index,
                                                           kb_dict=self.dataset,
                                                           precomputed_embd=None)

        context_set_key = context_set_key.unsqueeze(0).expand(batch_size, *context_set_key.shape)
        context_set_val = context_set_val.unsqueeze(0).expand(batch_size, *context_set_val.shape)

        # context_set_val = torch.randn_like(context_set_val)
        # Idea: Try torch.randn here context_set_tokens??
        true_kb_copy = 1
        kb_embedding = (torch.concat(tensors=[*([train_set_key] * true_kb_copy), context_set_key], dim=1),
                        torch.concat(tensors=[*([train_set_val] * true_kb_copy), context_set_val], dim=1))

        return kb_embedding
