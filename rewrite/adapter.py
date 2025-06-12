from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import LayerNorm
from torch.nn import Module
from transformers import FeatureExtractionMixin


class KeyAdapter(Module, FeatureExtractionMixin):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model = Sequential(Linear(self.in_dim, self.out_dim),
                                LayerNorm(self.out_dim, elementwise_affine=False, bias=False))

    def forward(self, key_embed=None):
        key_embed = self.model(key_embed)
        key_embed = key_embed.bfloat16()
        return key_embed


class ValueAdapter(Module, FeatureExtractionMixin):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model = Linear(self.in_dim, self.out_dim)

    def forward(self, value_embed=None):
        value_embed = self.model(value_embed)
        value_embed = value_embed.bfloat16()

        return value_embed