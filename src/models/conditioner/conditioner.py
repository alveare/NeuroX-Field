import torch
import torch.nn as nn
from src.models.conditioner.base import BaseConditioner

class EEGConditioner(BaseConditioner):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else nn.Identity()
    
    def _impl_condition(self, y, metadata):
        condition = self.encoder(y)
        return condition
    
    def _impl_uncondition(self, y, metadata):
        return torch.zeros_like(y)
