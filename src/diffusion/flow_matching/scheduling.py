import torch
from src.diffusion.base.scheduling import *


class LinearScheduler(BaseScheduler):
    def alpha(self, t) -> Tensor:
        return (t).view(-1, 1, 1, 1)
    def sigma(self, t) -> Tensor:
        return (1-t).view(-1, 1, 1, 1)
    def dalpha(self, t) -> Tensor:
        return torch.full_like(t, 1.0).view(-1, 1, 1, 1)
    def dsigma(self, t) -> Tensor:
        return torch.full_like(t, -1.0).view(-1, 1, 1, 1)

class ConstScheduler(BaseScheduler):
    def w(self, t):
        return torch.ones(1, 1, 1, 1).to(t.device, t.dtype)




