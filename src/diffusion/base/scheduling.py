from torch import Tensor

class BaseScheduler:
    def alpha(self, t) -> Tensor:
        ...
    def sigma(self, t) -> Tensor:
        ...

    def dalpha(self, t) -> Tensor:
        ...
    def dsigma(self, t) -> Tensor:
        ...

    def dalpha_over_alpha(self, t) -> Tensor:
        return self.dalpha(t) / self.alpha(t)

    def dsigma_mul_sigma(self, t) -> Tensor:
        return self.dsigma(t)*self.sigma(t)

    def w(self, t):
        return self.sigma(t)
