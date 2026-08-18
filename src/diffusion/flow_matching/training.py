import torch
from typing import Callable
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler


def constant(alpha, sigma):
    return 1

def time_shift_fn(t, timeshift=1.0):
    return t / (t + (1 - t) * timeshift)

def _pearson_corr_loss(x, x_hat):
    """1 − Pearson correlation, per sample over the flattened signal."""
    B = x.shape[0]
    x_f = x.reshape(B, -1)
    p_f = x_hat.reshape(B, -1)
    x_c = x_f - x_f.mean(dim=1, keepdim=True)
    p_c = p_f - p_f.mean(dim=1, keepdim=True)
    num = (x_c * p_c).sum(dim=1)
    den = torch.sqrt((x_c.pow(2).sum(dim=1) + 1e-8) * (p_c.pow(2).sum(dim=1) + 1e-8))
    corr = num / den
    return (1.0 - corr.clamp(-1.0, 1.0)).mean()


def _gather_visible(x, y, coords, observed_mask):
    """
    Per-sample gather of visible (mask==1) rows.

    Args:
        x:             (B, 1, N_total, T) noisy input
        y:             (B, 1, N_total, T) condition
        coords:        (B, N_total, coord_dim) per-sample coords
        observed_mask: (B, 1, N_total) 1=visible, 0=masked

    Returns:
        x_vis:      (B, 1, N_vis, T)
        y_vis:      (B, 1, N_vis, T)
        coords_vis: (B, N_vis, coord_dim)
    """
    B, _, _, T = x.shape
    coord_dim = coords.shape[-1]

    counts = observed_mask[:, 0, :].sum(dim=-1)
    N_vis = int(counts[0].item())
    if not torch.all(counts == N_vis):
        raise RuntimeError(
            f"Per-sample visible count differs across batch (got {counts.tolist()}); "
            "FlowMatchingTrainer requires a uniform mask ratio per batch."
        )

    # observed_mask is 0/1 → topk gives the visible indices per sample
    vis_idx = torch.topk(observed_mask[:, 0, :], k=N_vis, dim=-1).indices  # (B, N_vis)
    vis_idx, _ = torch.sort(vis_idx, dim=-1)  # stable order for readability

    idx_xy = vis_idx.view(B, 1, N_vis, 1).expand(B, 1, N_vis, T)
    x_vis = torch.gather(x, dim=2, index=idx_xy)
    y_vis = torch.gather(y, dim=2, index=idx_xy)

    idx_c = vis_idx.view(B, N_vis, 1).expand(B, N_vis, coord_dim)
    coords_vis = torch.gather(coords, dim=1, index=idx_c)

    return x_vis, y_vis, coords_vis


class FlowMatchingTrainer(BaseTrainer):
    """
    Each train step splits inputs into encoder-only and decoder-only views.

      Encoder (visible-only):
        x_vis, y_vis, coords_vis = _gather_visible(x_t, y, coords, observed_mask)
        Shape: (B, 1, N_vis, T) and (B, N_vis, coord_dim).

      Decoder (full set):
        x_query   = x_t at every row
        pos_query = coords     # all training-set positions
        Shape: (B, 1, N_total, T) and (B, N_total, coord_dim).
    """
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn: Callable = constant,
            lognorm_t=False,
            timeshift=1.0,
            prediction_type: str = "x",
            loss_space: str = None,
            sigma_eps: float = 1e-3,
            loss_type: str = "l2",
            huber_c: float = 0.001,
            masked_loss_weight: float = 1.0,
            corr_loss_weight: float = 0.0,
            aux_loss_min_alpha: float = 0.7,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        if prediction_type not in ("v", "x"):
            raise ValueError(
                f"prediction_type must be one of 'v', 'x'; got {prediction_type!r}."
            )
        self.prediction_type = prediction_type
        if loss_space not in (None, "v", "x"):
            raise ValueError(
                f"loss_space must be one of None, 'v', 'x'; got {loss_space!r}."
            )
        self.loss_space = loss_space
        self.sigma_eps = sigma_eps
        if loss_type not in ("l2", "l1", "huber"):
            raise ValueError(
                f"loss_type must be one of 'l2', 'l1', 'huber'; got {loss_type!r}."
            )
        self.loss_type = loss_type
        self.huber_c = huber_c
        self.masked_loss_weight = masked_loss_weight
        self.corr_loss_weight = corr_loss_weight
        self.aux_loss_min_alpha = aux_loss_min_alpha
        self.loss_weight_fn = loss_weight_fn

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        batch_size = x.shape[0]
        if self.lognorm_t:
            t = torch.randn(batch_size).to(x.device, x.dtype).sigmoid()
        else:
            t = torch.rand(batch_size).to(x.device, x.dtype)
        t = time_shift_fn(t, self.timeshift)
        noise = torch.randn_like(x)

        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)

        dims = [1] * (x.dim() - alpha.dim())
        alpha = alpha.view(*alpha.shape, *dims)
        dalpha = dalpha.view(*dalpha.shape, *dims)
        sigma = sigma.view(*sigma.shape, *dims)
        dsigma = dsigma.view(*dsigma.shape, *dims)

        x_t = alpha * x + sigma * noise
        v_t = dalpha * x + dsigma * noise

        if metadata is None or 'observed_mask' not in metadata or 'coords' not in metadata:
            raise RuntimeError(
                "FlowMatchingTrainer requires metadata with 'observed_mask' and 'coords'."
            )
        observed_mask = metadata['observed_mask'].to(x.device, x.dtype)
        coords = metadata['coords'].to(x.device, x.dtype)

        # Encoder input: visible electrodes only
        x_vis, y_vis, coords_vis = _gather_visible(x_t, y, coords, observed_mask)

        # Build decoder query
        x_query = x_t
        pos_query = coords

        out = net(
            x_vis, t, y_vis,
            pos=coords_vis,
            x_query=x_query,
            pos_query=pos_query,
        )

        # Calculate FM loss
        loss_space = self.loss_space or self.prediction_type
        if self.prediction_type == loss_space:
            pred = out
        elif loss_space == "v":
            x_hat = out
            sigma_c = sigma.clamp_min(self.sigma_eps)
            pred = dalpha * x_hat + dsigma * (x_t - alpha * x_hat) / sigma_c
        else:
            det = alpha * dsigma - sigma * dalpha
            pred = (dsigma * x_t - sigma * out) / det
        
        target = v_t if loss_space == "v" else x
        err = pred - target
        if self.loss_type == "l2":
            per_elem = err ** 2
        elif self.loss_type == "l1":
            per_elem = err.abs()
        else:
            per_elem = torch.sqrt(err ** 2 + self.huber_c ** 2) - self.huber_c

        weight = self.loss_weight_fn(alpha, sigma)
        loss = weight * per_elem

        if self.masked_loss_weight != 1.0:
            channel_weight = torch.where(
                observed_mask.unsqueeze(-1) == 1,
                torch.ones_like(loss),
                torch.full_like(loss, self.masked_loss_weight),
            )
            loss = loss * channel_weight

        fm_loss = loss.mean()
        out_dict = dict(fm_loss=fm_loss)
        total_loss = fm_loss

        # Auxiliary reconstruction loss (x-space)
        aux_active = self.corr_loss_weight > 0
        if aux_active:
            if self.prediction_type == "x":
                x_hat = out
            else:
                det = alpha * dsigma - sigma * dalpha
                x_hat = (dsigma * x_t - sigma * out) / det

            x_aux, x_hat_aux = x, x_hat
            if self.aux_loss_min_alpha > 0.0:
                gate = alpha.view(alpha.shape[0]) >= self.aux_loss_min_alpha
                x_aux = x[gate]
                x_hat_aux = x_hat[gate]
            aux_active_batch = x_aux.shape[0] > 0

            if aux_active_batch:
                corr_loss = _pearson_corr_loss(x_aux, x_hat_aux)
                out_dict['corr_loss'] = corr_loss
                total_loss = total_loss + self.corr_loss_weight * corr_loss
        
        out_dict['loss'] = total_loss
        return out_dict
