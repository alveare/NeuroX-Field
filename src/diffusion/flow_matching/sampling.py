import torch
from typing import Callable

from src.diffusion.base.guidance import *
from src.diffusion.base.scheduling import *
from src.diffusion.base.sampling import *
from src.diffusion.flow_matching.training import _gather_visible


def shift_respace_fn(t, shift=3.0):
    return t / (t + (1 - t) * shift)

def ode_step_fn(x, v, dt, s, w):
    return x + v * dt

def sde_mean_step_fn(x, v, dt, s, w):
    return x + v * dt + s * w * dt

def sde_step_fn(x, v, dt, s, w):
    return x + v*dt + s * w* dt + torch.sqrt(2*w*dt)*torch.randn_like(x)



import logging
logger = logging.getLogger(__name__)


def _pull_meta(metadata, device, dtype):
    """
    Extract (observed_mask, coords) tensors from the metadata.
    """
    observed_mask = None
    coords = None
    if metadata is None:
        return observed_mask, coords

    if isinstance(metadata, (list, tuple)):
        if isinstance(metadata[0], dict):
            if 'observed_mask' in metadata[0]:
                masks = [m['observed_mask'] for m in metadata]
                observed_mask = torch.stack(masks, dim=0).to(device, dtype)
            if 'coords' in metadata[0]:
                coord_list = [m['coords'] for m in metadata]
                coords = torch.stack(coord_list, dim=0).to(device, dtype)
    elif isinstance(metadata, dict):
        if 'observed_mask' in metadata:
            observed_mask = metadata['observed_mask'].to(device, dtype)
        if 'coords' in metadata:
            coords = metadata['coords'].to(device, dtype)
    return observed_mask, coords


class EulerSampler(BaseSampler):
    def __init__(
            self,
            w_scheduler: BaseScheduler = None,
            timeshift=1.0,
            prediction_type: str = "x",
            guidance_interval_min: float = 0.0,
            guidance_interval_max: float = 1.0,
            step_fn: Callable = ode_step_fn,
            last_step=None,
            last_step_fn: Callable = ode_step_fn,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if prediction_type not in ("v", "x"):
            raise ValueError(
                f"prediction_type must be one of 'v', 'x'; got {prediction_type!r}."
            )
        self.prediction_type = prediction_type
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        if self.w_scheduler is not None and self.step_fn == ode_step_fn:
            logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    def _impl_sampling(self, net, noise, condition, uncondition, metadata=None):
        """
        Standard FM iteration. At every step:
          encoder input     := visible-only gather of (cfg_x, cfg_condition)
          decoder x_query   := cfg_x  (full montage; zero-shot rows evolve from noise)
          decoder pos_query := cfg_coords  (full montage)
        """
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device, noise.dtype)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        x_trajs = [noise]
        v_trajs = []

        observed_mask, coords = _pull_meta(metadata, noise.device, noise.dtype)
        if observed_mask is None or coords is None:
            raise RuntimeError(
                "EulerSampler requires metadata with both 'observed_mask' and 'coords'."
            )
        cfg_mask = torch.cat([observed_mask, observed_mask], dim=0)
        cfg_coords = torch.cat([coords, coords], dim=0)

        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            dalpha_over_alpha = self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            w = self.w_scheduler.w(t_cur) if self.w_scheduler else 0.0

            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)

            # Encoder input: visible-only gather
            x_vis, y_vis, coords_vis = _gather_visible(
                cfg_x, cfg_condition, cfg_coords, cfg_mask,
            )

            out = net(
                x_vis, cfg_t, y_vis,
                pos=coords_vis,
                x_query=cfg_x,
                pos_query=cfg_coords,
            )

            if t_cur[0] > self.guidance_interval_min and t_cur[0] < self.guidance_interval_max:
                guidance = self.guidance
                out = self.guidance_fn(out, guidance)
            else:
                out = self.guidance_fn(out, 1.0)
            # Recover the velocity [ LinearScheduler: v = (x̂ − x_t)/(1−t) ]
            if self.prediction_type == "x":
                alpha = self.scheduler.alpha(t_cur)
                dalpha = self.scheduler.dalpha(t_cur)
                dsigma = self.scheduler.dsigma(t_cur)
                x_hat = out
                v = dalpha * x_hat + dsigma * (x - alpha * x_hat) / sigma
            else:
                v = out
            s = ((1 / dalpha_over_alpha) * v - x) / (sigma ** 2 - (1 / dalpha_over_alpha) * dsigma_mul_sigma)
            if i < self.num_steps - 1:
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, dt, s=s, w=w)

            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))

        return x_trajs, v_trajs
