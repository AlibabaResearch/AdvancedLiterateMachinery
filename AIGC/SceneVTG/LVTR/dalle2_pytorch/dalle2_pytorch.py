import math
import random
from collections import namedtuple
from contextlib import contextmanager
from functools import partial, wraps
from pathlib import Path

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from kornia.filters import gaussian_blur2d
from resize_right import resize
from torch import einsum, nn
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm

from dalle2_pytorch.vqgan_vae import NullVQGanVAE, VQGanVAE

NAT = 1.0 / math.log(2.0)
UnetOutput = namedtuple("UnetOutput", ["pred", "var_interp_frac_unnormalized"])

# helper functions


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(val, length=None, validate=True):
    if isinstance(val, list):
        val = tuple(val)

    out = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length) and validate:
        assert len(out) == length

    return out


def module_device(module):
    if isinstance(module, nn.Identity):
        return "cpu"  # It doesn't matter
    return next(module.parameters()).device


def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)


@contextmanager
def null_context(*args, **kwargs):
    yield


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def is_float_dtype(dtype):
    return any(
        [
            dtype == float_dtype
            for float_dtype in (
                torch.float64,
                torch.float32,
                torch.float16,
                torch.bfloat16,
            )
        ]
    )


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


def pad_tuple_to_length(t, length, fillvalue=None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))


# checkpointing helper function


def make_checkpointable(fn, **kwargs):
    if isinstance(fn, nn.ModuleList):
        return [maybe(make_checkpointable)(el, **kwargs) for el in fn]

    condition = kwargs.pop("condition", None)

    if exists(condition) and not condition(fn):
        return fn

    @wraps(fn)
    def inner(*args):
        input_needs_grad = any(
            [isinstance(el, torch.Tensor) and el.requires_grad for el in args]
        )

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner


# for controlling freezing of CLIP


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)


def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)


def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)


# tensor helpers


def log(t, eps=1e-12):
    return torch.log(t.clamp(min=eps))


def l2norm(t):
    return F.normalize(t, dim=-1)


def resize_image_to(
    image, target_image_size, clamp_range=None, nearest=False, **kwargs
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    if not nearest:
        scale_factors = target_image_size / orig_image_size
        out = resize(image, scale_factors=scale_factors, **kwargs)
    else:
        out = F.interpolate(image, target_image_size, mode="nearest")

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out


# image normalization functions
# ddpms expect images to be in the range of -1 to 1
# but CLIP may otherwise


def normalize_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5


# clip related adapters

EmbeddedText = namedtuple("EmbedTextReturn", ["text_embed", "text_encodings"])
EmbeddedImage = namedtuple("EmbedImageReturn", ["image_embed", "image_encodings"])


class BaseClipAdapter(nn.Module):
    def __init__(self, clip, **kwargs):
        super().__init__()
        self.clip = clip
        self.overrides = kwargs

    def validate_and_resize_image(self, image):
        image_size = image.shape[-1]
        assert (
            image_size >= self.image_size
        ), f"you are passing in an image of size {image_size} but CLIP requires the image size to be at least {self.image_size}"
        return resize_image_to(image, self.image_size)

    @property
    def dim_latent(self):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def image_channels(self):
        raise NotImplementedError

    @property
    def max_text_len(self):
        raise NotImplementedError

    def embed_text(self, text):
        raise NotImplementedError

    def embed_image(self, image):
        raise NotImplementedError


# classifier free guidance functions


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# gaussian diffusion helper functions


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def meanflat(x):
    return x.mean(dim=tuple(range(1, len(x.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (
        1.0 + torch.tanh(((2.0 / math.pi) ** 0.5) * (x + 0.044715 * (x**3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres=0.999):
    assert x.shape == means.shape == log_scales.shape

    # attempting to correct nan gradients when learned variance is turned on
    # in the setting of deepspeed fp16
    eps = 1e-12 if x.dtype == torch.float32 else 1e-3

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus, eps=eps)
    log_one_minus_cdf_min = log(1.0 - cdf_min, eps=eps)
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < -thres,
        log_cdf_plus,
        torch.where(x > thres, log_one_minus_cdf_min, log(cdf_delta, eps=eps)),
    )

    return log_probs


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / first(alphas_cumprod)
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return (
        torch.linspace(
            beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float64
        )
        ** 2
    )


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(-6, 6, timesteps, dtype=torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: tuple, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim[0] * spacial_dim[1] + 1, embed_dim)
            / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x[0]


class NoiseScheduler(nn.Module):
    def __init__(
        self,
        *,
        beta_schedule,
        timesteps,
        loss_type,
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1,
    ):
        super().__init__()

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        if loss_type == "l1":
            loss_fn = F.l1_loss
        elif loss_type == "l2":
            loss_fn = F.mse_loss
        elif loss_type == "huber":
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # register buffer helper function to cast double back to float

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 loss reweighting

        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.0
        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

    def sample_random_times(self, batch):
        return torch.randint(
            0, self.num_timesteps, (batch,), device=self.betas.device, dtype=torch.long
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def calculate_v(self, x_start, t, noise=None):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def q_sample_from_to(self, x_from, from_t, to_t, noise=None):
        shape = x_from.shape
        noise = default(noise, lambda: torch.randn_like(x_from))

        alpha = extract(self.sqrt_alphas_cumprod, from_t, shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, from_t, shape)
        alpha_next = extract(self.sqrt_alphas_cumprod, to_t, shape)
        sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, to_t, shape)

        return (
            x_from * (alpha_next / alpha)
            + noise * (sigma_next * alpha - sigma * alpha_next) / alpha
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p2_reweigh_loss(self, loss, times):
        if not self.has_p2_loss_reweighting:
            return loss
        return loss * extract(self.p2_loss_weight, times, loss.shape)


# rearrange image to sequence


class RearrangeToSequence(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = rearrange(x, "b c ... -> b ... c")
        x, ps = pack([x], "b * c")

        x = self.fn(x)

        (x,) = unpack(x, ps, "b * c")
        x = rearrange(x, "b ... c -> b c ...")
        return x


# diffusion prior


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, fp16_eps=1e-3, stable=False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, fp16_eps=1e-3, stable=False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim=1, keepdim=True).detach()

        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# mlp


class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        *,
        expansion_factor=2.0,
        depth=2,
        norm=False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = lambda: nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [nn.Sequential(nn.Linear(dim_in, hidden_dim), nn.SiLU(), norm_fn())]

        for _ in range(depth - 1):
            layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), norm_fn())
            )

        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


# relative positional bias for causal transformer


class RelPosBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )
        return torch.where(is_small, n, val_if_large)

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


# feedforward


class SwiGLU(nn.Module):
    """used successfully in https://arxiv.org/abs/2204.0231"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def FeedForward(dim, mult=4, dropout=0.0, post_activation_norm=False):
    """post-activation norm https://arxiv.org/abs/2110.09456"""

    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        dropout=0.0,
        causal=False,
        rotary_emb=None,
        cosine_sim=True,
        cosine_sim_scale=16,
    ):
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else (dim_head**-0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, mask=None, attn_bias=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        q = q * self.scale

        # rotary embeddings

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(
            lambda t: repeat(t, "d -> b 1 d", b=b), self.null_kv.unbind(dim=-2)
        )
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # whether to use cosine sim

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query / key similarities

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(
                j - i + 1
            )
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_in=False,
        norm_out=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        final_proj=True,
        normformer=False,
        rotary_emb=True,
    ):
        super().__init__()
        self.init_norm = (
            LayerNorm(dim) if norm_in else nn.Identity()
        )  # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads=heads)

        rotary_emb = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            causal=True,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_emb=rotary_emb,
                        ),
                        FeedForward(
                            dim=dim,
                            mult=ff_mult,
                            dropout=ff_dropout,
                            post_activation_norm=normformer,
                        ),
                    ]
                )
            )

        self.norm = (
            LayerNorm(dim, stable=True) if norm_out else nn.Identity()
        )  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = (
            nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()
        )

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device=device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)


# decoder


def NearestUpsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )


class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(conv, nn.SiLU(), nn.PixelShuffle(2))

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


def Downsample(dim, dim_out=None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange("b c (h s1) (w s2) -> b (c s1 s2) h w", s1=2, s2=2),
        nn.Conv2d(dim * 4, dim_out, 1),
    )


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        flattened_weights = rearrange(weight, "o ... -> o (...)")

        mean = reduce(weight, "o ... -> o 1 1 1", "mean")

        var = torch.var(flattened_weights, dim=-1, unbiased=False)
        var = rearrange(var, "o -> o 1 1 1")

        weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        assert is_float_dtype(dtype), "input to sinusoidal pos emb must be a float type"

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return torch.cat((emb.sin(), emb.cos()), dim=-1).type(dtype)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, weight_standardization=False):
        super().__init__()
        conv_klass = (
            nn.Conv2d if not weight_standardization else WeightStandardizedConv2d
        )

        self.project = conv_klass(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.project(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim=None,
        time_cond_dim=None,
        groups=8,
        weight_standardization=False,
        cosine_sim_cross_attn=False,
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            self.cross_attn = CrossAttention(
                dim=dim_out, context_dim=cond_dim, cosine_sim=cosine_sim_cross_attn
            )

        self.block1 = Block(
            dim, dim_out, groups=groups, weight_standardization=weight_standardization
        )
        self.block2 = Block(
            dim_out,
            dim_out,
            groups=groups,
            weight_standardization=weight_standardization,
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        if exists(self.cross_attn):
            assert exists(cond)

            h = rearrange(h, "b c ... -> b ... c")
            h, ps = pack([h], "b * c")

            h = self.cross_attn(h, context=cond) + h

            (h,) = unpack(h, ps, "b * c")
            h = rearrange(h, "b ... c -> b c ...")

        h = self.block2(h)
        return h + self.res_conv(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        dropout=0.0,
        norm_context=False,
        cosine_sim=False,
        cosine_sim_scale=16,
    ):
        super().__init__()
        self.cosine_sim = cosine_sim
        self.scale = cosine_sim_scale if cosine_sim else (dim_head**-0.5)
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(
            lambda t: repeat(t, "d -> b h 1 d", h=self.heads, b=b),
            self.null_kv.unbind(dim=-2),
        )

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head=32, heads=8, **kwargs):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.GELU()
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias=False), ChanLayerNorm(dim)
        )

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        seq_len = x * y

        fmap = self.norm(fmap)
        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> (b h) (x y) c", h=h), (q, k, v)
        )

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale
        v = l2norm(v)

        k, v = map(lambda t: t / math.sqrt(seq_len), (k, v))

        context = einsum("b n d, b n e -> b d e", k, v)
        out = einsum("b n d, b d e -> b n e", q, context)
        out = rearrange(out, "(b h) (x y) d -> b (h d) x y", h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)


class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, kernel_sizes, dim_out=None, stride=2):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(
                    dim_in,
                    dim_scale,
                    kernel,
                    stride=stride,
                    padding=(kernel - stride) // 2,
                )
            )

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


class UpsampleCombiner(nn.Module):
    def __init__(self, dim, *, enabled=False, dim_ins=tuple(), dim_outs=tuple()):
        super().__init__()
        assert len(dim_ins) == len(dim_outs)
        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList(
            [Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)]
        )
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps=None):
        target_size = x.shape[-1]

        fmaps = default(fmaps, tuple())

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim=1)


class Unet(nn.Module):
    def __init__(
        self,
        *,
        dim=128,
        cond_dim=128,
        image_embed_dim=512,
        text_embed_dim=512,
        num_image_tokens=4,
        num_time_tokens=2,
        out_dim=None,
        dim_mults=(1, 2, 2, 4, 4, 8),
        channels=3,
        channels_out=None,
        self_attn=False,
        attn_dim_head=32,
        attn_heads=16,
        lowres_cond=False,  # for cascading diffusion - https://cascaded-diffusion.github.io/
        lowres_noise_cond=False,  # for conditioning on low resolution noising, based on Imagen
        self_cond=False,  # set this to True to use the self-conditioning technique from - https://arxiv.org/abs/2208.04202
        sparse_attn=False,
        cosine_sim_cross_attn=False,
        cosine_sim_self_attn=False,
        attend_at_middle=True,  # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        cond_on_text_encodings=True,
        max_text_len=128,
        cond_on_image_embeds=False,
        add_image_embeds_to_time=True,  # alerted by @mhh0318 to a phrase in the paper - "Specifically, we modify the architecture described in Nichol et al. (2021) by projecting and adding CLIP embeddings to the existing timestep embedding"
        init_dim=None,
        init_conv_kernel_size=7,
        resnet_groups=8,
        resnet_weight_standardization=False,
        num_resnet_blocks=2,
        init_cross_embed=True,
        init_cross_embed_kernel_sizes=(3, 7, 15),
        cross_embed_downsample=False,
        cross_embed_downsample_kernel_sizes=(2, 4),
        memory_efficient=False,
        scale_skip_connection=False,
        pixel_shuffle_upsample=True,
        final_conv_kernel_size=1,
        combine_upsample_fmaps=False,  # whether to combine the outputs of all upsample blocks, as in unet squared paper
        checkpoint_during_training=False,
        init_with_removal=True,
        init_with_linepolymask=True,
        init_with_wordpolymask=False,
        **kwargs,
    ):
        super().__init__()
        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        del self._locals["self"]
        del self._locals["__class__"]

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one
        self.unnormalize_img = unnormalize_zero_to_one

        # for eventual cascading diffusion

        self.lowres_cond = lowres_cond

        # whether to do self conditioning

        self.self_cond = self_cond

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # initial number of channels depends on
        # (1) low resolution conditioning from cascading ddpm paper, conditioned on previous unet output in the cascade
        # (2) self conditioning (bit diffusion paper)

        init_channels = channels * (1 + int(lowres_cond) + int(self_cond))

        init_dim = default(init_dim, dim)

        if init_with_removal:
            init_channels += channels
        if init_with_linepolymask:
            init_channels += 1
        if init_with_wordpolymask:
            init_channels += 1

        self.init_with_removal = init_with_removal
        self.init_with_linepolymask = init_with_linepolymask
        self.init_with_wordpolymask = init_with_wordpolymask

        # char embedding layer
        self.word_embedding = nn.Embedding(128, 8)

        self.init_conv = (
            CrossEmbedLayer(
                init_channels,
                dim_out=init_dim,
                kernel_sizes=init_cross_embed_kernel_sizes,
                stride=1,
            )
            if init_cross_embed
            else nn.Conv2d(
                init_channels,
                init_dim,
                init_conv_kernel_size,
                padding=init_conv_kernel_size // 2,
            )
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        num_stages = len(in_out)

        # time, image embeddings, and optional text encoding

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4

        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, time_cond_dim), nn.GELU()
        )

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange("b (r d) -> b r d", r=num_time_tokens),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(time_cond_dim, time_cond_dim))

        self.image_to_tokens = (
            nn.Sequential(
                nn.Linear(image_embed_dim, cond_dim * num_image_tokens),
                Rearrange("b (n d) -> b n d", n=num_image_tokens),
            )
            if cond_on_image_embeds and image_embed_dim != cond_dim
            else nn.Identity()
        )

        self.to_image_hiddens = (
            nn.Sequential(nn.Linear(image_embed_dim, time_cond_dim), nn.GELU())
            if cond_on_image_embeds and add_image_embeds_to_time
            else None
        )

        self.norm_cond = nn.LayerNorm(cond_dim)
        self.norm_mid_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None
        self.text_embed_dim = None

        if cond_on_text_encodings:
            assert exists(
                text_embed_dim
            ), "text_embed_dim must be given to the unet if cond_on_text_encodings is True"
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)
            self.text_embed_dim = text_embed_dim

        # low resolution noise conditiong, based on Imagen's upsampler training technique

        self.lowres_noise_cond = lowres_noise_cond

        self.to_lowres_noise_cond = (
            nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_cond_dim),
                nn.GELU(),
                nn.Linear(time_cond_dim, time_cond_dim),
            )
            if lowres_noise_cond
            else None
        )

        # finer control over whether to condition on image embeddings and text encodings
        # so one can have the latter unets in the cascading DDPMs only focus on super-resoluting

        self.cond_on_text_encodings = cond_on_text_encodings
        self.cond_on_image_embeds = cond_on_image_embeds

        # for classifier free guidance

        self.null_image_embed = nn.Parameter(torch.randn(1, num_image_tokens, cond_dim))
        self.null_image_hiddens = nn.Parameter(torch.randn(1, time_cond_dim))
        self.null_imagestyle_embed = nn.Parameter(
            torch.randn(1, num_image_tokens, cond_dim)
        )

        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))

        # whether to scale skip connection, adopted in Imagen

        self.skip_connect_scale = 1.0 if not scale_skip_connection else (2**-0.5)

        # attention related params

        attn_kwargs = dict(
            heads=attn_heads, dim_head=attn_dim_head, cosine_sim=cosine_sim_self_attn
        )

        self_attn = cast_tuple(self_attn, num_stages)

        create_self_attn = lambda dim: RearrangeToSequence(
            Residual(Attention(dim, **attn_kwargs))
        )

        # resnet block klass

        resnet_groups = cast_tuple(resnet_groups, num_stages)
        top_level_resnet_group = first(resnet_groups)

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_stages)

        # downsample klass

        downsample_klass = Downsample
        if cross_embed_downsample:
            downsample_klass = partial(
                CrossEmbedLayer, kernel_sizes=cross_embed_downsample_kernel_sizes
            )

        # upsample klass

        upsample_klass = (
            NearestUpsample if not pixel_shuffle_upsample else PixelShuffleUpsample
        )

        # prepare resnet klass

        resnet_block = partial(
            ResnetBlock,
            cosine_sim_cross_attn=cosine_sim_cross_attn,
            weight_standardization=resnet_weight_standardization,
        )

        # give memory efficient unet an initial resnet block

        self.init_resnet_block = (
            resnet_block(
                init_dim,
                init_dim,
                time_cond_dim=time_cond_dim,
                groups=top_level_resnet_group,
            )
            if memory_efficient
            else None
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        skip_connect_dims = []  # keeping track of skip connection dimensions
        upsample_combiner_dims = (
            []
        )  # keeping track of dimensions for final upsample feature map combiner

        for ind, (
            (dim_in, dim_out),
            groups,
            layer_num_resnet_blocks,
            layer_self_attn,
        ) in enumerate(zip(in_out, resnet_groups, num_resnet_blocks, self_attn)):
            is_first = ind == 0
            is_last = ind >= (num_resolutions - 1)
            layer_cond_dim = cond_dim if not is_first else None

            dim_layer = dim_out if memory_efficient else dim_in
            skip_connect_dims.append(dim_layer)

            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_layer)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_layer, **attn_kwargs))

            self.downs.append(
                nn.ModuleList(
                    [
                        downsample_klass(dim_in, dim_out=dim_out)
                        if memory_efficient
                        else None,
                        resnet_block(
                            dim_layer,
                            dim_layer,
                            time_cond_dim=time_cond_dim,
                            groups=groups,
                        ),
                        nn.ModuleList(
                            [
                                resnet_block(
                                    dim_layer,
                                    dim_layer,
                                    cond_dim=layer_cond_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups,
                                )
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        attention,
                        downsample_klass(dim_layer, dim_out=dim_out)
                        if not is_last and not memory_efficient
                        else nn.Conv2d(dim_layer, dim_out, 1),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = resnet_block(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )
        self.mid_attn = create_self_attn(mid_dim)
        self.mid_block2 = resnet_block(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )

        for ind, (
            (dim_in, dim_out),
            groups,
            layer_num_resnet_blocks,
            layer_self_attn,
        ) in enumerate(
            zip(
                reversed(in_out),
                reversed(resnet_groups),
                reversed(num_resnet_blocks),
                reversed(self_attn),
            )
        ):
            is_last = ind >= (len(in_out) - 1)
            layer_cond_dim = cond_dim if not is_last else None

            skip_connect_dim = skip_connect_dims.pop()

            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_out)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_out, **attn_kwargs))

            upsample_combiner_dims.append(dim_out)

            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_block(
                            dim_out + skip_connect_dim,
                            dim_out,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups,
                        ),
                        nn.ModuleList(
                            [
                                resnet_block(
                                    dim_out + skip_connect_dim,
                                    dim_out,
                                    cond_dim=layer_cond_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups,
                                )
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        attention,
                        upsample_klass(dim_out, dim_in)
                        if not is_last or memory_efficient
                        else nn.Identity(),
                    ]
                )
            )

        # whether to combine outputs from all upsample blocks for final resnet block

        self.upsample_combiner = UpsampleCombiner(
            dim=dim,
            enabled=combine_upsample_fmaps,
            dim_ins=upsample_combiner_dims,
            dim_outs=(dim,) * len(upsample_combiner_dims),
        )

        # a final resnet block

        self.final_resnet_block = resnet_block(
            self.upsample_combiner.dim_out + dim,
            dim,
            time_cond_dim=time_cond_dim,
            groups=top_level_resnet_group,
        )

        out_dim_in = dim + (channels if lowres_cond else 0)

        self.to_out = nn.Conv2d(
            out_dim_in,
            self.channels_out,
            kernel_size=final_conv_kernel_size,
            padding=final_conv_kernel_size // 2,
        )

        zero_init_(self.to_out)  # since both OpenAI and @crowsonkb are doing it

        # whether to checkpoint during training

        self.checkpoint_during_training = checkpoint_during_training

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        lowres_noise_cond,
        channels,
        channels_out,
        cond_on_image_embeds,
        cond_on_text_encodings,
    ):
        if (
            lowres_cond == self.lowres_cond
            and channels == self.channels
            and cond_on_image_embeds == self.cond_on_image_embeds
            and cond_on_text_encodings == self.cond_on_text_encodings
            and lowres_noise_cond == self.lowres_noise_cond
            and channels_out == self.channels_out
        ):
            return self

        updated_kwargs = dict(
            lowres_cond=lowres_cond,
            channels=channels,
            channels_out=channels_out,
            cond_on_image_embeds=cond_on_image_embeds,
            cond_on_text_encodings=cond_on_text_encodings,
            lowres_noise_cond=lowres_noise_cond,
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        kwargs["drop_prob"] = {
            "textstring": 1.0,
            "renderimage": 1.0,
        }

        null_logits = self.forward(*args, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        *,
        image_embed,
        all_conditions=None,
        lowres_cond_img=None,
        lowres_noise_level=None,
        text_encodings=None,
        drop_prob=None,
        blur_sigma=None,
        blur_kernel_size=None,
        disable_checkpoint=False,
        self_cond=None,
    ):
        batch_size, device = x.shape[0], x.device

        # add low resolution conditioning, if present

        assert not (
            self.lowres_cond and not exists(lowres_cond_img)
        ), "low resolution conditioning image must be present"

        # concat self conditioning, if needed

        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim=1)

        # concat low resolution conditioning

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim=1)

        if self.init_with_removal:
            assert (
                "images_removal" in all_conditions
            ), "images_removal is not in all_conditions"
            img_removal = all_conditions["images_removal"]
            img_removal = self.normalize_img(img_removal)
            x = torch.cat((x, img_removal), dim=1)

        if self.init_with_linepolymask:
            assert (
                "line_poly_mask" in all_conditions
            ), "line_poly_mask is not in all_conditions"
            line_poly_mask = all_conditions["line_poly_mask"]
            line_poly_mask = self.normalize_img(line_poly_mask)
            x = torch.cat((x, line_poly_mask), dim=1)

        if self.init_with_wordpolymask:
            assert (
                "word_poly_mask" in all_conditions
            ), "word_poly_mask is not in all_conditions"
            word_poly_mask = all_conditions["word_poly_mask"]
            word_poly_mask = self.normalize_img(word_poly_mask)
            x = torch.cat((x, word_poly_mask), dim=1)

        # initial convolution

        x = self.init_conv(x)
        r = x.clone()  # final residual

        # time conditioning

        time = time.type_as(x)
        time_hiddens = self.to_time_hiddens(time)

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # low res noise conditioning (similar to time above)

        if exists(lowres_noise_level):
            assert exists(
                self.to_lowres_noise_cond
            ), "lowres_noise_cond must be set to True on instantiation of the unet in order to conditiong on lowres noise"
            lowres_noise_level = lowres_noise_level.type_as(x)
            t = t + self.to_lowres_noise_cond(lowres_noise_level)

        # conditional dropout
        image_keep_mask = prob_mask_like(
            (batch_size,), 1 - drop_prob["renderimage"], device=device
        )
        text_keep_mask = prob_mask_like(
            (batch_size,), 1 - drop_prob["textstring"], device=device
        )

        text_keep_mask = rearrange(text_keep_mask, "b -> b 1 1")

        # image embedding to be summed to time embedding
        # discovered by @mhh0318 in the paper

        if exists(image_embed) and exists(self.to_image_hiddens):
            image_hiddens = self.to_image_hiddens(image_embed)
            image_keep_mask_hidden = rearrange(image_keep_mask, "b -> b 1")
            null_image_hiddens = self.null_image_hiddens.to(image_hiddens.dtype)

            image_hiddens = torch.where(
                image_keep_mask_hidden, image_hiddens, null_image_hiddens
            )

            t = t + image_hiddens

        # mask out image embedding depending on condition dropout
        # for classifier free guidance

        image_tokens = None

        if self.cond_on_image_embeds:
            image_keep_mask_embed = rearrange(image_keep_mask, "b -> b 1 1")
            image_tokens = self.image_to_tokens(image_embed)
            null_image_embed = self.null_image_embed.to(
                image_tokens.dtype
            )  # for some reason pytorch AMP not working

            image_tokens = torch.where(
                image_keep_mask_embed, image_tokens, null_image_embed
            )

        # take care of text encodings (optional)

        text_tokens = None

        if exists(text_encodings) and self.cond_on_text_encodings:
            assert (
                text_encodings.shape[0] == batch_size
            ), f"the text encodings being passed into the unet does not have the proper batch size - text encoding shape {text_encodings.shape} - required batch size is {batch_size}"
            assert (
                self.text_embed_dim == text_encodings.shape[-1]
            ), f"the text encodings you are passing in have a dimension of {text_encodings.shape[-1]}, but the unet was created with text_embed_dim of {self.text_embed_dim}."

            text_mask = torch.any(text_encodings != 0.0, dim=-1)

            text_tokens = self.text_to_cond(text_encodings)

            text_tokens = text_tokens[:, : self.max_text_len]
            text_mask = text_mask[:, : self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))
                text_mask = F.pad(text_mask, (0, remainder), value=False)

            text_mask = rearrange(text_mask, "b n -> b n 1")

            assert (
                text_mask.shape[0] == text_keep_mask.shape[0]
            ), f"text_mask has shape of {text_mask.shape} while text_keep_mask has shape {text_keep_mask.shape}. text encoding is of shape {text_encodings.shape}"
            text_keep_mask = text_mask & text_keep_mask

            null_text_embed = self.null_text_embed.to(
                text_tokens.dtype
            )  # for some reason pytorch AMP not working

            text_tokens = torch.where(text_keep_mask, text_tokens, null_text_embed)

        # main conditioning tokens (c)

        c = time_tokens

        if exists(image_tokens):
            c = torch.cat((c, image_tokens), dim=-2)

        # text and image conditioning tokens (mid_c)
        # to save on compute, only do cross attention based conditioning on the inner most layers of the Unet

        mid_c = c if not exists(text_tokens) else torch.cat((c, text_tokens), dim=-2)

        # normalize conditioning tokens

        c = self.norm_cond(c)
        mid_c = self.norm_mid_cond(mid_c)

        # gradient checkpointing

        can_checkpoint = (
            self.training and self.checkpoint_during_training and not disable_checkpoint
        )
        apply_checkpoint_fn = make_checkpointable if can_checkpoint else identity

        # make checkpointable modules

        init_resnet_block, mid_block1, mid_attn, mid_block2, final_resnet_block = [
            maybe(apply_checkpoint_fn)(module)
            for module in (
                self.init_resnet_block,
                self.mid_block1,
                self.mid_attn,
                self.mid_block2,
                self.final_resnet_block,
            )
        ]

        can_checkpoint_cond = lambda m: isinstance(m, ResnetBlock)
        downs, ups = [
            maybe(apply_checkpoint_fn)(m, condition=can_checkpoint_cond)
            for m in (self.downs, self.ups)
        ]

        # initial resnet block

        if exists(init_resnet_block):
            x = init_resnet_block(x, t)

        # go through the layers of the unet, down and up

        down_hiddens = []
        up_hiddens = []

        for pre_downsample, init_block, resnet_blocks, attn, post_downsample in downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t, c)
                down_hiddens.append(x.contiguous())

            x = attn(x)
            down_hiddens.append(x.contiguous())

            if exists(post_downsample):
                x = post_downsample(x)

        x = mid_block1(x, t, mid_c)

        if exists(mid_attn):
            x = mid_attn(x)

        x = mid_block2(x, t, mid_c)

        connect_skip = lambda fmap: torch.cat(
            (fmap, down_hiddens.pop() * self.skip_connect_scale), dim=1
        )

        for init_block, resnet_blocks, attn, upsample in ups:
            x = connect_skip(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = connect_skip(x)
                x = resnet_block(x, t, c)

            x = attn(x)

            up_hiddens.append(x.contiguous())
            x = upsample(x)

        x = self.upsample_combiner(x, up_hiddens)

        x = torch.cat((x, r), dim=1)

        x = final_resnet_block(x, t)

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim=1)

        return self.to_out(x)


class LowresConditioner(nn.Module):
    def __init__(
        self,
        downsample_first=True,
        use_blur=True,
        blur_prob=0.5,
        blur_sigma=0.6,
        blur_kernel_size=3,
        use_noise=False,
        input_image_range=None,
        normalize_img_fn=identity,
        unnormalize_img_fn=identity,
    ):
        super().__init__()
        self.downsample_first = downsample_first
        self.input_image_range = input_image_range

        self.use_blur = use_blur
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        self.blur_kernel_size = blur_kernel_size

        self.use_noise = use_noise
        self.normalize_img = normalize_img_fn
        self.unnormalize_img = unnormalize_img_fn
        self.noise_scheduler = (
            NoiseScheduler(beta_schedule="linear", timesteps=1000, loss_type="l2")
            if use_noise
            else None
        )

    def noise_image(self, cond_fmap, noise_levels=None):
        assert exists(self.noise_scheduler)

        batch = cond_fmap.shape[0]
        cond_fmap = self.normalize_img(cond_fmap)

        random_noise_levels = default(
            noise_levels, lambda: self.noise_scheduler.sample_random_times(batch)
        )
        cond_fmap = self.noise_scheduler.q_sample(
            cond_fmap, t=random_noise_levels, noise=torch.randn_like(cond_fmap)
        )

        cond_fmap = self.unnormalize_img(cond_fmap)
        return cond_fmap, random_noise_levels

    def forward(
        self,
        cond_fmap,
        *,
        target_image_size,
        downsample_image_size=None,
        should_blur=True,
        blur_sigma=None,
        blur_kernel_size=None,
    ):
        if self.downsample_first and exists(downsample_image_size):
            cond_fmap = resize_image_to(
                cond_fmap,
                downsample_image_size,
                clamp_range=self.input_image_range,
                nearest=True,
            )

        # blur is only applied 50% of the time
        # section 3.1 in https://arxiv.org/abs/2106.15282

        if self.use_blur and should_blur and random.random() < self.blur_prob:

            # when training, blur the low resolution conditional image

            blur_sigma = default(blur_sigma, self.blur_sigma)
            blur_kernel_size = default(blur_kernel_size, self.blur_kernel_size)

            # allow for drawing a random sigma between lo and hi float values

            if isinstance(blur_sigma, tuple):
                blur_sigma = tuple(map(float, blur_sigma))
                blur_sigma = random.uniform(*blur_sigma)

            # allow for drawing a random kernel size between lo and hi int values

            if isinstance(blur_kernel_size, tuple):
                blur_kernel_size = tuple(map(int, blur_kernel_size))
                kernel_size_lo, kernel_size_hi = blur_kernel_size
                blur_kernel_size = random.randrange(kernel_size_lo, kernel_size_hi + 1)

            cond_fmap = gaussian_blur2d(
                cond_fmap, cast_tuple(blur_kernel_size, 2), cast_tuple(blur_sigma, 2)
            )

        # resize to target image size

        cond_fmap = resize_image_to(
            cond_fmap,
            target_image_size,
            clamp_range=self.input_image_range,
            nearest=True,
        )

        # noise conditioning, as done in Imagen
        # as a replacement for the BSR noising, and potentially replace blurring for first stage too

        random_noise_levels = None

        if self.use_noise:
            cond_fmap, random_noise_levels = self.noise_image(cond_fmap)

        # return conditioning feature map, as well as the augmentation noise levels

        return cond_fmap, random_noise_levels


class Decoder(nn.Module):
    def __init__(
        self,
        unet,
        *,
        clip=None,
        recognizer=None,
        image_size=(64, 512),
        image_pos_len=128,
        text_pos_len=128,
        image_cus_emb=512,
        text_pos_emb=512,
        channels=3,
        vae=tuple(),
        timesteps=1000,
        sample_timesteps=None,
        drop_prob=dict(),
        loss_type="l2",
        beta_schedule=None,
        predict_x_start=False,
        predict_v=False,
        predict_x_start_for_latent_diffusion=False,
        image_sizes=None,  # for cascading ddpm, image size at each stage
        random_crop_sizes=None,  # whether to random crop the image at that stage in the cascade (super resoluting convolutions at the end may be able to generalize on smaller crops)
        use_noise_for_lowres_cond=False,  # whether to use Imagen-like noising for low resolution conditioning
        use_blur_for_lowres_cond=True,  # whether to use the blur conditioning used in the original cascading ddpm paper, as well as DALL-E2
        lowres_downsample_first=True,  # cascading ddpm - resizes to lower resolution, then to next conditional resolution + blur
        blur_prob=0.5,  # cascading ddpm - when training, the gaussian blur is only applied 50% of the time
        blur_sigma=0.6,  # cascading ddpm - blur sigma
        blur_kernel_size=3,  # cascading ddpm - blur kernel size
        lowres_noise_sample_level=0.2,  # in imagen paper, they use a 0.2 noise level at sample time for low resolution conditioning
        clip_denoised=True,
        clip_x_start=True,
        clip_adapter_overrides=dict(),
        learned_variance=True,
        learned_variance_constrain_frac=False,
        vb_loss_weight=0.001,
        unconditional=False,  # set to True for generating images without conditioning
        auto_normalize_img=True,  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.95,
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        ddim_sampling_eta=0.0,  # can be set to 0. for deterministic sampling afaict
    ):
        super().__init__()

        # clip

        self.clip = None
        if exists(clip):
            assert (
                not unconditional
            ), "clip must not be given if doing unconditional image training"
            assert (
                channels == clip.image_channels
            ), f"channels of image ({channels}) should be equal to the channels that CLIP accepts ({clip.image_channels})"

            if isinstance(clip, CLIP):
                clip = XClipAdapter(clip, **clip_adapter_overrides)
            elif isinstance(clip, CoCa):
                clip = CoCaAdapter(clip, **clip_adapter_overrides)

            freeze_model_and_make_eval_(clip)
            assert isinstance(clip, BaseClipAdapter)

            self.clip = clip

        # determine image size, with image_size and image_sizes taking precedence

        if exists(image_size) or exists(image_sizes):
            assert exists(image_size) ^ exists(
                image_sizes
            ), "only one of image_size or image_sizes must be given"
            image_size = default(image_size, lambda: image_sizes[-1])
        elif exists(clip):
            image_size = clip.image_size
        else:
            raise Error(
                "either image_size, image_sizes, or clip must be given to decoder"
            )

        # channels

        self.channels = channels

        self.recognizer = None
        if exists(recognizer):
            freeze_model_and_make_eval_(recognizer)
            self.recognizer = recognizer

        self.abs_pos_emb_text = nn.Embedding(text_pos_len, text_pos_emb)

        # to customize image features

        self.attn_pooling = AttentionPool2d(
            spacial_dim=(1, image_pos_len),
            embed_dim=image_cus_emb,
            num_heads=4,
            output_dim=image_cus_emb,
        )
        self.attn_pooling_style = AttentionPool2d(
            spacial_dim=(1, image_pos_len),
            embed_dim=image_cus_emb,
            num_heads=4,
            output_dim=image_cus_emb,
        )

        # normalize and unnormalize image functions

        self.normalize_img = (
            normalize_neg_one_to_one if auto_normalize_img else identity
        )
        self.unnormalize_img = (
            unnormalize_zero_to_one if auto_normalize_img else identity
        )

        # verify conditioning method

        unets = cast_tuple(unet)
        num_unets = len(unets)
        self.num_unets = num_unets

        self.unconditional = unconditional

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        vaes = pad_tuple_to_length(
            cast_tuple(vae), len(unets), fillvalue=NullVQGanVAE(channels=self.channels)
        )

        # whether to use learned variance, defaults to True for the first unet in the cascade, as in paper

        learned_variance = pad_tuple_to_length(
            cast_tuple(learned_variance), len(unets), fillvalue=False
        )
        self.learned_variance = learned_variance
        self.learned_variance_constrain_frac = learned_variance_constrain_frac  # whether to constrain the output of the network (the interpolation fraction) from 0 to 1
        self.vb_loss_weight = vb_loss_weight

        # default and validate conditioning parameters

        use_noise_for_lowres_cond = cast_tuple(
            use_noise_for_lowres_cond, num_unets - 1, validate=False
        )
        use_blur_for_lowres_cond = cast_tuple(
            use_blur_for_lowres_cond, num_unets - 1, validate=False
        )

        if len(use_noise_for_lowres_cond) < num_unets:
            use_noise_for_lowres_cond = (False, *use_noise_for_lowres_cond)

        if len(use_blur_for_lowres_cond) < num_unets:
            use_blur_for_lowres_cond = (False, *use_blur_for_lowres_cond)

        assert not use_noise_for_lowres_cond[
            0
        ], "first unet will never need low res noise conditioning"
        assert not use_blur_for_lowres_cond[
            0
        ], "first unet will never need low res blur conditioning"

        assert num_unets == 1 or all(
            (use_noise or use_blur)
            for use_noise, use_blur in zip(
                use_noise_for_lowres_cond[1:], use_blur_for_lowres_cond[1:]
            )
        )

        # construct unets and vaes

        self.unets = nn.ModuleList([])
        self.vaes = nn.ModuleList([])

        for ind, (
            one_unet,
            one_vae,
            one_unet_learned_var,
            lowres_noise_cond,
        ) in enumerate(zip(unets, vaes, learned_variance, use_noise_for_lowres_cond)):
            assert isinstance(one_unet, Unet)
            assert isinstance(one_vae, (VQGanVAE, NullVQGanVAE))

            is_first = ind == 0
            latent_dim = one_vae.encoded_dim if exists(one_vae) else None

            unet_channels = default(latent_dim, self.channels)
            unet_channels_out = unet_channels * (1 if not one_unet_learned_var else 2)

            one_unet = one_unet.cast_model_parameters(
                lowres_cond=not is_first,
                lowres_noise_cond=lowres_noise_cond,
                cond_on_image_embeds=not unconditional and is_first,
                cond_on_text_encodings=not unconditional
                and one_unet.cond_on_text_encodings,
                channels=unet_channels,
                channels_out=unet_channels_out,
            )

            self.unets.append(one_unet)
            self.vaes.append(one_vae.copy_for_eval())

        # sampling timesteps, defaults to non-ddim with full timesteps sampling

        self.sample_timesteps = cast_tuple(sample_timesteps, num_unets)
        self.ddim_sampling_eta = ddim_sampling_eta

        # create noise schedulers per unet

        if not exists(beta_schedule):
            beta_schedule = (
                "cosine",
                *(("cosine",) * max(num_unets - 2, 0)),
                *(("linear",) * int(num_unets > 1)),
            )

        beta_schedule = cast_tuple(beta_schedule, num_unets)
        p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        self.noise_schedulers = nn.ModuleList([])

        for ind, (
            unet_beta_schedule,
            unet_p2_loss_weight_gamma,
            sample_timesteps,
        ) in enumerate(zip(beta_schedule, p2_loss_weight_gamma, self.sample_timesteps)):
            assert (
                not exists(sample_timesteps) or sample_timesteps <= timesteps
            ), f"sampling timesteps {sample_timesteps} must be less than or equal to the number of training timesteps {timesteps} for unet {ind + 1}"

            noise_scheduler = NoiseScheduler(
                beta_schedule=unet_beta_schedule,
                timesteps=timesteps,
                loss_type=loss_type,
                p2_loss_weight_gamma=unet_p2_loss_weight_gamma,
                p2_loss_weight_k=p2_loss_weight_k,
            )

            self.noise_schedulers.append(noise_scheduler)

        # unet image sizes

        image_sizes = default(image_sizes, (image_size,))
        image_sizes = tuple(sorted(set(image_sizes)))

        assert self.num_unets == len(
            image_sizes
        ), f"you did not supply the correct number of u-nets ({self.num_unets}) for resolutions {image_sizes}"
        self.image_sizes = image_sizes
        self.sample_channels = cast_tuple(self.channels, len(image_sizes))

        # random crop sizes (for super-resoluting unets at the end of cascade?)

        self.random_crop_sizes = cast_tuple(random_crop_sizes, len(image_sizes))
        assert not exists(
            self.random_crop_sizes[0]
        ), "you would not need to randomly crop the image for the base unet"

        # predict x0 config

        self.predict_x_start = (
            cast_tuple(predict_x_start, len(unets))
            if not predict_x_start_for_latent_diffusion
            else tuple(map(lambda t: isinstance(t, VQGanVAE), self.vaes))
        )

        # predict v

        self.predict_v = cast_tuple(predict_v, len(unets))

        # input image range

        self.input_image_range = (-1.0 if not auto_normalize_img else 0.0, 1.0)

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (
            False,
            *((True,) * (num_unets - 1)),
        ), "the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True"

        self.lowres_conds = nn.ModuleList([])

        for unet_index, use_noise, use_blur in zip(
            range(num_unets), use_noise_for_lowres_cond, use_blur_for_lowres_cond
        ):
            if unet_index == 0:
                self.lowres_conds.append(None)
                continue

            lowres_cond = LowresConditioner(
                downsample_first=lowres_downsample_first,
                use_blur=use_blur,
                use_noise=use_noise,
                blur_prob=blur_prob,
                blur_sigma=blur_sigma,
                blur_kernel_size=blur_kernel_size,
                input_image_range=self.input_image_range,
                normalize_img_fn=self.normalize_img,
                unnormalize_img_fn=self.unnormalize_img,
            )

            self.lowres_conds.append(lowres_cond)

        self.lowres_noise_sample_level = lowres_noise_sample_level

        # classifier free guidance

        self.can_classifier_guidance = False
        for d_v in drop_prob.values():
            if d_v > 0.0:
                self.can_classifier_guidance = True
                break
        self.drop_prob = drop_prob
        self.drop_prob_sample = {
            "textstring": 1.0,
            "renderimage": 1.0,
        }

        # whether to clip when sampling

        self.clip_denoised = clip_denoised
        self.clip_x_start = clip_x_start

        # dynamic thresholding settings, if clipping denoised during sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

        # device tracker

        self.register_buffer("_dummy", torch.Tensor([True]), persistent=False)

    def reloadrecognizer(self, recognizer):
        print("pre decoders recognizer params")
        print(self.recognizer.C.weight.sum())
        freeze_model_and_make_eval_(recognizer)
        self.recognizer = recognizer
        print("post decoders recognizer params")
        print(self.recognizer.C.weight.sum())

    @property
    def device(self):
        return self._dummy.device

    @property
    def condition_on_text_encodings(self):
        return any(
            [
                unet.cond_on_text_encodings
                for unet in self.unets
                if isinstance(unet, Unet)
            ]
        )

    def get_unet(self, unet_number):
        assert 0 < unet_number <= self.num_unets
        index = unet_number - 1
        return self.unets[index]

    def parse_unet_output(self, learned_variance, output):
        var_interp_frac_unnormalized = None

        if learned_variance:
            output, var_interp_frac_unnormalized = output.chunk(2, dim=1)

        return UnetOutput(output, var_interp_frac_unnormalized)

    @contextmanager
    def one_unet_in_gpu(self, unet_number=None, unet=None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.get_unet(unet_number)

        # devices

        cuda, cpu = torch.device("cuda"), torch.device("cpu")

        self.cuda()

        devices = [module_device(unet) for unet in self.unets]

        self.unets.to(cpu)
        unet.to(cuda)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    def dynamic_threshold(self, x):
        """proposed in https://arxiv.org/abs/2205.11487 as an improved clamping in the setting of classifier free guidance"""

        # s is the threshold amount
        # static thresholding would just be s = 1
        s = 1.0
        if self.use_dynamic_thres:
            s = torch.quantile(
                rearrange(x, "b ... -> b (...)").abs(),
                self.dynamic_thres_percentile,
                dim=-1,
            )

            s.clamp_(min=1.0)
            s = s.view(-1, *((1,) * (x.ndim - 1)))

        # clip by threshold, depending on whether static or dynamic
        x = x.clamp(-s, s) / s
        return x

    def p_mean_variance(
        self,
        unet,
        x,
        t,
        image_embed,
        noise_scheduler,
        all_conditions=None,
        drop_prob=None,
        text_encodings=None,
        lowres_cond_img=None,
        self_cond=None,
        clip_denoised=True,
        predict_x_start=False,
        predict_v=False,
        learned_variance=False,
        cond_scale=1.0,
        model_output=None,
        lowres_noise_level=None,
    ):
        assert not (
            cond_scale != 1.0 and not self.can_classifier_guidance
        ), "the decoder was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)"

        model_output = default(
            model_output,
            lambda: unet.forward_with_cond_scale(
                x,
                t,
                all_conditions=all_conditions,
                drop_prob=drop_prob,
                image_embed=image_embed,
                text_encodings=text_encodings,
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
                self_cond=self_cond,
                lowres_noise_level=lowres_noise_level,
            ),
        )

        pred, var_interp_frac_unnormalized = self.parse_unet_output(
            learned_variance, model_output
        )

        if predict_v:
            x_start = noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        elif predict_x_start:
            x_start = pred
        else:
            x_start = noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        if clip_denoised:
            x_start = self.dynamic_threshold(x_start)

        (
            model_mean,
            posterior_variance,
            posterior_log_variance,
        ) = noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)

        if learned_variance:
            # if learned variance, posterio variance and posterior log variance are predicted by the network
            # by an interpolation of the max and min log beta values
            # eq 15 - https://arxiv.org/abs/2102.09672
            min_log = extract(
                noise_scheduler.posterior_log_variance_clipped, t, x.shape
            )
            max_log = extract(torch.log(noise_scheduler.betas), t, x.shape)
            var_interp_frac = unnormalize_zero_to_one(var_interp_frac_unnormalized)

            if self.learned_variance_constrain_frac:
                var_interp_frac = var_interp_frac.sigmoid()

            posterior_log_variance = (
                var_interp_frac * max_log + (1 - var_interp_frac) * min_log
            )
            posterior_variance = posterior_log_variance.exp()

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        unet,
        x,
        t,
        image_embed,
        noise_scheduler,
        all_conditions=None,
        drop_prob=None,
        text_encodings=None,
        cond_scale=1.0,
        lowres_cond_img=None,
        self_cond=None,
        predict_x_start=False,
        predict_v=False,
        learned_variance=False,
        clip_denoised=True,
        lowres_noise_level=None,
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            unet,
            x=x,
            t=t,
            all_conditions=all_conditions,
            drop_prob=drop_prob,
            image_embed=image_embed,
            text_encodings=text_encodings,
            cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img,
            self_cond=self_cond,
            clip_denoised=clip_denoised,
            predict_x_start=predict_x_start,
            predict_v=predict_v,
            noise_scheduler=noise_scheduler,
            learned_variance=learned_variance,
            lowres_noise_level=lowres_noise_level,
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(
        self,
        unet,
        shape,
        image_embed,
        noise_scheduler,
        all_conditions=None,
        drop_prob=None,
        predict_x_start=False,
        predict_v=False,
        learned_variance=False,
        clip_denoised=True,
        lowres_cond_img=None,
        text_encodings=None,
        cond_scale=1,
        is_latent_diffusion=False,
        lowres_noise_level=None,
        inpaint_image=None,
        inpaint_mask=None,
        inpaint_resample_times=5,
    ):
        device = self.device

        b = shape[0]

        img = torch.randn(shape, device=device)

        x_start = None  # for self-conditioning

        is_inpaint = exists(inpaint_image)
        resample_times = inpaint_resample_times if is_inpaint else 1

        if is_inpaint:
            inpaint_image = self.normalize_img(inpaint_image)
            inpaint_image = resize_image_to(inpaint_image, shape[-1], nearest=True)
            inpaint_mask = rearrange(inpaint_mask, "b h w -> b 1 h w").float()
            inpaint_mask = resize_image_to(inpaint_mask, shape[-1], nearest=True)
            inpaint_mask = inpaint_mask.bool()

        if not is_latent_diffusion:
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        for time in tqdm(
            reversed(range(0, noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=noise_scheduler.num_timesteps,
        ):
            is_last_timestep = time == 0

            for r in reversed(range(0, resample_times)):
                is_last_resample_step = r == 0

                times = torch.full((b,), time, device=device, dtype=torch.long)

                if is_inpaint:
                    # following the repaint paper
                    # https://arxiv.org/abs/2201.09865
                    noised_inpaint_image = noise_scheduler.q_sample(
                        inpaint_image, t=times
                    )
                    img = (img * ~inpaint_mask) + (noised_inpaint_image * inpaint_mask)

                self_cond = x_start if unet.self_cond else None

                img, x_start = self.p_sample(
                    unet,
                    img,
                    times,
                    all_conditions=all_conditions,
                    drop_prob=drop_prob,
                    image_embed=image_embed,
                    text_encodings=text_encodings,
                    cond_scale=cond_scale,
                    self_cond=self_cond,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_level=lowres_noise_level,
                    predict_x_start=predict_x_start,
                    predict_v=predict_v,
                    noise_scheduler=noise_scheduler,
                    learned_variance=learned_variance,
                    clip_denoised=clip_denoised,
                )

                if is_inpaint and not (is_last_timestep or is_last_resample_step):
                    # in repaint, you renoise and resample up to 10 times every step
                    img = noise_scheduler.q_sample_from_to(img, times - 1, times)

        if is_inpaint:
            img = (img * ~inpaint_mask) + (inpaint_image * inpaint_mask)

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @torch.no_grad()
    def p_sample_loop_ddim(
        self,
        unet,
        shape,
        image_embed,
        noise_scheduler,
        timesteps,
        eta=1.0,
        all_conditions=None,
        drop_prob=None,
        predict_x_start=False,
        predict_v=False,
        learned_variance=False,
        clip_denoised=True,
        lowres_cond_img=None,
        text_encodings=None,
        cond_scale=1,
        is_latent_diffusion=False,
        lowres_noise_level=None,
        inpaint_image=None,
        inpaint_mask=None,
        inpaint_resample_times=5,
    ):
        batch, device, total_timesteps, alphas, eta = (
            shape[0],
            self.device,
            noise_scheduler.num_timesteps,
            noise_scheduler.alphas_cumprod,
            self.ddim_sampling_eta,
        )

        times = torch.linspace(0.0, total_timesteps, steps=timesteps + 2)[:-1]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = list(filter(lambda t: t[0] > t[1], time_pairs))

        is_inpaint = exists(inpaint_image)
        resample_times = inpaint_resample_times if is_inpaint else 1

        if is_inpaint:
            inpaint_image = self.normalize_img(inpaint_image)
            inpaint_image = resize_image_to(inpaint_image, shape[-1], nearest=True)
            inpaint_mask = rearrange(inpaint_mask, "b h w -> b 1 h w").float()
            inpaint_mask = resize_image_to(inpaint_mask, shape[-1], nearest=True)
            inpaint_mask = inpaint_mask.bool()

        img = torch.randn(shape, device=device)

        x_start = None  # for self-conditioning

        if not is_latent_diffusion:
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            is_last_timestep = time_next == 0

            for r in reversed(range(0, resample_times)):
                is_last_resample_step = r == 0

                alpha = alphas[time]
                alpha_next = alphas[time_next]

                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

                if is_inpaint:
                    # following the repaint paper
                    # https://arxiv.org/abs/2201.09865
                    noised_inpaint_image = noise_scheduler.q_sample(
                        inpaint_image, t=time_cond
                    )
                    img = (img * ~inpaint_mask) + (noised_inpaint_image * inpaint_mask)

                self_cond = x_start if unet.self_cond else None

                unet_output = unet.forward_with_cond_scale(
                    img,
                    time_cond,
                    all_conditions=all_conditions,
                    drop_prob=drop_prob,
                    image_embed=image_embed,
                    text_encodings=text_encodings,
                    cond_scale=cond_scale,
                    self_cond=self_cond,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_level=lowres_noise_level,
                )

                pred, _ = self.parse_unet_output(learned_variance, unet_output)

                # predict x0

                if predict_v:
                    x_start = noise_scheduler.predict_start_from_v(
                        img, t=time_cond, v=pred
                    )
                elif predict_x_start:
                    x_start = pred
                else:
                    x_start = noise_scheduler.predict_start_from_noise(
                        img, t=time_cond, noise=pred
                    )

                # maybe clip x0

                if clip_denoised:
                    x_start = self.dynamic_threshold(x_start)

                # predict noise

                pred_noise = noise_scheduler.predict_noise_from_start(
                    img, t=time_cond, x0=x_start
                )

                c1 = (
                    eta
                    * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                )
                c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
                noise = torch.randn_like(img) if not is_last_timestep else 0.0

                img = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise

                if is_inpaint and not (is_last_timestep or is_last_resample_step):
                    # in repaint, you renoise and resample up to 10 times every step
                    time_next_cond = torch.full(
                        (batch,), time_next, device=device, dtype=torch.long
                    )
                    img = noise_scheduler.q_sample_from_to(
                        img, time_next_cond, time_cond
                    )

        if exists(inpaint_image):
            img = (img * ~inpaint_mask) + (inpaint_image * inpaint_mask)

        img = self.unnormalize_img(img)
        return img

    @torch.no_grad()
    def p_sample_loop(self, *args, noise_scheduler, timesteps=None, **kwargs):
        num_timesteps = noise_scheduler.num_timesteps

        timesteps = default(timesteps, num_timesteps)
        assert timesteps <= num_timesteps
        is_ddim = timesteps < num_timesteps

        if not is_ddim:
            return self.p_sample_loop_ddpm(
                *args, noise_scheduler=noise_scheduler, **kwargs
            )

        return self.p_sample_loop_ddim(
            *args, noise_scheduler=noise_scheduler, timesteps=timesteps, **kwargs
        )

    def p_losses(
        self,
        unet,
        x_start,
        times,
        *,
        image_embed,
        noise_scheduler,
        all_conditions=None,
        lowres_cond_img=None,
        text_encodings=None,
        predict_x_start=False,
        predict_v=False,
        noise=None,
        learned_variance=False,
        clip_denoised=False,
        is_latent_diffusion=False,
        lowres_noise_level=None,
    ):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]

        if not is_latent_diffusion:
            x_start = self.normalize_img(x_start)
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t

        x_noisy = noise_scheduler.q_sample(x_start=x_start, t=times, noise=noise)

        # unet kwargs

        unet_kwargs = dict(
            image_embed=image_embed,
            text_encodings=text_encodings,
            lowres_cond_img=lowres_cond_img,
            lowres_noise_level=lowres_noise_level,
        )

        # self conditioning

        self_cond = None

        if unet.self_cond and random.random() < 0.5:
            with torch.no_grad():
                unet_output = unet(x_noisy, times, **unet_kwargs)
                self_cond, _ = self.parse_unet_output(learned_variance, unet_output)
                self_cond = self_cond.detach()

        # forward to get model prediction

        unet_output = unet(
            x_noisy,
            times,
            **unet_kwargs,
            all_conditions=all_conditions,
            self_cond=self_cond,
            drop_prob=self.drop_prob,
        )

        pred, _ = self.parse_unet_output(learned_variance, unet_output)

        x0_denoise = noise_scheduler.predict_start_from_noise(x_noisy, times, pred)
        x0_denoise = self.unnormalize_img(x0_denoise)

        if predict_v:
            target = noise_scheduler.calculate_v(x_start, times, noise)
        elif predict_x_start:
            target = x_start
        else:
            target = noise

        loss = noise_scheduler.loss_fn(pred, target, reduction="none")

        loss = reduce(loss, "b ... -> b (...)", "mean")

        loss = noise_scheduler.p2_reweigh_loss(loss, times)

        loss = loss.mean()

        if not learned_variance:
            return {"loss": loss, "x0_denoise": x0_denoise}

        # most of the code below is transcribed from
        # https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
        # the Improved DDPM paper then further modified it so that the mean is detached (shown a couple lines before), and weighted to be smaller than the l1 or l2 "simple" loss
        # it is questionable whether this is really needed, looking at some of the figures in the paper, but may as well stay faithful to their implementation

        # if learning the variance, also include the extra weight kl loss

        true_mean, _, true_log_variance_clipped = noise_scheduler.q_posterior(
            x_start=x_start, x_t=x_noisy, t=times
        )
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            unet,
            x=x_noisy,
            t=times,
            image_embed=image_embed,
            noise_scheduler=noise_scheduler,
            clip_denoised=clip_denoised,
            learned_variance=True,
            model_output=unet_output,
        )

        # kl loss with detached model predicted mean, for stability reasons as in paper

        detached_model_mean = model_mean.detach()

        kl = normal_kl(
            true_mean,
            true_log_variance_clipped,
            detached_model_mean,
            model_log_variance,
        )
        kl = meanflat(kl) * NAT

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=detached_model_mean, log_scales=0.5 * model_log_variance
        )
        decoder_nll = meanflat(decoder_nll) * NAT

        # at the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        vb_losses = torch.where(times == 0, decoder_nll, kl)

        # weight the vb loss smaller, for stability, as in the paper (recommended 0.001)

        vb_loss = vb_losses.mean() * self.vb_loss_weight

        return {"loss": loss + vb_loss, "x0_denoise": x0_denoise}

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        image=None,
        all_conditions=None,
        drop_prob=None,
        image_embed=None,
        text=None,
        text_encodings=None,
        batch_size=1,
        cond_scale=1.0,
        start_at_unet_number=1,
        stop_at_unet_number=None,
        distributed=False,
        inpaint_image=None,
        inpaint_mask=None,
        inpaint_resample_times=5,
        one_unet_in_gpu_at_time=True,
        sample_use_render=False,
        timesteps=None,
    ):
        if not exists(image_embed) and "images_render" in all_conditions:
            assert exists(
                self.recognizer
            ), "if you want to derive recognizer image embeddings automatically, you must supply `recognizer` to the decoder on init"
            image_embed = self.recognizer.get_image_embed(
                all_conditions["images_render"]
            )
            image_embed = self.attn_pooling(image_embed)

        if not self.unconditional:
            batch_size = image_embed.shape[0]

        device = image_embed.device

        if (
            "text" in all_conditions
            and not exists(text_encodings)
            and not self.unconditional
        ):
            text_encodings, _ = self.recognizer.get_text_embed(all_conditions["text"])
            t_b, t_n, _ = text_encodings.shape
            pos_emb_text = self.abs_pos_emb_text(torch.arange(t_n, device=device))
            text_encodings = text_encodings + rearrange(pos_emb_text, "n d -> 1 n d")

        assert not (
            self.condition_on_text_encodings and not exists(text_encodings)
        ), "text or text encodings must be passed into decoder if specified"
        assert not (
            not self.condition_on_text_encodings and exists(text_encodings)
        ), "decoder specified not to be conditioned on text, yet it is presented"

        img = None
        if start_at_unet_number > 1:
            # Then we are not generating the first image and one must have been passed in
            assert exists(
                image
            ), "image must be passed in if starting at unet number > 1"
            assert (
                image.shape[0] == batch_size
            ), "image must have batch size of {} if starting at unet number > 1".format(
                batch_size
            )
            prev_unet_output_size = self.image_sizes[start_at_unet_number - 2]
            img = resize_image_to(image, prev_unet_output_size, nearest=True)

        is_cuda = next(self.parameters()).is_cuda

        num_unets = self.num_unets
        cond_scale = cast_tuple(cond_scale, num_unets)

        drop_prob_sample = self.drop_prob_sample
        drop_prob_sample.update(drop_prob)

        for (
            unet_number,
            unet,
            vae,
            channel,
            image_size,
            predict_x_start,
            predict_v,
            learned_variance,
            noise_scheduler,
            lowres_cond,
            sample_timesteps,
            unet_cond_scale,
        ) in tqdm(
            zip(
                range(1, num_unets + 1),
                self.unets,
                self.vaes,
                self.sample_channels,
                self.image_sizes,
                self.predict_x_start,
                self.predict_v,
                self.learned_variance,
                self.noise_schedulers,
                self.lowres_conds,
                self.sample_timesteps,
                cond_scale,
            )
        ):
            if unet_number < start_at_unet_number:
                continue  # It's the easiest way to do it

            context = (
                self.one_unet_in_gpu(unet=unet)
                if is_cuda and one_unet_in_gpu_at_time
                else null_context()
            )

            with context:
                # prepare low resolution conditioning for upsamplers

                lowres_cond_img = lowres_noise_level = None
                shape = (batch_size, channel, image_size[0], image_size[1])

                if unet.lowres_cond:
                    lowres_cond_img = resize_image_to(
                        img,
                        target_image_size=image_size,
                        clamp_range=self.input_image_range,
                        nearest=True,
                    )

                    if lowres_cond.use_noise:
                        lowres_noise_level = torch.full(
                            (batch_size,),
                            int(self.lowres_noise_sample_level * 1000),
                            dtype=torch.long,
                            device=self.device,
                        )
                        lowres_cond_img, _ = lowres_cond.noise_image(
                            lowres_cond_img, lowres_noise_level
                        )

                # latent diffusion

                is_latent_diffusion = isinstance(vae, VQGanVAE)
                image_size = vae.get_encoded_fmap_size(image_size)
                shape = (batch_size, vae.encoded_dim, image_size[0], image_size[1])

                lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

                # denoising loop for image

                img = self.p_sample_loop(
                    unet,
                    shape,
                    all_conditions=all_conditions,
                    drop_prob=drop_prob_sample,
                    image_embed=image_embed,
                    text_encodings=text_encodings,
                    cond_scale=unet_cond_scale,
                    predict_x_start=predict_x_start,
                    predict_v=predict_v,
                    learned_variance=learned_variance,
                    clip_denoised=not is_latent_diffusion,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_level=lowres_noise_level,
                    is_latent_diffusion=is_latent_diffusion,
                    noise_scheduler=noise_scheduler,
                    timesteps=timesteps,
                    inpaint_image=inpaint_image,
                    inpaint_mask=inpaint_mask,
                    inpaint_resample_times=inpaint_resample_times,
                )

                img = vae.decode(img)

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        return img

    def forward(
        self,
        image,
        text=None,
        all_conditions=None,
        image_embed=None,
        text_encodings=None,
        unet_number=None,
    ):
        assert not (
            self.num_unets > 1 and not exists(unet_number)
        ), f"you must specify which unet you want trained, from a range of 1 to {self.num_unets}, if you are training cascading DDPM (multiple unets)"
        unet_number = default(unet_number, 1)
        unet_index = unet_number - 1

        unet = self.get_unet(unet_number)

        vae = self.vaes[unet_index]
        noise_scheduler = self.noise_schedulers[unet_index]
        lowres_conditioner = self.lowres_conds[unet_index]
        target_image_size = self.image_sizes[unet_index]
        predict_x_start = self.predict_x_start[unet_index]
        predict_v = self.predict_v[unet_index]
        random_crop_size = self.random_crop_sizes[unet_index]
        learned_variance = self.learned_variance[unet_index]
        b, c, h, w, device, = (
            *image.shape,
            image.device,
        )

        assert image.shape[1] == self.channels
        assert h >= target_image_size[0] and w >= target_image_size[1]
        times = torch.randint(
            0, noise_scheduler.num_timesteps, (b,), device=device, dtype=torch.long
        )

        if (
            "images_render" in all_conditions
            and not exists(image_embed)
            and not self.unconditional
        ):
            assert exists(
                self.recognizer
            ), "if you want to derive recognizer image embeddings automatically, you must supply `recognizer` to the decoder on init"
            image_embed = self.recognizer.get_image_embed(
                all_conditions["images_render"]
            )
            image_embed = self.attn_pooling(image_embed)

        if (
            "text" in all_conditions
            and not exists(text_encodings)
            and not self.unconditional
        ):
            text_encodings, _ = self.recognizer.get_text_embed(all_conditions["text"])
            t_b, t_n, _ = text_encodings.shape
            pos_emb_text = self.abs_pos_emb_text(torch.arange(t_n, device=device))
            text_encodings = text_encodings + rearrange(pos_emb_text, "n d -> 1 n d")

        assert not (
            self.condition_on_text_encodings and not exists(text_encodings)
        ), "text or text encodings must be passed into decoder if specified"
        assert not (
            not self.condition_on_text_encodings and exists(text_encodings)
        ), "decoder specified not to be conditioned on text, yet it is presented"

        lowres_cond_img, lowres_noise_level = (
            lowres_conditioner(
                image,
                target_image_size=target_image_size,
                downsample_image_size=self.image_sizes[unet_index - 1],
            )
            if exists(lowres_conditioner)
            else (None, None)
        )
        image = resize_image_to(image, target_image_size, nearest=True)

        if exists(random_crop_size):
            aug = K.RandomCrop((random_crop_size, random_crop_size), p=1.0)

            # make sure low res conditioner and image both get augmented the same way
            # detailed https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            image = aug(image)
            lowres_cond_img = aug(lowres_cond_img, params=aug._params)

        is_latent_diffusion = not isinstance(vae, NullVQGanVAE)

        vae.eval()
        with torch.no_grad():
            image = vae.encode(image)
            lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

        outputs = self.p_losses(
            unet,
            image,
            times,
            all_conditions=all_conditions,
            image_embed=image_embed,
            text_encodings=text_encodings,
            lowres_cond_img=lowres_cond_img,
            predict_x_start=predict_x_start,
            predict_v=predict_v,
            learned_variance=learned_variance,
            is_latent_diffusion=is_latent_diffusion,
            noise_scheduler=noise_scheduler,
            lowres_noise_level=lowres_noise_level,
        )

        return outputs
