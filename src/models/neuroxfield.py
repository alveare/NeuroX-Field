import torch
import torch.nn as nn

from functools import lru_cache
from torch.nn.functional import scaled_dot_product_attention as attention
from src.models.layers.time_embed import TimestepEmbedder as TimestepEmbedder
from src.models.layers.patch_embed import Embed as Embed
from src.models.layers.swiglu import SwiGLU as FeedForward
from src.models.layers.rmsnorm import RMSNorm as Norm


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.contiguous())

        q = q.view(B, self.num_heads, -1, self.head_dim)
        k = k.view(B, self.num_heads, -1, self.head_dim).contiguous()
        v = v.view(B, self.num_heads, -1, self.head_dim).contiguous()

        x = attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

        # set store_attn=True to capture attention weights in last_attn
        self.store_attn = False
        self.last_attn = None

    def forward(self, q_in, kv_in):
        B, Nq, D = q_in.shape
        Nkv = kv_in.shape[1]
        H = self.num_heads
        Hd = self.head_dim

        q = self.q_proj(q_in).view(B, Nq, H, Hd).permute(0, 2, 1, 3).contiguous()
        kv = self.kv_proj(kv_in).view(B, Nkv, 2, H, Hd).permute(2, 0, 3, 1, 4)
        k, v = kv[0].contiguous(), kv[1].contiguous()

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.store_attn:
            scale = 1.0 / (self.head_dim ** 0.5)
            attn = (q.float() @ k.float().transpose(-2, -1)) * scale
            self.last_attn = attn.softmax(dim=-1).detach().cpu()

        x = attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, Nq, D)
        x = self.proj(x)
        return x


class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attn + MLP, both AdaLN-modulated by timestep conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm_q = Norm(hidden_size, eps=1e-6)
        self.norm_kv = Norm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, q, kv, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        q = q + gate_msa * self.cross_attn(
            modulate(self.norm_q(q), shift_msa, scale_msa),
            self.norm_kv(kv),
        )
        q = q + gate_mlp * self.mlp(modulate(self.norm2(q), shift_mlp, scale_mlp))
        return q


class NFEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs, hidden_size_input, bias=True),
        )

    @lru_cache
    def fetch_pos(self, height, width, device, dtype):
        pos_tp = torch.arange(width, device=device, dtype=dtype) / width
        freqs = torch.arange(1, self.max_freqs + 1, dtype=dtype, device=device)
        dct = torch.cos(pos_tp[:, None] * freqs[None, :] * torch.pi)
        dct = dct.repeat(height, 1)
        return dct.unsqueeze(0)

    def forward(self, inputs, patch_size_ct=None):
        B, P_area, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype
        if patch_size_ct is not None:
            patch_c, patch_t = patch_size_ct
        else:
            patch_size = int(P_area ** 0.5)
            patch_c, patch_t = patch_size, patch_size
        dct = self.fetch_pos(patch_c, patch_t, device, dtype)
        dct = dct.repeat(B, 1, 1)
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        return inputs


class NFBlock(nn.Module):
    def __init__(self, hidden_size_s, hidden_size_x, mlp_ratio=2):
        super().__init__()
        self.param_generator1 = nn.Sequential(
            nn.Linear(hidden_size_s, 2 * hidden_size_x ** 2 * mlp_ratio, bias=True),
        )
        self.norm = Norm(hidden_size_x, eps=1e-6)
        self.hidden_size_x = hidden_size_x
        self.mlp_ratio = mlp_ratio

    def forward(self, x, s):
        batch_size, _, hidden_size_x = x.shape
        mlp_params1 = self.param_generator1(s)
        fc1_param1, fc2_param1 = mlp_params1.chunk(2, dim=-1)
        fc1_param1 = fc1_param1.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc2_param1 = fc2_param1.view(batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x)
        normalized_fc1_param1 = torch.nn.functional.normalize(fc1_param1, dim=-2)
        res_x = x
        x = self.norm(x)
        x = torch.bmm(x, normalized_fc1_param1)
        x = torch.nn.functional.silu(x)
        x = torch.bmm(x, fc2_param1)
        x = x + res_x
        return x


class NFFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        return self.linear(x)


class NeuroXField(nn.Module):
    """
    Denoiser for electrode super-resolution: a set-to-set map from observed
    electrodes to a continuous field queried at arbitrary scalp positions.

    forward(x, t, y, pos, x_query, pos_query) -> (B, 1, C_query, T)
        x         (B, 1, C_enc,   T)  noisy signal at the observed electrodes
        y         (B, 1, C_enc,   T)  conditioning signal, same rows as x
        t         (B,)                flow-matching timestep
        pos       (B, C_enc,   D)     observed electrode coords, normalized
        x_query   (B, 1, C_query, T)  noisy signal at the queried positions
        pos_query (B, C_query, D)     queried coords, normalized (D=3)
    """
    def __init__(
            self,
            in_channels=1,
            num_groups=4,
            hidden_size=512,
            decoder_hidden_size=64,
            num_encoder_blocks=6,
            num_cross_blocks=2,
            num_decoder_blocks=2,
            patch_size_c=1,
            patch_size_t=100,
            nf_max_freqs=32,
            s_scale=16.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.decoder_hidden_size = decoder_hidden_size
        self.num_encoder_blocks = num_encoder_blocks
        self.num_cross_blocks = num_cross_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.patch_size_c = patch_size_c
        self.patch_size_t = patch_size_t
        self.patch_area = patch_size_c * patch_size_t
        self.nf_max_freqs = nf_max_freqs
        self.s_scale = s_scale

        # Encoder sees visible-only (x, y)
        self.s_embedder = Embed(2 * in_channels * self.patch_area, hidden_size, bias=True)
        self.x_embedder = NFEmbedder(in_channels, decoder_hidden_size, max_freqs=self.nf_max_freqs)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.final_layer = NFFinalLayer(decoder_hidden_size, in_channels)

        # Learnable latent query token
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        encoder_blocks = nn.ModuleList([
            FlattenDiTBlock(hidden_size, num_groups) for _ in range(num_encoder_blocks)
        ])
        cross_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_size, num_groups) for _ in range(num_cross_blocks)
        ])
        decoder_blocks = nn.ModuleList([
            NFBlock(hidden_size, decoder_hidden_size, mlp_ratio=2)
            for _ in range(num_decoder_blocks)
        ])
        self.encoder_blocks = encoder_blocks
        self.cross_blocks = cross_blocks
        self.decoder_blocks = decoder_blocks
        self.blocks = nn.ModuleList(list(encoder_blocks) + list(cross_blocks) + list(decoder_blocks))

        self.initialize_weights()
    
    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _build_freqs(self, pos, T_patches):
        """
        pos: (B, N, coord_dim) per-sample coords (already normalized).

        Returns (B, N*T_patches, hidden_size) absolute sin/cos PE,
            added to the token embeddings before the blocks.
        """
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        B, N, coord_dim = pos.shape
        device = pos.device
        dtype = pos.dtype

        pos_spatial = pos.repeat_interleave(T_patches, dim=1)
        t_coords = torch.arange(T_patches, dtype=dtype, device=device)
        pos_temporal = t_coords.repeat(N).view(1, -1, 1).expand(B, -1, -1)
        pos_st = torch.cat([pos_spatial, pos_temporal], dim=-1)

        return self.sincos_pos_embed(pos_st, T_patches)

    def sincos_pos_embed(self, pos_st, T_patches, theta=10000.0):
        """
        Absolute sin/cos (Fourier-feature) PE over continuous (spatial, temporal) coords.

        pos_st: (B, M, P) where P = coord_dim+1 and the last component is the
                temporal patch index. Returns (B, M, hidden_size) to be ADDED to tokens.
        """
        B, M, P = pos_st.shape
        device, dtype = pos_st.device, pos_st.dtype
        dim = self.hidden_size

        scales = torch.full((P,), float(self.s_scale), device=device, dtype=dtype)
        scales[-1] = float(T_patches)
        pos_scaled = pos_st * scales

        per_axis = dim // P
        num_freqs = max(per_axis // 2, 1)
        freqs = 1.0 / (theta ** (torch.arange(num_freqs, device=device, dtype=torch.float32) / num_freqs))
        freqs = freqs.to(dtype)

        feats = []
        for p in range(P):
            ang = pos_scaled[..., p:p + 1] * freqs
            feats.append(torch.sin(ang))
            feats.append(torch.cos(ang))
        pe = torch.cat(feats, dim=-1)

        if pe.shape[-1] < dim:
            print(f"Padding PE from {pe.shape[-1]} to {dim}")
            pad = torch.zeros(B, M, dim - pe.shape[-1], device=device, dtype=pe.dtype)
            pe = torch.cat([pe, pad], dim=-1)
        elif pe.shape[-1] > dim:
            print(f"Truncating PE from {pe.shape[-1]} to {dim}")
            pe = pe[..., :dim]
        return pe

    def forward(self, x, t, y, pos=None, x_query=None, pos_query=None):
        assert pos is not None, "pos (encoder spatial coords) is required"
        assert x_query is not None and pos_query is not None, (
            "x_query and pos_query are required — encoder/decoder "
            "operate on different sets of electrodes."
        )
        B, _, _, T = x.shape
        T_patches = T // self.patch_size_t
        N_query = x_query.shape[2]

        # ---- Encoder input: [x, y] concat across channel (visible-only) ----
        xy = torch.cat([x, y], dim=1)
        xy = torch.nn.functional.unfold(
            xy,
            kernel_size=(self.patch_size_c, self.patch_size_t),
            stride=(self.patch_size_c, self.patch_size_t),
        ).transpose(1, 2)

        # ---- Continuous-coord sincos PE for encoder + decoder queries ----
        xpos_enc = self._build_freqs(pos, T_patches)
        xpos_q = self._build_freqs(pos_query, T_patches)

        # ---- Timestep conditioning ----
        t_emb = self.t_embedder(t.view(-1)).view(B, 1, self.hidden_size)
        c = nn.functional.silu(t_emb)

        # ---- DiT Encoder ----
        s = self.s_embedder(xy)
        s = s + xpos_enc  # sinusoidal PE enters additively here
        for blk in self.encoder_blocks: 
            s = blk(s, c)

        # ---- Cross-attention: decoder queries gather encoder context ----
        q = self.query_token.expand(B, N_query * T_patches, -1).contiguous()
        q = q + xpos_q 
        for blk in self.cross_blocks:
            q = blk(q, s, c)

        s_q = nn.functional.silu(t_emb + q)

        # ---- Per-patch Neural Field Decoder ----
        x_dec = torch.nn.functional.unfold(
            x_query,
            kernel_size=(self.patch_size_c, self.patch_size_t),
            stride=(self.patch_size_c, self.patch_size_t),
        ).transpose(1, 2)

        BN = B * N_query * T_patches
        x_dec = x_dec.reshape(BN, self.in_channels, self.patch_area).transpose(1, 2)
        s_flat = s_q.reshape(BN, self.hidden_size)

        x_dec = self.x_embedder(x_dec, (self.patch_size_c, self.patch_size_t))
        for blk in self.decoder_blocks:
            x_dec = blk(x_dec, s_flat)

        x_dec = self.final_layer(x_dec)
        x_dec = x_dec.transpose(1, 2)
        x_dec = x_dec.reshape(B, N_query * T_patches, -1)

        x_out = torch.nn.functional.fold(
            x_dec.transpose(1, 2).contiguous(),
            (N_query, T),
            kernel_size=(self.patch_size_c, self.patch_size_t),
            stride=(self.patch_size_c, self.patch_size_t),
        )
        return x_out
