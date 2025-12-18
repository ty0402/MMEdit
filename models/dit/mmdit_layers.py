from typing import Optional
from typing import Union

import torch
from einops import rearrange
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


import torch
from torch import nn
from torch.nn import functional as F

from .modules import RMSNorm

# https://github.com/facebookresearch/DiT
# Ref: https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
# Ref: https://github.com/lucidrains/rotary-embedding-torch


def compute_rope_rotations(length: int,
                           dim: int,
                           theta: int,
                           *,
                           freq_scaling: float = 1.0,
                           device: Union[torch.device, str] = 'cpu') -> Tensor:
    assert dim % 2 == 0

    with torch.amp.autocast(device_type='cuda', enabled=False):
        pos = torch.arange(length, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freqs *= freq_scaling

        rot = torch.einsum('..., f -> ... f', pos, freqs)
        rot = torch.stack([torch.cos(rot), -torch.sin(rot), torch.sin(rot), torch.cos(rot)], dim=-1)
        rot = rearrange(rot, 'n d (i j) -> 1 n d i j', i=2, j=2)
        return rot


def apply_rope(x: Tensor, rot: Tensor) -> tuple[Tensor, Tensor]:
    with torch.amp.autocast(device_type='cuda', enabled=False):
        _x = x.float()
        _x = _x.view(*_x.shape[:-1], -1, 1, 2)
        x_out = rot[..., 0] * _x[..., 0] + rot[..., 1] * _x[..., 1]
        return x_out.reshape(*x.shape).to(dtype=x.dtype)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, dim, frequency_embedding_size, max_period):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.dim = dim
        self.max_period = max_period
        assert dim % 2 == 0, 'dim must be even.'

        with torch.autocast('cuda', enabled=False):
            # 1. 先计算出最终的张量
            initial_freqs = 1.0 / (10000**(torch.arange(0, frequency_embedding_size, 2, dtype=torch.float32) /
                                           frequency_embedding_size))
            freq_scale = 10000 / max_period
            freqs_tensor = freq_scale * initial_freqs

            # 2. 使用 register_buffer() 将最终的张量注册为 buffer
            self.register_buffer('freqs', freqs_tensor, persistent=False)

    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
    
class ChannelLastConv1d(nn.Conv1d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


# https://github.com/Stability-AI/sd3-ref
class MLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ConvMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim,
                                    dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w3 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift




def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # training will crash without these contiguous calls and the CUDNN limitation
    # I believe this is related to https://github.com/pytorch/pytorch/issues/133974
    # unresolved at the time of writing
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = F.scaled_dot_product_attention(q, k, v)
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    return out


class SelfAttention(nn.Module):

    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(dim // nheads)
        self.k_norm = RMSNorm(dim // nheads)

        self.split_into_heads = Rearrange('b n (h d j) -> b h n d j',
                                          h=nheads,
                                          d=dim // nheads,
                                          j=3)

    def pre_attention(
            self, x: torch.Tensor,
            rot: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch_size * n_tokens * n_channels
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)

        return q, k, v

    def forward(
            self,
            x: torch.Tensor,  # batch_size * n_tokens * n_channels
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(x)
        out = attention(q, k, v)
        return out


class MMDitSingleBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 nhead: int,
                 mlp_ratio: float = 4.0,
                 pre_only: bool = False,
                 kernel_size: int = 7,
                 padding: int = 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, nhead)

        self.pre_only = pre_only
        if pre_only:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        else:
            if kernel_size == 1:
                self.linear1 = nn.Linear(dim, dim)
            else:
                self.linear1 = ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

            if kernel_size == 1:
                self.ffn = MLP(dim, int(dim * mlp_ratio))
            else:
                self.ffn = ConvMLP(dim,
                                   int(dim * mlp_ratio),
                                   kernel_size=kernel_size,
                                   padding=padding)

            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: Optional[torch.Tensor]):
        # x: BS * N * D
        # cond: BS * D
        modulation = self.adaLN_modulation(c)
        if self.pre_only:
            (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
             gate_mlp) = modulation.chunk(6, dim=-1)

        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor]):
        if self.pre_only:
            return x

        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.ffn(r) * gate_mlp

        return x
    # 这里的forward似乎没有用到
    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                rot: Optional[torch.Tensor]) -> torch.Tensor:
        # x: BS * N * D
        # cond: BS * D
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        attn_out = attention(*x_qkv)
        x = self.post_attention(x, attn_out, x_conditions)

        return x




class JointBlock_AT(nn.Module):
    """
    Audio + Text only JointBlock（去掉 clip 分支）
    返回 (latent, text_f)
    """
    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, pre_only: bool = False):
        super().__init__()
        self.pre_only = pre_only
        self.latent_block = MMDitSingleBlock(dim,
                                             nhead,
                                             mlp_ratio,
                                             pre_only=False,
                                             kernel_size=3,
                                             padding=1)
        # text_block 仍保留 pre_only 参数（可能是 pre-only 的 AdaLN）
        self.text_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)

    def forward(self, latent: torch.Tensor, text_f: torch.Tensor,
                global_c: torch.Tensor, extended_c: torch.Tensor, latent_rot: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # latent: (B, N_latent, D)
        # text_f: (B, N_text, D)
        # global_c: (B, 1, D) or (B, D)
        # extended_c: (B, N_latent, D) or (B, 1, D)
        x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, latent_rot)
        # text没有做rope编码, 也有点奇怪，可能audiollm中带有

        t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=None)

        latent_len = latent.shape[1]
        text_len = text_f.shape[1]

        # 只拼接 latent + text
        joint_qkv = [torch.cat([x_qkv[i], t_qkv[i]], dim=2) for i in range(3)]  # dim=2=token dim

        attn_out = attention(*joint_qkv)  # (B, latent_len + text_len, D)
        x_attn_out = attn_out[:, :latent_len]      # (B, latent_len, D)
        t_attn_out = attn_out[:, latent_len:]      # (B, text_len, D)

        latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
        if not self.pre_only:
            text_f = self.text_block.post_attention(text_f, t_attn_out, t_mod)

        return latent, text_f


    # 改一下mask的逻辑
    # def forward(self, latent, text_f, global_c, extended_c, latent_rot,
    #         latent_mask: torch.Tensor, text_mask: torch.Tensor):
    #     # latent_mask: (B, N_latent) {0,1}
    #     # text_mask:   (B, N_text)   {0,1}

    #     x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, latent_rot)
    #     t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=None)

    #     latent_len = latent.shape[1]
    #     text_len   = text_f.shape[1]

    #     # 1) 拼 qkv
    #     joint_qkv = [torch.cat([x_qkv[i], t_qkv[i]], dim=2) for i in range(3)]  # 这里假设 token 维=2

    #     # 2) 构造 key mask（拼接后的）
    #     key_mask = torch.cat([latent_mask, text_mask], dim=1).bool()   # (B, N_total)

    #     # 3) 调用注意力（要求 attention 支持 key_mask）
    #     # 若你的 attention 不支持，需要自己在里面对 logits 做 -inf 掩码；示例见后
    #     attn_out = attention(*joint_qkv, key_mask=key_mask)  # (B, N_total, D)

    #     # 4) 切回两段
    #     x_attn_out = attn_out[:, :latent_len, :]
    #     t_attn_out = attn_out[:, latent_len:, :]

    #     # 5) 对 query 端输出做屏蔽（避免 padding query 写回）
    #     x_attn_out = x_attn_out * latent_mask.unsqueeze(-1)  # (B, N_latent, D)
    #     t_attn_out = t_attn_out * text_mask.unsqueeze(-1)    # (B, N_text,  D)

    #     # 6) post_attention 内部**还要**用 query mask 把残差和 FFN 的更新再屏蔽一次（见下一节）
    #     latent = self.latent_block.post_attention(latent, x_attn_out, x_mod,
    #                                             query_mask=latent_mask)
    #     if not self.text_block.pre_only:
    #         text_f = self.text_block.post_attention(text_f, t_attn_out, t_mod,
    #                                                 query_mask=text_mask)

    #     return latent, text_f



class FinalBlock(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent