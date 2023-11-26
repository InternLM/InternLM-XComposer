import math
from typing import List, Union
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_InternLM_XComposer import InternLMXComposerConfig
from .modeling_utils import LoRALinear

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "InternLMXComposerConfig"


def rotary_embed(x1, x2, cos, sin, conj):
    x1, x2 = x1.float(), x2.float()
    if conj:
        x1, x2 = x1 * cos + x2 * sin, x1 * sin + x2 * cos
    else:
        x1, x2 = x1 * cos - x2 * sin, x1 * sin + x2 * cos
    return x1, x2


class LegacyApplyRotaryEmbQKV_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
        """
            qkv: (batch_size, seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        batch, seqlen, three, nheads, headdim = qkv.shape
        assert three == 3
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        q_ro = qkv[:, :, 0, :, :rotary_dim]
        q1, q2 = q_ro.chunk(2, dim=-1) if not interleaved else (q_ro[..., ::2], q_ro[..., 1::2])
        # rotary_emb.apply_rotary(q1, q2, rearrange(cos[:seqlen], 's d -> s 1 d'),
        #                         rearrange(sin[:seqlen], 's d -> s 1 d'), q1, q2, False)
        q1, q2 = rotary_embed(q1, q2, rearrange(cos[:seqlen], 's d -> s 1 d'), rearrange(sin[:seqlen], 's d -> s 1 d'), False)
        qkv[:, :, 0, :, :rotary_dim] = torch.cat([q1, q2], dim=-1)
        k_ro = qkv[:, :, 1, :, :rotary_dim]
        k1, k2 = k_ro.chunk(2, dim=-1) if not interleaved else (k_ro[..., ::2], k_ro[..., 1::2])
        # rotary_emb.apply_rotary(k1, k2, rearrange(cos_k[:seqlen], 's d -> s 1 d'),
        #                         rearrange(sin_k[:seqlen], 's d -> s 1 d'), k1, k2, False)
        k1, k2 = rotary_embed(k1, k2, rearrange(cos_k[:seqlen], 's d -> s 1 d'), rearrange(sin_k[:seqlen], 's d -> s 1 d'), False)
        qkv[:, :, 1, :, :rotary_dim] = torch.cat([k1, k2], dim=-1)
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        _, seqlen, _, _, headdim = dqkv.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq_ro = dqkv[:, :, 0, :, :rotary_dim]
        dq1, dq2 = (dq_ro.chunk(2, dim=-1) if not ctx.interleaved
                    else (dq_ro[..., ::2], dq_ro[..., 1::2]))
        rotary_emb.apply_rotary(dq1, dq2, rearrange(cos[:seqlen], 's d -> s 1 d'),
                                rearrange(sin[:seqlen], 's d -> s 1 d'), dq1, dq2, True)
        dk_ro = dqkv[:, :, 1, :, :rotary_dim]
        dk1, dk2 = (dk_ro.chunk(2, dim=-1) if not ctx.interleaved
                    else (dk_ro[..., ::2], dk_ro[..., 1::2]))
        rotary_emb.apply_rotary(dk1, dk2, rearrange(cos_k[:seqlen], 's d -> s 1 d'),
                                rearrange(sin_k[:seqlen], 's d -> s 1 d'), dk1, dk2, True)
        return dqkv, None, None, None, None, None


class ConvertedInternLMRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, base=10000, scale_base=0, device=None):
        """ """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base**(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.scale_base = scale_base
        scale = ((torch.arange(0, dim, 2, device=device, dtype=torch.float32) +
                  0.4 * dim) / (1.4 * dim) if scale_base > 0 else None)
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item() + 1
        else:
            seqlen = indexes + 1  # eval_forward
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen,
                             device=x.device,
                             dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = (torch.arange(
                    seqlen, dtype=self.scale.dtype, device=self.scale.device) -
                         seqlen // 2) / self.scale_base
                scale = self.scale.to(device=power.device)**rearrange(
                    power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def eval_forward(self, qkv, seqlen_offset=0):
        """
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(qkv, seqlen_offset + qkv.shape[1])
        if self.scale is None:
            return legacy_apply_rotary_embed_qkv(
                qkv, self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:])
        else:
            return legacy_apply_rotary_embed_qkv(
                qkv,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self._cos_k_cached[seqlen_offset:],
                self._sin_k_cached[seqlen_offset:],
            )


legacy_apply_rotary_embed_qkv = LegacyApplyRotaryEmbQKV_.apply


class InternConvertedInternLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: InternLMXComposerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")
        if config.lora_cfg is None:
            self.q_proj = nn.Linear(self.hidden_size,
                                    self.num_heads * self.head_dim,
                                    bias=config.kqvo_bias)
            self.k_proj = nn.Linear(self.hidden_size,
                                    self.num_heads * self.head_dim,
                                    bias=config.kqvo_bias)
            self.v_proj = nn.Linear(self.hidden_size,
                                    self.num_heads * self.head_dim,
                                    bias=config.kqvo_bias)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                    self.hidden_size,
                                    bias=config.kqvo_bias)

        else:
            lora_cfg = config.lora_cfg
            if 'q' in lora_cfg['learn_param']:
                self.q_proj = LoRALinear(self.hidden_size,
                                         self.num_heads * self.head_dim,
                                         bias=config.kqvo_bias,
                                         **lora_cfg)
            else:
                self.q_proj = nn.Linear(
                    self.hidden_size,
                    self.num_heads * self.head_dim,
                    bias=config.kqvo_bias,
                )
            if 'k' in lora_cfg['learn_param']:
                self.k_proj = LoRALinear(self.hidden_size,
                                         self.num_heads * self.head_dim,
                                         bias=config.kqvo_bias,
                                         **lora_cfg)
            else:
                self.k_proj = nn.Linear(
                    self.hidden_size,
                    self.num_heads * self.head_dim,
                    bias=config.kqvo_bias,
                )
            if 'v' in lora_cfg['learn_param']:
                self.v_proj = LoRALinear(self.hidden_size,
                                         self.num_heads * self.head_dim,
                                         bias=config.kqvo_bias,
                                         **lora_cfg)
            else:
                self.v_proj = nn.Linear(
                    self.hidden_size,
                    self.num_heads * self.head_dim,
                    bias=config.kqvo_bias,
                )

            if 'o' in lora_cfg['learn_param']:
                self.o_proj = LoRALinear(self.num_heads * self.head_dim,
                                         self.hidden_size,
                                         bias=config.kqvo_bias,
                                         **lora_cfg)
            else:
                self.o_proj = nn.Linear(
                    self.num_heads * self.head_dim,
                    self.hidden_size,
                    bias=config.kqvo_bias,
                )

        self.rotary_emb = ConvertedInternLMRotaryEmbedding(self.head_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        q = query_states
        k = key_states
        v = value_states

        qkv = torch.cat([q, k, v], dim=2).contiguous()
        qkv = qkv.view(bsz, q_len, -1)
        qkv = rearrange(qkv,
                        "b s (three h d) -> b s three h d",
                        three=3,
                        d=self.head_dim)

        if past_key_value is not None:
            qkv = self.rotary_emb.eval_forward(
                qkv, seqlen_offset=past_key_value[0].shape[2])
        else:
            qkv = self.rotary_emb.eval_forward(qkv)

        query_states, key_states, value_states = qkv.unbind(2)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class ConvertedLoRALinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.05,
                 **kwargs) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.lora_scaling = self.lora_alpha / self.lora_r

        self.lora_A = nn.Linear(in_features,
                                self.lora_r,
                                bias=False,
                                device=device,
                                dtype=dtype)
        self.lora_B = nn.Linear(self.lora_r,
                                out_features,
                                bias=False,
                                device=device,
                                dtype=dtype)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            # print ("lora weight init {} {}".format(torch.mean(self.lora_A.weight), torch.mean(self.lora_B.weight)))

    def forward(self, x):
        orig_type = x.dtype
        res = super().forward(x)

        dim = int(res.shape[-1] // 2)

        r1 = res[..., :dim]
        r2 = res[..., dim:]

        r1 = r1.float()
        r2 = r2.float()
        x_ = x.float()

        tmp = self.lora_B(self.lora_A(
            self.lora_dropout(x_))) * self.lora_scaling
        tmp1 = tmp[..., ::2]
        tmp2 = tmp[..., 1::2]

        r1 += tmp1
        r2 += tmp2

        r1 = r1.to(orig_type)
        r2 = r2.to(orig_type)

        res = torch.cat([r1, r2], -1)

        # res += self.lora_B(self.lora_A(
        #     self.lora_dropout(x))) * self.lora_scaling
        return res


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size,
                      dtype: torch.dtype,
                      device: torch.device,
                      past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len),
                      torch.tensor(torch.finfo(dtype).min, device=device),
                      device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([
            torch.zeros(
                tgt_len, past_key_values_length, dtype=dtype, device=device),
            mask
        ],
                         dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len,
                                         tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor,
                 dtype: torch.dtype,
                 tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len,
                                                  src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool),
                                     torch.finfo(dtype).min)


class InternLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        InternLMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1,
                                                               keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class InternLMRotaryEmbedding(torch.nn.Module):
    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None):
        super().__init__()
        inv_freq = 1.0 / (base
                          **(torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached,
                         device=self.inv_freq.device,
                         dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos()[None, None, :, :],
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin()[None, None, :, :],
                             persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached,
                             device=x.device,
                             dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached",
                                 emb.cos()[None, None, :, :],
                                 persistent=False)
            self.register_buffer("sin_cached",
                                 emb.sin()[None, None, :, :],
                                 persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2,
                       gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2,
                       gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class InternLMMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 hidden_act: str, config: InternLMXComposerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        if config.lora_cfg is not None and 'ffn' in config.lora_cfg[
                'learn_param']:
            lora_cfg = config.lora_cfg
            self.down_proj = LoRALinear(intermediate_size,
                                        hidden_size,
                                        bias=False,
                                        **lora_cfg)
            self.up_proj = LoRALinear(hidden_size,
                                      intermediate_size,
                                      bias=False,
                                      **lora_cfg)
        else:
            self.down_proj = nn.Linear(intermediate_size,
                                       hidden_size,
                                       bias=False)
            self.up_proj = nn.Linear(hidden_size,
                                     intermediate_size,
                                     bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class InternLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: InternLMXComposerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")
        if config.lora_cfg is None:
            self.q_proj = nn.Linear(self.hidden_size,
                                    self.num_heads * self.head_dim,
                                    bias=False)
            self.k_proj = nn.Linear(self.hidden_size,
                                    self.num_heads * self.head_dim,
                                    bias=False)
            self.v_proj = nn.Linear(self.hidden_size,
                                    self.num_heads * self.head_dim,
                                    bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                    self.hidden_size,
                                    bias=False)
        else:
            lora_cfg = config.lora_cfg
            if 'q' in lora_cfg['learn_param']:
                self.q_proj = LoRALinear(self.hidden_size,
                                         self.num_heads * self.head_dim,
                                         bias=False,
                                         **lora_cfg)
            else:
                self.q_proj = nn.Linear(self.hidden_size,
                                        self.num_heads * self.head_dim,
                                        bias=False)

            if 'k' in lora_cfg['learn_param']:
                self.k_proj = LoRALinear(self.hidden_size,
                                         self.num_heads * self.head_dim,
                                         bias=False,
                                         **lora_cfg)
            else:
                self.k_proj = nn.Linear(self.hidden_size,
                                        self.num_heads * self.head_dim,
                                        bias=False)

            if 'v' in lora_cfg['learn_param']:
                self.v_proj = LoRALinear(self.hidden_size,
                                         self.num_heads * self.head_dim,
                                         bias=False,
                                         **lora_cfg)
            else:
                self.v_proj = nn.Linear(self.hidden_size,
                                        self.num_heads * self.head_dim,
                                        bias=False)

            if 'o' in lora_cfg['learn_param']:
                self.o_proj = LoRALinear(self.num_heads * self.head_dim,
                                         self.hidden_size,
                                         bias=False,
                                         **lora_cfg)
            else:
                self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                        self.hidden_size,
                                        bias=False)

        self.rotary_emb = InternLMRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class InternLMDecoderLayer(nn.Module):
    def __init__(self, config: InternLMXComposerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        if hasattr(config,
                   'intern_converted_llm') and config.intern_converted_llm:
            self.self_attn = InternConvertedInternLMAttention(config=config)
        else:
            self.self_attn = InternLMAttention(config=config)
        self.mlp = InternLMMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            config=config,
        )
        self.input_layernorm = InternLMRMSNorm(config.hidden_size,
                                               eps=config.rms_norm_eps)
        self.post_attention_layernorm = InternLMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


class InternLMPreTrainedModel(PreTrainedModel):
    config_class = InternLMXComposerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternLMDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, InternLMModel):
            module.gradient_checkpointing = value


class InternLMModel(InternLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`InternLMDecoderLayer`]

    Args:
        config: InternLMXComposerConfig
    """
    def __init__(self, config: InternLMXComposerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList([
            InternLMDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = InternLMRMSNorm(config.hidden_size,
                                    eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask,
                                              inputs_embeds.dtype,
                                              tgt_len=input_shape[-1]).to(
                                                  inputs_embeds.device)
            combined_attention_mask = (expanded_attn_mask
                                       if combined_attention_mask is None else
                                       expanded_attn_mask +
                                       combined_attention_mask)

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        query_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if query_embeds is not None:
            inputs_embeds = torch.cat([query_embeds, inputs_embeds], dim=1)
            batch_size, seq_length, _ = inputs_embeds.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length,
                                        seq_length + past_key_values_length,
                                        dtype=torch.long,
                                        device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        dtype=torch.bool,
                                        device=inputs_embeds.device)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds,
            past_key_values_length)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            past_key_value = past_key_values[
                idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1], )

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class InternLMForCausalLM(InternLMPreTrainedModel):
    lora_cfg = None  # init in MiniGPT4

    def __init__(self, config):
        super().__init__(config)
        # TODO: find a way to explicitly initialize InternLM
        setattr(config, 'lora_cfg', self.lora_cfg)

        if hasattr(config, 'kqvo_bias'):
            setattr(config, 'kqvo_bias', config.kqvo_bias)
        else:
            setattr(config, 'kqvo_bias', False)
        self.model = InternLMModel(config)

        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)
        if hasattr(config, 'ex_size'):
            self.ex_size = config.ex_size
        else:
            self.ex_size = 0

        if hasattr(config, 'sp_id'):
            self.sp_id = config.sp_id
        else:
            self.sp_id = -1

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        llm_cfg=None,
                        *model_args,
                        **kwargs):
        if llm_cfg:
            if 'torch_dtype' in kwargs:
                llm_cfg.torch_dtype = kwargs['torch_dtype']
            if 'load_in_8bit' in kwargs:
                llm_cfg.load_in_8bit = kwargs['load_in_8bit']
            if 'device_map' in kwargs:
                llm_cfg.device_map = kwargs['device_map']
            return cls._from_config(llm_cfg)
        else:
            return super().from_pretrained(pretrained_model_name_or_path,
                                           *model_args, **kwargs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        query_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, InternLMForCausalLM

        >>> model = InternLMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            query_embeds=query_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens

            loss_fct = CrossEntropyLoss(reduce=False)
            loss_reduce = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            ###
            if self.sp_id >= 0:
                ori_mask = (shift_labels != self.sp_id).float()
                ori_mask = ori_mask * (shift_labels >= 0).float()
                local_mask = (shift_labels == self.sp_id).float()
            else:
                ori_mask = (shift_labels <
                            self.config.vocab_size - self.ex_size).float()
                ori_mask = ori_mask * (shift_labels >= 0).float()
                local_mask = (shift_labels >=
                              self.config.vocab_size - self.ex_size).float()

            # Enable model parallelism

            loss = loss_reduce(shift_logits, shift_labels)

            loss_all = loss_fct(shift_logits, shift_labels)
            loss_o = (loss_all * ori_mask).sum() / ori_mask.sum()
            if torch.sum(local_mask) == 0:
                loss_l = loss_o * 0
            else:
                loss_l = (loss_all * local_mask).sum() / local_mask.sum()

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        if (self.ex_size > 0 or self.sp_id >= 0) and labels is not None:
            return loss, loss_o, loss_l

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      query_embeds=None,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
                query_embeds = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "query_embeds": query_embeds,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past), )
        return reordered_past
