# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_block import TransformerBlock

import torch
from torch import Tensor, einsum, nn

from megatron.core import parallel_state
from megatron import get_args

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

_GLOBAL_BUFFERS = {}
def get_global_buffers():
    return _GLOBAL_BUFFERS


def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=pos_emb.device)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


def _duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

class RotaryEmbedding(nn.Module):
    """Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None
    """

    def __init__(
        self, kv_channels: int, rotary_percent: float, seq_len_interpolation_factor: float = None
    ) -> None:
        super().__init__()

        args = get_args()
        # we will use this as a cache for the sequence length
        # it's possible that in forward() we will get a different sequence length, but we will
        # only recompute the rotary embeddings if the sequence length has changed
        self.cache_seq_length = args.seq_length 
        self.rotary_interleave_repeat = args.rotary_interleave_repeat
        self.stay_fp32 = args.rotary_stay_fp32

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)) 
        
        # TODO: at this moment inv_freq is fp32
        # but with --bf16, it will be casted to bf16 and cause precision loss
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        seq = torch.arange(self.cache_seq_length, device=self.inv_freq.device).float()
        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor
        assert seq.dtype == torch.float32 and self.inv_freq.dtype == torch.float32, f"seq.dtype: {seq.dtype}, inv_freq.dtype: {self.inv_freq.dtype}"
        freqs = einsum('i , j -> i j', seq, self.inv_freq) 
        assert freqs.dtype == torch.float32, f"freqs.dtype: {freqs.dtype}"

        if self.rotary_interleave_repeat:
            sinusoid_inp = _duplicate_interleave(freqs)  # in the last dim [f1, f2, ... , fn] --> [f1, f1, f2, f2, ... , fn, fn]
        else:
            sinusoid_inp = torch.cat((freqs, freqs), dim=-1) # in the last dim [f1, f2, ... , fn] --> [f1, f2, ... , fn, f1, f2, ... , fn]

        cached_sin, cached_cos = (
            torch.sin(sinusoid_inp),
            torch.cos(sinusoid_inp),
        )

        # self.register_buffer('sinusoid_inp', sinusoid_inp, persistent=False)
        # self.register_buffer('cached_sin', cached_sin, persistent=False)
        # self.register_buffer('cached_cos', cached_cos, persistent=False)

        # store the precomputed values in a global buffer so that 
        # it will not be automatically converted to bf16

        global _GLOBAL_BUFFERS
        _GLOBAL_BUFFERS['cached_sin'] = cached_sin.to(torch.cuda.current_device())
        _GLOBAL_BUFFERS['cached_cos'] = cached_cos.to(torch.cuda.current_device())
        _GLOBAL_BUFFERS['sinusoid_inp'] = sinusoid_inp.to(torch.cuda.current_device())

                # after this point, cached_sin and cached_cos can be fp16 or bf16
        if args.fp16:
            _GLOBAL_BUFFERS['cached_sin'] = _GLOBAL_BUFFERS['cached_sin'].half()
            _GLOBAL_BUFFERS['cached_cos'] = _GLOBAL_BUFFERS['cached_cos'].half()
        elif args.bf16:
            _GLOBAL_BUFFERS['cached_sin'] = _GLOBAL_BUFFERS['cached_sin'].bfloat16()
            _GLOBAL_BUFFERS['cached_cos'] = _GLOBAL_BUFFERS['cached_cos'].bfloat16()


    def forward(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): _description_. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        # use the cached pre-computed cos and sin value
        if max_seq_len == self.cache_seq_length and offset == 0:
            buffers = get_global_buffers()
            cached_sin = buffers['cached_sin']
            cached_cos = buffers['cached_cos']
            sinusoid_inp = buffers['sinusoid_inp']
            emb = sinusoid_inp
        else:
            seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
            if self.seq_len_interpolation_factor is not None:
                seq = seq.type_as(self.inv_freq)
                seq *= 1 / self.seq_len_interpolation_factor
            # freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
            freqs = einsum('i , j -> i j', seq.float(), self.inv_freq)  # bf16 can lower precision, e.g, 1996.0 --> 2000.0 in bf16
            # first part even vector components, second part odd vector components,
            #  2 * dim in dimension size
            
            if self.rotary_interleave_repeat:
                emb = _duplicate_interleave(freqs)  # in the last dim [f1, f2, ... , fn] --> [f1, f1, f2, f2, ... , fn, fn]
            else:
                emb = torch.cat((freqs, freqs), dim=-1) # in the last dim [f1, f2, ... , fn] --> [f1, f2, ... , fn, f1, f2, ... , fn]
            # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        if parallel_state.get_context_parallel_world_size() > 1:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0)
        return emb

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f'{prefix}inv_freq', None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_rotary_seq_len(
        self,
        inference_params,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
    ) -> float:
        """Function to get the rotary sequence length.

        Args:
            inference_params : Used during Inference time
            transformer (TransformerBlock): The transformer block (decoder/encoder) used by the model
            transformer_input (Tensor): _description_
            transformer_config (TransformerConfig): Transformer config used by the model

        Returns:
            float: The rotary sequence length
        """
        if inference_params is not None:
            rotary_seq_len = inference_params.max_sequence_length
        else:
            if transformer.input_tensor is not None:
                rotary_seq_len = transformer.input_tensor.size(0)
            else:
                rotary_seq_len = transformer_input.size(0)

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

        rotary_seq_len *= transformer_config.context_parallel_size

        return rotary_seq_len


def _rotate_half(x: Tensor) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """

    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def _rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    return x.flatten(-2)

def apply_rotary_pos_emb(t: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # if we use the interleaved version, we need to use a different rotation function
    args = get_args()
    rotate_func = _rotate_every_two if args.rotary_interleave_repeat else _rotate_half

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method

    if 'apply_rotary_pos_emb_freqs' in get_global_buffers() and freqs.size() == get_global_buffers()['apply_rotary_pos_emb_freqs'].size():
        # use the cached pre-computed cos and sin value
        cos = get_global_buffers()['apply_rotary_pos_emb_cached_cos']
        sin = get_global_buffers()['apply_rotary_pos_emb_cached_sin']
    else:
        # compute cos and sin on the fly
        cos = freqs.cos()
        sin = freqs.sin()
        if args.fp16:
            cos = cos.half()
            sin = sin.half()
        elif args.bf16:
            cos = cos.bfloat16()
            sin = sin.bfloat16()
        # store the precomputed values in a global buffer
        buffers = get_global_buffers()
        buffers['apply_rotary_pos_emb_freqs'] = freqs
        buffers['apply_rotary_pos_emb_cached_cos'] = cos
        buffers['apply_rotary_pos_emb_cached_sin'] = sin

    t = (t * cos) + (rotate_func(t) * sin)
    return torch.cat((t, t_pass), dim=-1)
