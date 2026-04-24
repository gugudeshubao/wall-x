"""Pure PyTorch replacement for flash_attn.ops.triton.rotary (no triton needed).
Drop-in for Jetson/aarch64 where triton is unavailable.
"""
from typing import Optional, Union
import torch


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Pure PyTorch apply_rotary (drop-in replacement for triton version).
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
              else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: same shape as x
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen

    seqlen_ro, half_rotary_dim = cos.shape
    rotary_dim = half_rotary_dim * 2
    assert rotary_dim <= headdim

    if conjugate:
        sin = -sin

    # --- build per-position cos/sin ---
    if not is_varlen:
        if isinstance(seqlen_offsets, int):
            # cos_slice: (seqlen, half_rotary_dim)
            cos_slice = cos[seqlen_offsets : seqlen_offsets + seqlen]
            sin_slice = sin[seqlen_offsets : seqlen_offsets + seqlen]
            # -> (1, seqlen, 1, half_rotary_dim)
            cos_slice = cos_slice.unsqueeze(0).unsqueeze(2)
            sin_slice = sin_slice.unsqueeze(0).unsqueeze(2)
        else:
            # seqlen_offsets: (batch,)  per-batch offsets
            arange = torch.arange(seqlen, device=x.device).unsqueeze(0)  # (1, seqlen)
            offsets = seqlen_offsets.unsqueeze(1)  # (batch, 1)
            indices = (arange + offsets).long()  # (batch, seqlen)
            cos_slice = cos[indices]  # (batch, seqlen, half_rotary_dim)
            sin_slice = sin[indices]
            cos_slice = cos_slice.unsqueeze(2)  # (batch, seqlen, 1, half_rotary_dim)
            sin_slice = sin_slice.unsqueeze(2)
    else:
        # Variable-length: handle each sequence in the batch
        positions = torch.zeros(total_seqlen, device=x.device, dtype=torch.long)
        for i in range(batch):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len_i = end - start
            offset = seqlen_offsets[i].item() if isinstance(seqlen_offsets, torch.Tensor) else seqlen_offsets
            positions[start:end] = torch.arange(seq_len_i, device=x.device) + offset
        cos_slice = cos[positions].unsqueeze(1)  # (total_seqlen, 1, half_rotary_dim)
        sin_slice = sin[positions].unsqueeze(1)

    # --- apply rotary ---
    x_rot = x[..., :rotary_dim]
    if not interleaved:
        x1 = x_rot[..., :half_rotary_dim]
        x2 = x_rot[..., half_rotary_dim:]
        o1 = x1 * cos_slice - x2 * sin_slice
        o2 = x2 * cos_slice + x1 * sin_slice
        out_rot = torch.cat([o1, o2], dim=-1)
    else:
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        o1 = x1 * cos_slice - x2 * sin_slice
        o2 = x2 * cos_slice + x1 * sin_slice
        out_rot = torch.stack([o1, o2], dim=-1).flatten(-2)

    if inplace:
        x[..., :rotary_dim] = out_rot
        return x
    else:
        if rotary_dim < headdim:
            return torch.cat([out_rot, x[..., rotary_dim:]], dim=-1)
        else:
            return out_rot
