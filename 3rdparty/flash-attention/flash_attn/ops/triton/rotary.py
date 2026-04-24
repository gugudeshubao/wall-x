# Copyright (c) 2025, Tri Dao.
# Modified for Jetson/aarch64: Pure PyTorch fallback (triton unavailable on aarch64).
# Original uses triton >= 3.0 for the rotary kernel. This version provides a
# functionally equivalent PyTorch implementation so flash_attn works on Orin etc.

from typing import Optional, Union

import torch

# ---------- Try to import triton; fall back to pure PyTorch ----------
_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


# =====================================================================
# Pure-PyTorch apply_rotary  (drop-in replacement when triton missing)
# =====================================================================
def _apply_rotary_torch(
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
    Pure PyTorch apply_rotary — same API as the triton version.
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
              else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
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
            cos_slice = cos[seqlen_offsets : seqlen_offsets + seqlen]
            sin_slice = sin[seqlen_offsets : seqlen_offsets + seqlen]
            cos_slice = cos_slice.unsqueeze(0).unsqueeze(2)  # (1,seqlen,1,half)
            sin_slice = sin_slice.unsqueeze(0).unsqueeze(2)
        else:
            arange = torch.arange(seqlen, device=x.device).unsqueeze(0)
            offsets = seqlen_offsets.unsqueeze(1)
            indices = (arange + offsets).long()
            cos_slice = cos[indices].unsqueeze(2)  # (batch,seqlen,1,half)
            sin_slice = sin[indices].unsqueeze(2)
    else:
        positions = torch.zeros(total_seqlen, device=x.device, dtype=torch.long)
        for i in range(batch):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len_i = end - start
            offset = (
                seqlen_offsets[i].item()
                if isinstance(seqlen_offsets, torch.Tensor)
                else seqlen_offsets
            )
            positions[start:end] = torch.arange(seq_len_i, device=x.device) + offset
        cos_slice = cos[positions].unsqueeze(1)  # (total_seqlen,1,half)
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


# =====================================================================
# Triton-based apply_rotary  (original, used when triton IS available)
# =====================================================================
if _HAS_TRITON:

    @triton.jit
    def rotary_kernel(
        OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS,
        seqlen, nheads, seqlen_ro,
        stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim,
        stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
        ROTARY_DIM: tl.constexpr,
        IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        CONJUGATE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
        ROTARY_DIM_HALF = ROTARY_DIM // 2
        pid_head = tl.program_id(axis=0)
        pid_m = tl.program_id(axis=1)
        pid_batch = tl.program_id(axis=2)

        if not IS_VARLEN:
            X = X + pid_batch * stride_x_batch
            OUT = OUT + pid_batch * stride_out_batch
        else:
            start_idx = tl.load(CU_SEQLENS + pid_batch)
            seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
            X = X + start_idx * stride_x_seqlen
            OUT = OUT + start_idx * stride_out_seqlen

        if pid_m * BLOCK_M >= seqlen:
            return

        rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        if not IS_SEQLEN_OFFSETS_TENSOR:
            rm_cs = rm + SEQLEN_OFFSETS
        else:
            rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

        rk_half = tl.arange(0, BLOCK_K // 2)
        COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
        mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
        cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
        sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin

        if not INTERLEAVED:
            X = X + (rh[:, None, None] * stride_x_nheads + rm[None, :, None] * stride_x_seqlen + rk_half[None, None, :] * stride_x_headdim)
            OUT = OUT + (rh[:, None, None] * stride_out_nheads + rm[None, :, None] * stride_out_seqlen + rk_half[None, None, :] * stride_out_headdim)
            mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk_half[None, None, :] < ROTARY_DIM_HALF)
            x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
            x1 = tl.load(X + ROTARY_DIM_HALF * stride_x_headdim, mask=mask, other=0.0).to(tl.float32)
            o0 = x0 * cos - x1 * sin
            o1 = x0 * sin + x1 * cos
            tl.store(OUT, o0, mask=mask)
            tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
        else:
            rk = tl.arange(0, BLOCK_K)
            X = X + (rh[:, None, None] * stride_x_nheads + rm[None, :, None] * stride_x_seqlen + rk[None, None, :] * stride_x_headdim)
            OUT = OUT + (rh[:, None, None] * stride_out_nheads + rm[None, :, None] * stride_out_seqlen + rk[None, None, :] * stride_out_headdim)
            mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk[None, None, :] < ROTARY_DIM)
            x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
            x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
            o0 = x0 * cos - x1 * sin
            o1 = x0 * sin + x1 * cos
            o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
            tl.store(OUT, o, mask=mask)

    def _apply_rotary_triton(
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
        is_varlen = cu_seqlens is not None
        if not is_varlen:
            batch, seqlen, nheads, headdim = x.shape
        else:
            assert max_seqlen is not None
            total_seqlen, nheads, headdim = x.shape
            batch = cu_seqlens.shape[0] - 1
            seqlen = max_seqlen
        seqlen_ro, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen_ro >= seqlen

        cos, sin = cos.contiguous(), sin.contiguous()
        if isinstance(seqlen_offsets, torch.Tensor):
            assert seqlen_offsets.shape == (batch,)
            seqlen_offsets = seqlen_offsets.contiguous()
        else:
            assert seqlen_offsets + seqlen <= seqlen_ro

        output = torch.empty_like(x) if not inplace else x
        if rotary_dim < headdim and not inplace:
            output[..., rotary_dim:].copy_(x[..., rotary_dim:])

        grid = lambda META: (triton.cdiv(nheads, META["BLOCK_H"]), triton.cdiv(seqlen, META["BLOCK_M"]), batch)
        BLOCK_M = 8 if rotary_dim <= 128 else 4

        with torch.cuda.device(x.device.index):
            torch.library.wrap_triton(rotary_kernel)[grid](
                output, x, cos, sin, cu_seqlens, seqlen_offsets,
                seqlen, nheads, seqlen_ro,
                output.stride(0) if not is_varlen else 0,
                output.stride(-3), output.stride(-2), output.stride(-1),
                x.stride(0) if not is_varlen else 0,
                x.stride(-3), x.stride(-2), x.stride(-1),
                rotary_dim,
                isinstance(seqlen_offsets, torch.Tensor),
                is_varlen, interleaved, conjugate,
                BLOCK_M=BLOCK_M, BLOCK_H=2,
            )
        return output


# =====================================================================
# Public API — auto-selects triton or PyTorch backend
# =====================================================================
apply_rotary = _apply_rotary_triton if _HAS_TRITON else _apply_rotary_torch
