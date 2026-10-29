"""Experimental BF16 gate/up GEMM with a SwiGLU epilogue for Ampere."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@torch.no_grad()
def pack_weights(gate, up, transpose=False):
    # [K, N, 2] pairs keep gate/up results together in the MMA registers.
    if transpose:
        return torch.stack((gate, up), dim=1).flatten(0, 1).contiguous()
    return torch.stack((gate.T, up.T), dim=-1).flatten(1).contiguous()


@gluon.jit
def _copy(a, b, a_smem, b_smem, rows, cols, ka, kb, index,
          M: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
          BK: gl.constexpr, BUFFERS: gl.constexpr, TRANSPOSE: gl.constexpr, CACHE: gl.constexpr):
    ak = index * BK + ka
    bk = index * BK + kb
    cp.async_load(a_smem.index(index % BUFFERS), a + rows[:, None] * K + ak[None, :],
                  mask=(rows[:, None] < M) & (ak[None, :] < K), cache_modifier=CACHE)
    if TRANSPOSE:
        cp.async_load(b_smem.index(index % BUFFERS), b + cols[:, None] * K + bk[None, :],
                      mask=(cols[:, None] < 2 * N) & (bk[None, :] < K), cache_modifier=CACHE)
    else:
        cp.async_load(b_smem.index(index % BUFFERS), b + bk[:, None] * (2 * N) + cols[None, :],
                      mask=(bk[:, None] < K) & (cols[None, :] < 2 * N), cache_modifier=CACHE)
    cp.commit_group()


@gluon.jit
def _mma(a_smem, b_smem, index, acc, A_DOT: gl.constexpr, B_DOT: gl.constexpr,
         BUFFERS: gl.constexpr, TRANSPOSE: gl.constexpr, BK: gl.constexpr, MMA_K: gl.constexpr):
    a_tile = a_smem.index(index % BUFFERS)
    if TRANSPOSE:
        b_tile = b_smem.index(index % BUFFERS).permute((1, 0))
    else:
        b_tile = b_smem.index(index % BUFFERS)
    if MMA_K:
        for start in gl.static_range(0, BK, MMA_K):
            a = a_tile.slice(start, MMA_K, dim=1).load(A_DOT)
            b = b_tile.slice(start, MMA_K, dim=0).load(B_DOT)
            acc = mma_v2(a, b, acc)
    else:
        a = a_tile.load(A_DOT)
        b = b_tile.load(B_DOT)
        acc = mma_v2(a, b, acc)
    return acc


@gluon.jit
def _swiglu(a, b, out, M: gl.constexpr, K: gl.constexpr, N: gl.constexpr,
            BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
            WARPS: gl.constexpr, WM: gl.constexpr, BUFFERS: gl.constexpr,
            SWIZZLE: gl.constexpr, PHASE_A: gl.constexpr, PHASE_B: gl.constexpr,
            TRANSPOSE: gl.constexpr, CACHE: gl.constexpr, MMA_K: gl.constexpr):
    pid = gl.program_id(0)
    grid_m = gl.cdiv(M, BM)
    grid_n = gl.cdiv(2 * N, BN)
    group = pid // (8 * grid_n)
    first_m = group * 8
    group_m = gl.minimum(grid_m - first_m, 8)
    pid_m = first_m + (pid % (8 * grid_n)) % group_m
    pid_n = (pid % (8 * grid_n)) // group_m

    load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [WARPS, 1], [1, 0])
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout([2, 0], [WM, WARPS // WM], [16, 8])
    a_dot: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 2)
    b_dot: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 2)
    a_shared: gl.constexpr = gl.SwizzledSharedLayout(8, PHASE_A, min(SWIZZLE, BK // 8), [1, 0])
    b_shared: gl.constexpr = gl.SwizzledSharedLayout(
        8, PHASE_B, min(SWIZZLE, BK // 8) if TRANSPOSE else SWIZZLE, [1, 0]
    )
    a_smem = gl.allocate_shared_memory(a.dtype.element_ty, [BUFFERS, BM, BK], a_shared)
    if TRANSPOSE:
        b_smem = gl.allocate_shared_memory(b.dtype.element_ty, [BUFFERS, BN, BK], b_shared)
    else:
        b_smem = gl.allocate_shared_memory(b.dtype.element_ty, [BUFFERS, BK, BN], b_shared)

    rows = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, load_layout))
    cols = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(1 if TRANSPOSE else 0, load_layout))
    ka = gl.arange(0, BK, gl.SliceLayout(0, load_layout))
    kb = gl.arange(0, BK, gl.SliceLayout(0 if TRANSPOSE else 1, load_layout))
    acc = gl.full((BM, BN), 0, gl.float32, mma_layout)
    tiles: gl.constexpr = gl.cdiv(K, BK)

    for i in gl.static_range(BUFFERS - 1):
        _copy(a, b, a_smem, b_smem, rows, cols, ka, kb, i, M, K, N, BK, BUFFERS, TRANSPOSE, CACHE)
    for i in range(tiles - (BUFFERS - 1)):
        _copy(a, b, a_smem, b_smem, rows, cols, ka, kb, i + BUFFERS - 1, M, K, N, BK, BUFFERS, TRANSPOSE, CACHE)
        cp.wait_group(BUFFERS - 1)
        acc = _mma(a_smem, b_smem, i, acc, a_dot, b_dot, BUFFERS, TRANSPOSE, BK, MMA_K)
    for i in gl.static_range(BUFFERS - 1):
        cp.wait_group(BUFFERS - 2 - i)
        acc = _mma(a_smem, b_smem, tiles - (BUFFERS - 1) + i, acc, a_dot, b_dot, BUFFERS, TRANSPOSE, BK, MMA_K)

    acc = acc.to(out.dtype.element_ty).to(gl.float32)
    gate, up = gl.split(acc.reshape((BM, BN // 2, 2)))
    activated = (gate / (1.0 + gl.exp(-gate))).to(out.dtype.element_ty).to(gl.float32) * up
    out_layout: gl.constexpr = activated.type.layout
    out_rows = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, out_layout))
    out_cols = pid_n * (BN // 2) + gl.arange(0, BN // 2, gl.SliceLayout(0, out_layout))
    gl.store(out + out_rows[:, None] * N + out_cols[None, :], activated,
             mask=(out_rows[:, None] < M) & (out_cols[None, :] < N))


DEFAULT_CONFIG = (128, 128, 64, 4, 2, 2, 8, 1, 1, True, ".ca")


def launch(x, packed_weight, out, config=DEFAULT_CONFIG, maxnreg=None, mma_k=0):
    m = x.numel() // x.shape[-1]
    bm, bn, bk, warps, wm, buffers, swizzle, phase_a, phase_b, transpose, cache = config
    k = x.shape[-1]
    n2 = packed_weight.shape[0] if transpose else packed_weight.shape[1]
    n = n2 // 2
    options = {"num_warps": warps, "num_stages": 1}
    if maxnreg is not None:
        options["maxnreg"] = maxnreg
    return _swiglu[(triton.cdiv(m, bm) * triton.cdiv(n2, bn),)](
        x, packed_weight, out, m, k, n, bm, bn, bk, warps, wm, buffers, swizzle, phase_a, phase_b, transpose, cache, mma_k,
        **options,
    )


def swiglu(x, packed_weight, config=DEFAULT_CONFIG, maxnreg=None, mma_k=0):
    n = packed_weight.shape[0 if config[9] else 1] // 2
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    launch(x, packed_weight, out, config, maxnreg, mma_k)
    return out


class GluonSwiGLU(torch.nn.Module):
    """Pack the fixed weights once; keep the down projection in PyTorch."""

    def __init__(self, source, config=DEFAULT_CONFIG, maxnreg=None, mma_k=0):
        super().__init__()
        self.config = config
        self.maxnreg = maxnreg
        self.mma_k = mma_k
        self.register_buffer("gate_up_weight", pack_weights(source.gate.weight, source.up.weight, config[9]))
        self.down = source.down

    def project(self, x):
        return swiglu(x.contiguous(), self.gate_up_weight, self.config, self.maxnreg, self.mma_k)

    def forward(self, x):
        return self.down(self.project(x))
