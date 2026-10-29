import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rms_mod_kernel(
    x_ptr, weight_ptr, scale_ptr, shift_ptr, out_ptr,
    N: tl.constexpr, ROWS_PER_BATCH: tl.constexpr,
    EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask, 0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask, 0).to(tl.float32)
    batch = row // ROWS_PER_BATCH
    scale = tl.load(scale_ptr + batch * N + cols, mask, 0).to(tl.float32)
    shift = tl.load(shift_ptr + batch * N + cols, mask, 0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    normalized = (x * inv_rms * (1.0 + weight)).to(x_ptr.dtype.element_ty).to(tl.float32)
    y = (normalized * (1.0 + scale)).to(x_ptr.dtype.element_ty).to(tl.float32) + shift
    tl.store(out_ptr + row * N + cols, y, mask)


def fused_rms_mod(x, weight, scale, shift, eps):
    """fuse Krea rmsnorm with its following scale/shift modulation. --> y = (1 + scale) * rmsnorm(x) + shift"""
    n = x.shape[-1]
    rows_per_batch = x.shape[-2]
    x2 = x.reshape(-1, n).contiguous()
    scale = scale.reshape(x.shape[0], n).contiguous()
    shift = shift.reshape(x.shape[0], n).contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(x2)
    rows = x2.shape[0]
    _fused_rms_mod_kernel[(rows,)](
        x2, weight, scale, shift, out, n, rows_per_batch, eps,
        triton.next_power_of_2(n), num_warps=8,
    )
    return out.reshape(x.shape)
