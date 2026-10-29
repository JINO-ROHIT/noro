import torch
import triton
import triton.language as tl


@triton.jit
def rms_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0).to(tl.float32) # we just have weight of shape [N]

    mean_square = tl.sum(x * x, axis=0) / N
    inv_rms = tl.rsqrt(mean_square + eps)
    y = x * inv_rms * (1.0 + weight)

    tl.store(out_ptr + row * N + cols, y, mask=mask)


def rms_norm(
    hidden_states: torch.Tensor, # (batch, seq_len, hidden_size)
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """krea 2 rmsnorm which does fp32 rms and zero centered scale."""

    N = hidden_states.shape[-1]

    x = hidden_states.reshape(-1, N).to(torch.float32).contiguous() # (bs * seq len, hidden size)
    weight_f32 = weight.to(torch.float32).contiguous()
    output = torch.empty_like(x)

    rows = x.shape[0]
    if rows:
        rms_kernel[(rows,)](
            x,
            weight_f32,
            output,
            N,
            eps,
            BLOCK_SIZE=triton.next_power_of_2(N),
        )

    return output.reshape(hidden_states.shape).to(hidden_states.dtype)
