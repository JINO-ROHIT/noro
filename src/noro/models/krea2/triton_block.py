"""Krea-2 block with Triton RMSNorm-modulation fusion."""

from noro.kernels.fused_rms_mod import fused_rms_mod
from noro.models.krea2.torch_block import TorchKrea2TransformerBlock


class TritonKrea2TransformerBlock(TorchKrea2TransformerBlock):
    """Diffusers Krea-2 block with RMSNorm and modulation fused in Triton."""

    def forward(self, hidden_states, temb, image_rotary_emb, attention_mask=None):
        modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        attn_input = fused_rms_mod(
            hidden_states, self.norm1.weight, prescale, preshift, self.norm1.eps
        )
        attn_out = self.attn(
            attn_input,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + pregate * attn_out

        ff_input = fused_rms_mod(
            hidden_states, self.norm2.weight, postscale, postshift, self.norm2.eps
        )
        return hidden_states + postgate * self.ff(ff_input)
