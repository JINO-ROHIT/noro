"""Diffusers style Krea-2 pipeline"""

# gotta make this a lot more cleaner lol

from __future__ import annotations

from typing import Any


class Krea2Pipeline:
    def __init__(self, pipeline: Any, backend: str = "diffusers", *, compile: bool = False):
        self._pipeline = pipeline
        self.backend = backend
        self._compiled = False
        if compile:
            self.compile()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        backend: str = "diffusers",
        compile: bool = False,
        **kwargs,
    ):
        if backend not in ("diffusers", "triton"):
            raise ValueError("backend must be 'diffusers' or 'triton'")
        from diffusers import Krea2Pipeline as DiffusersKrea2Pipeline

        pipeline = DiffusersKrea2Pipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if backend == "triton":
            cls._replace_transformer_blocks(pipeline)
        return cls(pipeline, backend, compile=compile)

    def compile(self):
        """torch compile step"""
        if not self._compiled:
            for block in self._pipeline.transformer.transformer_blocks:
                block.compile(mode="reduce-overhead", fullgraph=True) # compile as full graph, check if other alts work better
            self._compiled = True
        return self

    @staticmethod
    def _replace_transformer_blocks(pipeline) -> None:
        import gc

        import torch

        from noro.models.krea2.triton_block import TritonKrea2TransformerBlock

        transformer = pipeline.transformer
        config = transformer.config
        original_blocks = transformer.transformer_blocks
        blocks = torch.nn.ModuleList()
        for original in original_blocks:
            with torch.device("meta"): # we use meta so we dont have to load the weights yet
                block = TritonKrea2TransformerBlock(
                    hidden_size=config.attention_head_dim * config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    norm_eps=config.norm_eps,
                )
            block.load_state_dict(original.state_dict(), strict=True, assign=True)
            blocks.append(block)
        transformer.transformer_blocks = blocks
        del original_blocks, block
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def enable_gluon_swiglu(self):
        """to enable the experimental fused FFN"""
        if self.backend != "triton":
            raise ValueError("Gluon SwiGLU requires backend='triton'")

        import torch

        blocks = self._pipeline.transformer.transformer_blocks
        for block in blocks:
            weight = next(block.ff.parameters())
            if weight.device.type != "cuda" or weight.dtype != torch.bfloat16:
                raise ValueError("Move the pipeline to CUDA with BF16 weights before enabling Gluon SwiGLU")
            if torch.cuda.get_device_capability(weight.device)[0] < 8:
                raise ValueError("Gluon SwiGLU requires an Ampere or newer NVIDIA GPU")

        from noro.kernels.swiglu_gluon import GluonSwiGLU

        for block in blocks:
            if not isinstance(block.ff, GluonSwiGLU):
                block.ff = GluonSwiGLU(block.ff)
        return self

    def to(self, *args, **kwargs):
        self._pipeline.to(*args, **kwargs)
        return self

    def __getattr__(self, name):
        # keep access to familiar Diffusers components and helpers, e.g. pipe.vae.
        pipeline = self.__dict__.get("_pipeline")
        if pipeline is None:
            raise AttributeError(name)
        return getattr(pipeline, name)

    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        sigmas=None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=("latents",),
        attention_kwargs=None,
        max_sequence_length: int = 512,
    ):
        """Generate images with the selected backend."""
        if self.backend == "diffusers":
            return self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                output_type=output_type,
                return_dict=return_dict,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=list(callback_on_step_end_tensor_inputs),
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
            )

        if callback_on_step_end is not None:
            raise NotImplementedError("callback_on_step_end is not supported by the Triton backend yet")
        if attention_kwargs is not None:
            raise NotImplementedError("attention_kwargs is not supported by the Triton backend yet")
        if guidance_scale != 0.0:
            raise NotImplementedError("The Triton backend currently supports guidance_scale=0.0 only")
        import torch

        if self._compiled:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.inference_mode():
            return self._triton_call(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                output_type=output_type,
                return_dict=return_dict,
                max_sequence_length=max_sequence_length,
            )

    def _triton_call(
        self,
        *,
        prompt,
        negative_prompt,
        height,
        width,
        num_inference_steps,
        sigmas,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        output_type,
        return_dict,
        max_sequence_length,
    ):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift
        from diffusers.pipelines.krea2.pipeline_output import Krea2PipelineOutput

        pipe = self._pipeline
        multiple = pipe.vae_scale_factor * pipe.patch_size
        if height % multiple or width % multiple:
            height = ((height + multiple - 1) // multiple) * multiple
            width = ((width + multiple - 1) // multiple) * multiple
        pipe.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            max_sequence_length=max_sequence_length,
        )

        pipe._guidance_scale = guidance_scale
        pipe._attention_kwargs = None
        pipe._current_timestep = None
        pipe._interrupt = False
        if prompt is not None:
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = pipe._execution_device

        prompt_embeds, prompt_mask = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            max_sequence_length=max_sequence_length,
        )
        transformer = pipe.transformer
        effective_batch = prompt_embeds.shape[0]
        channels = transformer.config.in_channels // (pipe.patch_size**2)
        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        grid_height = height // (pipe.vae_scale_factor * pipe.patch_size)
        grid_width = width // (pipe.vae_scale_factor * pipe.patch_size)
        position_ids = pipe.prepare_position_ids(prompt_embeds.shape[1], grid_height, grid_width, device)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if pipe.config.is_distilled:
            mu = 1.15
        else:
            scheduler_config = pipe.scheduler.config
            mu = calculate_shift(
                latents.shape[1],
                scheduler_config.get("base_image_seq_len", 256),
                scheduler_config.get("max_image_seq_len", 6400),
                scheduler_config.get("base_shift", 0.5),
                scheduler_config.get("max_shift", 1.15),
            )
        pipe.scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
        timesteps = pipe.scheduler.timesteps
        pipe._num_timesteps = len(timesteps)
        pipe.scheduler.set_begin_index(0)

        image_rotary_emb = transformer.rotary_emb(position_ids)

        def prepare_text(embeds, mask):
            fused = transformer.text_fusion(embeds, attention_mask=mask[:, None, None, :])
            states = transformer.txt_in(fused)
            full_mask = torch.cat(
                [mask, mask.new_ones((effective_batch, latents.shape[1]))], dim=1
            )[:, None, None, :]
            return states, full_mask

        text_states, attention_mask = prepare_text(prompt_embeds, prompt_mask)

        time_data = []
        for timestep in timesteps:
            t = (timestep / pipe.scheduler.config.num_train_timesteps).expand(effective_batch).to(latents.dtype)
            temb = transformer.time_embed(t, dtype=latents.dtype)
            temb_mod = transformer.time_mod_proj(F.gelu(temb, approximate="tanh"))
            time_data.append((temb, temb_mod))

        def predict(text, mask, temb, temb_mod):
            hidden = torch.cat([text, transformer.img_in(latents)], dim=1)
            for block in transformer.transformer_blocks:
                hidden = block(hidden, temb_mod, image_rotary_emb, mask)
            return transformer.final_layer(hidden[:, prompt_mask.shape[1]:], temb)

        for timestep, (temb, temb_mod) in zip(timesteps, time_data):
            pipe._current_timestep = timestep
            velocity = predict(text_states, attention_mask, temb, temb_mod)
            latents = pipe.scheduler.step(velocity, timestep, latents, return_dict=False)[0]

        pipe._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = pipe._unpack_latents(latents, height, width).to(pipe.vae.dtype)
            z_dim = pipe.vae.config.z_dim
            mean = torch.tensor(pipe.vae.config.latents_mean, device=device, dtype=latents.dtype).view(
                1, z_dim, 1, 1, 1
            )
            inv_std = (1.0 / torch.tensor(pipe.vae.config.latents_std, device=device, dtype=latents.dtype)).view(
                1, z_dim, 1, 1, 1
            )
            latents = latents / inv_std + mean
            decoded = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = pipe.image_processor.postprocess(decoded, output_type=output_type)

        pipe.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return Krea2PipelineOutput(images=image)
