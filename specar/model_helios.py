# SPDX-License-Identifier: Apache-2.0
"""SpecAR decoder for Helios 14B: Helios-Base generates the target chunks, Helios-Distilled the others.

Helios-Base's own pipeline call runs unchanged (history windows, first-frame anchor, per-chunk noise draws, per-chunk
VAE decode); only its per-chunk sampler is replaced:
  target chunk  -> Base's stage-1 sampler (50 steps, classifier-free guidance 5.0)
  drafted chunk -> Distilled's stage-2 sampler (pyramid [2, 2, 2], guidance 1.0, amplified first chunk)
Both samplers start from the noise the native loop drew for that chunk.
"""
import os
import sys
import time

import numpy as np
import torch

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor", "helios")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, "
    "worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn "
    "faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in "
    "the background, walking backwards")
HEIGHT, WIDTH = 384, 640
FRAMES_PER_CHUNK = 33
DRAFT = dict(pyramid_num_stages=3, pyramid_num_inference_steps_list=[2, 2, 2])


def _transformer(path):
    from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
    from helios.modules.helios_kernels import replace_all_norms_with_flash_norms, replace_rmsnorm_with_fp32
    tr = HeliosTransformer3DModel.from_pretrained(path, subfolder="transformer", torch_dtype=torch.bfloat16)
    tr = replace_rmsnorm_with_fp32(tr)
    tr = replace_all_norms_with_flash_norms(tr)
    try:
        tr.set_attention_backend("_flash_3_hub")
    except Exception:
        tr.set_attention_backend("flash_hub")
    return tr


def _to_uint8(v):
    return ((v[0].float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


class HeliosSpecAR:
    """Loads Helios-Base and Helios-Distilled from `weights` (see `specar download --backbone helios`).

    The text encoder is used once, for `prompts`, and then freed so both 14B transformers fit on one 80 GB GPU.
    """

    def __init__(self, weights, prompts):
        from diffusers.models import AutoencoderKLWan
        from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
        from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
        from helios.modules.helios_kernels import replace_rope_with_flash_rope
        torch.set_grad_enabled(False)
        mb, md = os.path.join(weights, "Helios-Base"), os.path.join(weights, "Helios-Distilled")
        self.vae = AutoencoderKLWan.from_pretrained(mb, subfolder="vae", torch_dtype=torch.float32)
        self.base = HeliosPipeline.from_pretrained(mb, transformer=_transformer(mb), vae=self.vae,
                                                   scheduler=HeliosScheduler.from_pretrained(mb, subfolder="scheduler"),
                                                   torch_dtype=torch.bfloat16).to("cuda")
        replace_rope_with_flash_rope()
        self.embeds = {}
        for p in dict.fromkeys(prompts):
            pe, ne = self.base.encode_prompt(prompt=p, negative_prompt=NEGATIVE_PROMPT, do_classifier_free_guidance=True,
                                             device="cuda")
            self.embeds[p] = (pe.to(torch.bfloat16), ne.to(torch.bfloat16))
        self.base.text_encoder = None
        torch.cuda.empty_cache()
        self.dist = HeliosPipeline.from_pretrained(md, transformer=_transformer(md), vae=self.vae, text_encoder=None,
                                                   tokenizer=self.base.tokenizer,
                                                   scheduler=HeliosScheduler.from_pretrained(md, subfolder="scheduler"),
                                                   torch_dtype=torch.bfloat16).to("cuda")
        for p in (self.base, self.dist):
            p.set_progress_bar_config(disable=True)
        self._stage1 = self.base.stage1_sample
        self._decode = self.vae.decode
        self._chunks = None
        self.vae.decode = self._capture_decode

    def _capture_decode(self, *args, **kw):
        v = self._decode(*args, **kw)
        if self._chunks is not None:
            self._chunks.append(v[0])
        return v

    def _draft(self, kw, chunk):
        d = self.dist
        d._guidance_scale, d._attention_kwargs, d._current_timestep, d._interrupt = 1.0, kw.get("attention_kwargs"), None, False
        return d.stage2_sample(
            latents=kw["latents"], **DRAFT, prompt_embeds=kw["prompt_embeds"], negative_prompt_embeds=None, guidance_scale=1.0,
            indices_hidden_states=kw["indices_hidden_states"], indices_latents_history_short=kw["indices_latents_history_short"],
            indices_latents_history_mid=kw["indices_latents_history_mid"],
            indices_latents_history_long=kw["indices_latents_history_long"],
            latents_history_short=kw["latents_history_short"], latents_history_mid=kw["latents_history_mid"],
            latents_history_long=kw["latents_history_long"], attention_kwargs=kw.get("attention_kwargs"), device=kw["device"],
            transformer_dtype=kw["transformer_dtype"], use_zero_init=kw["use_zero_init"], zero_steps=kw["zero_steps"],
            is_amplify_first_chunk=(chunk == 0), callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=kw["callback_on_step_end_tensor_inputs"], progress_bar=kw["progress_bar"])

    def generate(self, prompt, anchors, num_chunks, seed):
        """Returns (latents [C, T, h, w], uint8 frames [N, 384, 640, 3], per-chunk seconds)."""
        anchors, times, idx = set(anchors), [], [-1]

        def stage1(**kw):
            idx[0] += 1
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = self._stage1(**kw) if idx[0] in anchors else self._draft(kw, idx[0])
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            return out

        pe, ne = self.embeds[prompt]
        self._chunks = []
        torch.manual_seed(seed)                       # the distilled sampler draws from the global RNG
        self.base.stage1_sample = stage1
        try:
            out = self.base(prompt_embeds=pe, negative_prompt_embeds=ne, height=HEIGHT, width=WIDTH,
                            num_frames=num_chunks * FRAMES_PER_CHUNK, num_inference_steps=50, guidance_scale=5.0,
                            generator=torch.Generator("cuda").manual_seed(seed), output_type="latent")
            chunks, self._chunks = self._chunks, None
        finally:
            self.base.stage1_sample = self._stage1
            self._chunks = None
        latents = out.frames[0]
        assert len(chunks) == latents.shape[1] // 9, (len(chunks), tuple(latents.shape))
        frames = np.concatenate([_to_uint8(c) for c in chunks])
        frames = frames[:(len(frames) - 1) // 4 * 4 + 1]
        return latents, frames, times
