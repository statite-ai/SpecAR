# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Adapted from Krea Realtime (https://github.com/krea-ai/realtime-video, CC BY-NC-SA 4.0): the generation loop,
# KV-cache layout and loaders follow Krea's session code so that SpecAR outputs match it bit for bit.
"""SpecAR decoder for Krea Realtime 14B (target) with the Self-Forcing 1.3B model as drafter.

Per block b of H blocks (3 latent frames each, 5 denoising steps by default):
  1. rebuild the drafter's context KV cache from the committed latents (the target's only on target blocks)
  2. draw the block's step noises from the session RNG, always, so the RNG stream does not depend on the schedule
  3. target block (b in the schedule): the 14B target denoises it; otherwise the 1.3B drafter does
  4. commit the latents, VAE-decode, emit frames
"""
import os
import sys
import time
from collections import deque
from pathlib import Path

import torch

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor", "krea")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
os.environ.setdefault("DISABLE_SAGEATTENTION", "1")   # SageAttention changes numerics


def _install_attention(backend=None):
    """Route every attention call in the vendored Krea/Wan code through specar.attention, which picks the kernel for
    this GPU (FlashAttention 3 on Hopper, 4 on Blackwell; see attention.py). Call after MODEL_FOLDER is set."""
    import importlib
    from . import attention as A
    name = A.select(backend)
    if sys.path[0] != _VENDOR:          # a kernel package may have put its own `utils` ahead of Krea's
        if _VENDOR in sys.path:
            sys.path.remove(_VENDOR)
        sys.path.insert(0, _VENDOR)
    importlib.import_module("wan.modules.attention")
    for mod_name, mod in list(sys.modules.items()):
        if mod_name == "wan" or mod_name.startswith("wan."):
            if hasattr(mod, "flash_attention"):
                mod.flash_attention = A.flash_attention
            for flag in ("FLASH_ATTN_2_AVAILABLE", "FLASH_ATTN_3_AVAILABLE"):
                if hasattr(mod, flag):
                    setattr(mod, flag, True)
    return name


DEV = "cuda"
FRAME_SEQ_LENGTH = 1560          # tokens per latent frame at 832x480 (Krea pipeline constant)
NUM_FRAME_PER_BLOCK = 3
CROSSATTN_LEN = 512
TARGET_CKPT = "krea-realtime-video-14b.safetensors"
DRAFTER_CKPT = "self_forcing_dmd.pt"


def _install_config_only_loader():
    """The 14B weights come from Krea's state dict, so Wan2.1-T2V-14B/ holds only config.json: build from config."""
    from wan.modules.causal_model import CausalWanModel
    if getattr(CausalWanModel, "_specar_config_only", False):
        return
    orig = CausalWanModel.from_pretrained.__func__

    def from_pretrained(cls, path, **kw):
        if any(f.startswith("diffusion_pytorch_model") for f in os.listdir(path)):
            return orig(cls, path, **kw)
        return cls.from_config(cls.load_config(path), **kw)

    CausalWanModel.from_pretrained = classmethod(from_pretrained)
    CausalWanModel._specar_config_only = True


# ---------------------------------------------------------------- model loading (Krea's loaders)
def load_state_dict_auto(checkpoint_path):
    from safetensors.torch import load_file
    p = str(checkpoint_path)
    suffix = Path(p).suffix.lower()
    if suffix in {".safetensors", ".sft"}:
        return load_file(p, device="cpu")
    obj = torch.load(p, map_location="cpu")
    if isinstance(obj, dict) and any(k.startswith("model.") or k.startswith("module.") for k in obj):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "generator", "generator_ema", "net"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    raise RuntimeError(f"unrecognised checkpoint structure in {p}")


def load_wan_wrapper(checkpoint_path, timestep_shift=5.0):
    from utils.wan_wrapper import WanDiffusionWrapper
    sd = load_state_dict_auto(checkpoint_path)
    k0 = sd["model.blocks.0.self_attn.k.weight"]
    model_name = "Wan2.1-T2V-1.3B" if k0.shape[0] == 1536 else "Wan2.1-T2V-14B"
    w = WanDiffusionWrapper(model_name=model_name, timestep_shift=timestep_shift, is_causal=True)
    w.load_state_dict(sd)
    w = w.to(dtype=torch.bfloat16)
    w.eval()
    w.requires_grad_(False)
    w.to(DEV)
    for block in w.model.blocks:
        block.self_attn.fuse_projections()
    return w


def load_text_encoder():
    from utils.wan_wrapper import WanTextEncoder
    te = WanTextEncoder()
    te.eval()
    te.to(dtype=torch.bfloat16)
    te.requires_grad_(False)
    te.to(DEV)
    return te


def load_vae():
    from demo_utils.vae_block3 import VAEEncoderWrapper, VAEDecoderWrapper
    from wan.modules.vae import WanVAE
    vae_path = os.path.join(os.environ["MODEL_FOLDER"], "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth")
    vae = WanVAE(vae_pth=vae_path, dtype=torch.float16, device=DEV)
    vae_encoder = VAEEncoderWrapper(vae)
    vae_decoder = VAEDecoderWrapper()
    sd = torch.load(vae_path, map_location="cpu")
    vae_decoder.load_state_dict({k: v for k, v in sd.items() if "decoder." in k or "conv2" in k}, strict=False)
    for m in (vae_encoder, vae_decoder):
        m.eval()
        m.to(dtype=torch.float16)
        m.requires_grad_(False)
        m.to(DEV)
    return vae_encoder, vae_decoder


class Models:
    """Loads the target, drafter, text encoder and VAE from one weights directory (see `specar download`)."""

    def __init__(self, weights, timestep_shift=5.0, attention=None):
        weights = os.path.abspath(weights)
        os.environ["MODEL_FOLDER"] = weights          # read by the vendored Krea code at import time
        self.attention = _install_attention(attention)
        _install_config_only_loader()
        self.text_encoder = load_text_encoder()
        self.vae_encoder, self.vae_decoder = load_vae()
        self.target = load_wan_wrapper(os.path.join(weights, TARGET_CKPT), timestep_shift)
        self.drafter = load_wan_wrapper(os.path.join(weights, DRAFTER_CKPT), timestep_shift)

    def encode_prompts(self, prompts):
        """Prompt embeddings computed once (same numbers as computing them inside the session)."""
        return {p: {k: v.clone() for k, v in self.text_encoder(text_prompts=[p]).items()} for p in prompts}


# ---------------------------------------------------------------- KV caches (shapes exactly as Krea's pipeline allocates them)
def _kv_cache(model_wrapper, n_tokens, device, dtype=torch.bfloat16):
    cfg = model_wrapper.model.config
    shape = [1, n_tokens, cfg.num_heads, cfg.dim // cfg.num_heads]
    return [{"k": torch.zeros(shape, dtype=dtype, device=device).contiguous(),
             "v": torch.zeros(shape, dtype=dtype, device=device).contiguous(),
             "global_end_index": 0, "local_end_index": 0} for _ in model_wrapper.model.blocks]


def _crossattn_cache(model_wrapper, device, dtype=torch.bfloat16):
    cfg = model_wrapper.model.config
    shape = [1, CROSSATTN_LEN, cfg.num_heads, cfg.dim // cfg.num_heads]
    return [{"k": torch.zeros(shape, dtype=dtype, device=device).contiguous(),
             "v": torch.zeros(shape, dtype=dtype, device=device).contiguous(),
             "is_init": False} for _ in model_wrapper.model.blocks]


def _zero_kv(cache):
    for c in cache:
        c["k"].zero_(); c["v"].zero_(); c["global_end_index"] = 0; c["local_end_index"] = 0


# ---------------------------------------------------------------- the session
class SpecARSession:
    def __init__(self, models, prompt, seed, num_blocks, anchors, prompt_embeds=None, width=832, height=480,
                 kv_cache_num_frames=3, num_denoising_steps=5, timestep_shift=5.0, strength=1.0,
                 keep_first_frame=False, lazy_target_kv=True, skip_draft_on_anchors=True, frame_callback=None):
        from utils.scheduler import FlowMatchScheduler
        from v2v import get_denoising_schedule
        self.models = models
        self.prompt = prompt
        self.anchors = set(anchors)
        self.num_blocks = num_blocks
        self.kv_cache_num_frames = kv_cache_num_frames
        self.keep_first_frame = keep_first_frame
        self.lazy_target_kv = lazy_target_kv
        self.skip_draft_on_anchors = skip_draft_on_anchors
        self.frame_callback = frame_callback or (lambda *a: None)
        self.gpu = torch.cuda.current_device()
        self.width, self.height = width // 8 * 8, height // 8 * 8
        self.latent_width, self.latent_height = self.width // 8, self.height // 8
        self.nfpb = NUM_FRAME_PER_BLOCK
        self.block_idx = 0
        self.current_start_frame = 0
        self.frame_context_cache = deque(maxlen=1 + (kv_cache_num_frames - 1) * 4)
        self.encode_vae_cache = [None] * 55
        self.decode_vae_cache = [None] * 55
        self.counts = {}          # block -> forward counts {"D_ctx","D","T_ctx","T"}
        self.block_times = {}     # block -> seconds (cuda-synced)
        # RNG and noise: identical draw order to Krea's session (noise tensor first, then per-block step noises)
        self.rnd = torch.Generator(self.gpu).manual_seed(seed)
        latent_shape = [1, num_blocks * self.nfpb, 16, self.latent_height, self.latent_width]
        self.all_latents = torch.zeros(latent_shape, device=self.gpu, dtype=torch.bfloat16).contiguous()
        self.noise = torch.randn(latent_shape, device=self.gpu, dtype=torch.bfloat16, generator=self.rnd).contiguous()
        # attention window and caches, as init_models does
        attn_size = kv_cache_num_frames + self.nfpb
        for block in models.target.model.blocks:
            block.self_attn.local_attn_size = -1
        self.local_attn_size = attn_size
        n_tokens = attn_size * FRAME_SEQ_LENGTH
        self.target_kv = _kv_cache(models.target, n_tokens, self.gpu)
        self.target_ca = _crossattn_cache(models.target, self.gpu)
        self.drafter_kv = _kv_cache(models.drafter, n_tokens, self.gpu)
        self.drafter_ca = _crossattn_cache(models.drafter, self.gpu)
        models.target.model.block_mask = None
        # scheduler and denoising schedule (strength 1.0 -> [1000, 937.5, 833.3, 625, 0] for 5 steps)
        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        st = self.scheduler.timesteps
        zero_padded = torch.cat((st.cpu(), torch.tensor([0], dtype=torch.float32))).to(self.gpu)
        self.denoising_step_list = get_denoising_schedule(zero_padded, strength, steps=num_denoising_steps)
        # prompt conditioning (bf16, contiguous), computed here or passed in from a one-time encode
        cond = prompt_embeds if prompt_embeds is not None else models.text_encoder(text_prompts=[prompt])
        self.conditional_dict = {k: v.to(dtype=torch.bfloat16).contiguous() for k, v in cond.items()}
        self.current_prompt_embeds = self.conditional_dict["prompt_embeds"]
        self.last_pred = None

    # ---- context ------------------------------------------------------------------------------------------
    def _clean_context_frames(self):
        from v2v import encode_video_latent
        n = self.kv_cache_num_frames
        clean = self.all_latents[:, :self.current_start_frame]
        if self.keep_first_frame or (self.block_idx - 1) * self.nfpb < n:
            if n == 1:
                return clean[:, :1]
            return torch.cat((clean[:, :1], clean[:, 1:][:, -n + 1:]), dim=1)
        clean = clean[:, 1:][:, -n + 1:]
        frame_for_vae = self.frame_context_cache[0][0].half().to(DEV)
        first = encode_video_latent(self.models.vae_encoder, [None] * 55, resample_to=16, max_frames=81,
                                    video_path_or_url=None, frames=frame_for_vae, height=480, width=832,
                                    stream=False)[0].transpose(0, 1)[None]
        return torch.cat((first.to(DEV), clean), dim=1).to(self.all_latents)

    def _build_target_context(self):
        """Target context forward over the committed clean latents (kv cache rebuilt from zero)."""
        m = self.models.target
        _zero_kv(self.target_kv)
        m.model.block_mask = self._ctx_block_mask
        m(noisy_image_or_video=self._ctx_clean, conditional_dict=self._ctx_cond, timestep=self._ctx_timestep,
          kv_cache=self.target_kv, crossattn_cache=self.target_ca,
          current_start=self._ctx_start * FRAME_SEQ_LENGTH)
        m.model.block_mask = None
        self.counts[self.block_idx]["T_ctx"] += 1

    def _recompute_context(self):
        """Returns model_input_start_frame; mirrors recompute_kv_cache + the drafter context forward."""
        if self.block_idx == 0:
            _zero_kv(self.target_kv)
            _zero_kv(self.drafter_kv)
            return self.current_start_frame
        for block in self.models.target.model.blocks:
            block.self_attn.num_frame_per_block = self.nfpb
        start = min(self.current_start_frame, self.kv_cache_num_frames)
        clean = self._clean_context_frames()
        _zero_kv(self.drafter_kv)
        mask = self.models.target.model._prepare_blockwise_causal_attn_mask(
            device=str(clean.device), num_frames=clean.shape[1], frame_seqlen=FRAME_SEQ_LENGTH,
            num_frame_per_block=self.nfpb, local_attn_size=-1)
        ctx_t = torch.ones([clean.shape[0], clean.shape[1]], device=clean.device, dtype=torch.int64) * 0
        cond = {k: v.to(DEV, non_blocking=True) for k, v in self.conditional_dict.items()}
        self._ctx_clean, self._ctx_timestep, self._ctx_start, self._ctx_block_mask, self._ctx_cond = clean, ctx_t, start, mask, cond
        if not self.lazy_target_kv:
            self._build_target_context()
        d = self.models.drafter
        d.model.block_mask = mask
        d(noisy_image_or_video=clean, conditional_dict=cond, timestep=ctx_t, kv_cache=self.drafter_kv,
          crossattn_cache=self.drafter_ca, current_start=start * FRAME_SEQ_LENGTH)
        d.model.block_mask = None
        self.counts[self.block_idx]["D_ctx"] += 1
        return start

    # ---- denoising -----------------------------------------------------------------------------------------
    def _denoise(self, model, kv, ca, start, noisy_input, step_noises, cond, key):
        steps = self.denoising_step_list
        pred = None
        for index, t in enumerate(steps):
            timestep = torch.ones([1, self.nfpb], device=noisy_input.device, dtype=torch.int64) * t
            cond["prompt_embeds"] = self.current_prompt_embeds.to(DEV, non_blocking=True)
            _, pred = model(noisy_image_or_video=noisy_input, conditional_dict=cond, timestep=timestep,
                            kv_cache=kv, crossattn_cache=ca, current_start=start * FRAME_SEQ_LENGTH)
            self.counts[self.block_idx][key] += 1
            if index < len(steps) - 1:
                nxt = steps[index + 1]
                noisy_input = self.scheduler.add_noise(
                    pred.flatten(0, 1), step_noises[index],
                    nxt * torch.ones([1 * self.nfpb], device=noisy_input.device, dtype=torch.long),
                ).unflatten(0, pred.shape[:2])
        return pred

    @torch.inference_mode()
    def generate_block(self):
        idx = self.block_idx
        if idx >= self.num_blocks:
            return None
        t0 = time.perf_counter()
        self.counts[idx] = {"D_ctx": 0, "D": 0, "T_ctx": 0, "T": 0}
        start = self._recompute_context()
        s, e = self.current_start_frame, self.current_start_frame + self.nfpb
        original = self.noise[:, s:e]
        cond = {k: v.to(DEV, non_blocking=True) for k, v in self.conditional_dict.items()}
        noisy_input = original.clone()
        step_noises = [torch.randn(*noisy_input.flatten(0, 1).shape, generator=self.rnd, device=noisy_input.device,
                                   dtype=torch.bfloat16) for _ in range(len(self.denoising_step_list) - 1)]
        is_anchor = idx in self.anchors
        pred = None
        if not (is_anchor and self.skip_draft_on_anchors):
            pred = self._denoise(self.models.drafter, self.drafter_kv, self.drafter_ca, start, noisy_input, step_noises, cond, "D")
        if is_anchor:
            if self.lazy_target_kv and idx != 0:
                self._build_target_context()
            pred = self._denoise(self.models.target, self.target_kv, self.target_ca, start, original.clone(), step_noises, cond, "T")
        self.all_latents[:, s:e] = pred
        self.last_pred = pred
        pixels, self.decode_vae_cache = self.models.vae_decoder(pred.to(DEV, non_blocking=True).half(), *self.decode_vae_cache)
        pixels = pixels.to(DEV, non_blocking=True)
        self.frame_context_cache.extend(pixels.split(1, dim=1))
        if idx == 0:
            pixels = pixels[:, 3:]
        event = torch.cuda.Event()
        event.record()
        self.frame_callback(pixels, event)
        torch.cuda.synchronize()
        self.block_times[idx] = time.perf_counter() - t0
        self.current_start_frame += self.nfpb
        self.block_idx += 1
        return pixels

    def generate(self):
        """Run all blocks; returns the committed latents [1, 3H, 16, h, w]."""
        for _ in range(self.num_blocks):
            self.generate_block()
        return self.all_latents
