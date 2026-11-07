# SPDX-License-Identifier: Apache-2.0
"""SpecAR decoder for LongLive-2.0 5B: the 4-step generator (target) generates the k scheduled blocks, the 2-step
generator (the same network with NVIDIA's 2-step DMD LoRA merged, shipped as `model_te.pt`) drafts the rest.

LongLive's own `CausalDiffusionInferencePipeline.inference` runs unchanged (block loop, UniPC scheduler, rolling KV
cache with sink, VAE). Two seams are patched for the duration of a call:
  pipe.generator                    -> a dispatcher that routes each forward to the target or the drafter
  pipe._initialize_sample_scheduler -> the 4-step scheduler on target blocks, the 2-step one otherwise
                                       (LongLive calls it once per block, right before the block's denoising loop)

A LongLive block is 8 latent frames = 32 video frames at 24 fps (the first block gives 29). Target and drafter
share one KV cache (`kv_mode="shared"`): the drafter writes its blocks' keys and values with its own clean pass, the
target writes the scheduled blocks', and each reads whatever is there. Two slower variants keep separate caches and
rebuild the target's before each scheduled block: `replay` (clean-pass every drafted block since the last scheduled
one; exact, the target cache is bit for bit what a target-only run would hold for those latents) and `window` (only
the last 3 blocks, the attention window; exact when the gap is at most 3 blocks).
"""
import contextlib
import os
import sys
import time

import torch

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor", "longlive")
# Inside LongLive's own repository its modules (pipeline, utils, wan_5b) are already importable: use those.
_HOST = "wan_5b" in sys.modules
if not _HOST and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

FPS = 24
NUM_FRAME_PER_BLOCK = 8
TARGET_CKPT = os.path.join("LongLive-2.0-5B", "model_bf16.pt")
DRAFTER_CKPT = os.path.join("LongLive-2.0-5B-S2", "model_te.pt")
WAN_DIR = os.path.join("wan_models", "Wan2.2-TI2V-5B")     # the path LongLive's loaders use, relative to the weights dir
TARGET_STEPS, DRAFTER_STEPS = 4, 2


def frames_for_blocks(blocks):
    return (NUM_FRAME_PER_BLOCK * blocks - 1) * 4 + 1


def _install_attention(backend=None):
    """LongLive's attention module picks FlashAttention 2 unless told otherwise. On Hopper, give it FlashAttention 3
    from the Hugging Face kernel hub under the module name it imports (`flash_attn_interface`); on Blackwell its own
    FlashAttention 4 path (`flash_attn.cute`). Returns the kernel name."""
    from . import attention as A
    name = A.select(backend)
    if name == "fa3":
        if "flash_attn_interface" not in sys.modules:
            try:
                import flash_attn_interface  # noqa: F401  (a source build of FlashAttention 3)
            except ImportError:
                from kernels import get_kernel
                sys.modules["flash_attn_interface"] = get_kernel("kernels-community/flash-attn3")
        os.environ["LLV2_USE_FA3"] = "1"
        os.environ.pop("LLV2_USE_FA4", None)
    elif name == "fa4":
        os.environ["LLV2_USE_FA4"] = "1"
        os.environ.pop("LLV2_USE_FA3", None)
    else:
        os.environ.pop("LLV2_USE_FA3", None)
        os.environ.pop("LLV2_USE_FA4", None)
    if not _HOST and sys.path[0] != _VENDOR:
        if _VENDOR in sys.path:
            sys.path.remove(_VENDOR)
        sys.path.insert(0, _VENDOR)
    return name


@contextlib.contextmanager
def _in_dir(path):
    """LongLive's loaders read `wan_models/Wan2.2-TI2V-5B/...` relative to the working directory."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class Models:
    """Loads LongLive's pipeline (target generator, T5, VAE) and the 2-step drafter from `weights`
    (see `specar download --backbone longlive`)."""

    def __init__(self, weights, attention=None):
        from omegaconf import OmegaConf
        self.weights = os.path.abspath(weights)
        for p in (TARGET_CKPT, DRAFTER_CKPT, os.path.join(WAN_DIR, "Wan2.2_VAE.pth")):
            if not os.path.exists(os.path.join(self.weights, p)):
                raise FileNotFoundError(f"{p} not found in {self.weights}; run: specar download --backbone longlive")
        self.attention = _install_attention(attention)
        from pipeline import CausalDiffusionInferencePipeline
        from utils.config import normalize_config
        from utils.inference_utils import load_generator_checkpoint
        from utils.wan_5b_wrapper import WanDiffusionWrapper
        torch.set_grad_enabled(False)
        config = normalize_config(OmegaConf.load(os.path.join(_VENDOR, "configs", "inference.yaml")))
        config.inference.streaming_vae = False
        config.inference.vae_device = None
        config.inference.sampling_steps = TARGET_STEPS
        self.config = normalize_config(config)
        self.device = torch.device("cuda")
        with _in_dir(self.weights):
            self.pipe = CausalDiffusionInferencePipeline(self.config, device=self.device)
            load_generator_checkpoint(self.pipe.generator, os.path.join(self.weights, TARGET_CKPT))
            self.pipe = self.pipe.to(device=self.device, dtype=torch.bfloat16)
            self.pipe.generator.model.eval().requires_grad_(False)
            self.drafter = WanDiffusionWrapper(**self.config.model_kwargs, is_causal=True)
            load_generator_checkpoint(self.drafter, os.path.join(self.weights, DRAFTER_CKPT))
            self.drafter = self.drafter.to(device=self.device, dtype=torch.bfloat16)
            self.drafter.model.eval().requires_grad_(False)
        self.target = self.pipe.generator


class _Dispatcher(torch.nn.Module):
    """Stands in for pipe.generator (a registered child module of the pipeline); attribute reads it does not own go
    to the target wrapper, which the pipeline inspects (`.model`, named_modules)."""

    def __init__(self, engine):
        super().__init__()
        object.__setattr__(self, "_e", engine)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._e.target, name)

    def forward(self, **kw):
        return self._e._forward(**kw)


class LongLiveSpecAR:
    """The decoder. `generate(prompt, anchors, num_blocks, seed)` -> (latents [1, 8H, 48, h, w], uint8 frames
    [N, 720, 1280, 3], per-block seconds). `kv_mode` may be changed between videos."""

    def __init__(self, models, kv_mode="shared"):
        assert kv_mode in ("shared", "replay", "window"), kv_mode
        self.models, self.pipe, self.target, self.drafter = models, models.pipe, models.target, models.drafter
        self.kv_mode = kv_mode
        pipe = self.pipe
        self.nfpb, self.fsl = pipe.num_frame_per_block, pipe.frame_seq_length
        self.bt = self.nfpb * self.fsl
        self.sink_tokens = (pipe.sink_size or 0) * self.fsl
        self.window_blocks = pipe.local_attn_size // self.nfpb - 1
        dev, dt = next(self.target.parameters()).device, next(self.target.parameters()).dtype
        # the drafter gets the runtime settings inference() puts on the target's model, and its own caches
        self.drafter.model.num_frame_per_block = self.nfpb
        saved = pipe.generator
        pipe.generator = self.drafter
        try:
            pipe._set_all_modules_max_attention_size(pipe.local_attn_size)
            if pipe.sink_size is not None:
                pipe._set_all_modules_sink_size(pipe.sink_size)
            pipe._set_all_modules_global_sink_size(pipe.global_sink_size)
            dd = pipe._dit_model
            dd.local_attn_size = pipe.local_attn_size
            dd.use_relative_rope = pipe.use_relative_rope
            dd.rope_temporal_offset = 0.0
            if pipe.inference_t_scale is not None:
                dd.t_scale = pipe.inference_t_scale
            if pipe._rope_method_override is not None:
                dd.rope_method = pipe._rope_method_override
            if pipe._original_seq_len_override is not None:
                dd.original_seq_len = pipe._original_seq_len_override
            pipe.kv_cache_pos = pipe.crossattn_cache_pos = None
            pipe._initialize_crossattn_cache(batch_size=1, dtype=dt, device=dev)
            self.drafter_ca = pipe.crossattn_cache_pos
            self.drafter_kv = None                                        # built on first use of replay/window
            pipe.kv_cache_pos = pipe.crossattn_cache_pos = None
        finally:
            pipe.generator = saved
        self._orig_sched = pipe._initialize_sample_scheduler
        self.dispatcher = _Dispatcher(self)

    # ---- the pipeline seams -----------------------------------------------------------------------------
    def _install(self):
        self.pipe.generator = self.dispatcher
        self.pipe._initialize_sample_scheduler = self._scheduler

    def _remove(self):
        self.pipe.generator = self.target
        self.pipe._initialize_sample_scheduler = self._orig_sched

    def _begin(self, anchors, num_blocks):
        self.anchors, self.H, self.block = set(anchors), num_blocks, -1
        self.latents, self.target_last = {}, -1
        self.counts, self.t_start, self.block_s = {}, {}, {}
        for c in self.drafter_ca:
            c["is_init"] = False
        if self.kv_mode != "shared":
            if self.drafter_kv is None:
                dev, dt = next(self.target.parameters()).device, next(self.target.parameters()).dtype
                saved = self.pipe.kv_cache_pos
                self.pipe.kv_cache_pos = None
                self.pipe._initialize_kv_cache(batch_size=1, dtype=dt, device=dev)
                self.drafter_kv, self.pipe.kv_cache_pos = self.pipe.kv_cache_pos, saved
            for kv in self.drafter_kv:
                d = kv["global_end_index"].device
                kv["global_end_index"] = torch.tensor([0], dtype=torch.long, device=d)
                kv["local_end_index"] = torch.tensor([0], dtype=torch.long, device=d)
                kv["pinned_start"].fill_(-1)
                kv["pinned_len"].zero_()

    def _scheduler(self, noise):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self.block >= 0:
            self.block_s[self.block] = now - self.t_start[self.block]
        self.block += 1
        b = self.block
        self.t_start[b] = now
        self.counts[b] = {"T": 0, "T_clean": 0, "T_rebuild": 0, "D": 0, "D_clean": 0}
        for c in self.drafter_ca:
            c["is_init"] = False
        if b in self.anchors:
            return self._orig_sched(noise)
        saved = self.pipe.sampling_steps
        self.pipe.sampling_steps = DRAFTER_STEPS
        try:
            return self._orig_sched(noise)
        finally:
            self.pipe.sampling_steps = saved

    def _finish(self):
        torch.cuda.synchronize()
        if self.block >= 0 and self.block not in self.block_s:
            self.block_s[self.block] = time.perf_counter() - self.t_start[self.block]
        return [self.block_s[b] for b in range(self.H)]

    def _forward(self, noisy_image_or_video, conditional_dict, timestep, kv_cache, crossattn_cache, current_start,
                 cache_start=None, **extra):
        b = current_start // self.bt
        assert b == self.block, (b, self.block)
        clean = bool((timestep == 0).all())
        target_kv, target_ca = kv_cache, crossattn_cache
        d_kv = target_kv if self.kv_mode == "shared" else self.drafter_kv
        call = dict(conditional_dict=conditional_dict, timestep=timestep, current_start=current_start,
                    cache_start=cache_start, **extra)
        if b in self.anchors:
            if not clean and self.counts[b]["T"] == 0 and self.kv_mode != "shared":
                self._rebuild_target(b, target_kv, target_ca, conditional_dict, cache_start)
            out = self.target(noisy_image_or_video=noisy_image_or_video, kv_cache=target_kv, crossattn_cache=target_ca, **call)
            self.counts[b]["T_clean" if clean else "T"] += 1
            if clean:
                self.latents[b] = noisy_image_or_video
                self.target_last = b
                if self.kv_mode != "shared" and any(j not in self.anchors for j in range(b + 1, self.H)):
                    # separate caches: the drafter's must hold every block, so it clean-passes the target's too
                    self.drafter(noisy_image_or_video=noisy_image_or_video, kv_cache=d_kv, crossattn_cache=self.drafter_ca, **call)
                    self.counts[b]["D_clean"] += 1
            return out
        out = self.drafter(noisy_image_or_video=noisy_image_or_video, kv_cache=d_kv, crossattn_cache=self.drafter_ca, **call)
        self.counts[b]["D_clean" if clean else "D"] += 1
        if clean:
            self.latents[b] = noisy_image_or_video
        return out

    def _rebuild_target(self, b, target_kv, target_ca, conditional_dict, cache_start):
        first_missing = self.target_last + 1
        if first_missing >= b:
            return
        start = first_missing
        if self.kv_mode == "window" and b - start > self.window_blocks:
            start = b - self.window_blocks
            for kv in target_kv:              # keep the sink only; the next clean pass lands right after it
                d = kv["global_end_index"].device
                kv["global_end_index"] = torch.tensor([start * self.bt], dtype=torch.long, device=d)
                kv["local_end_index"] = torch.tensor([self.sink_tokens], dtype=torch.long, device=d)
        for blk in range(start, b):
            lat = self.latents[blk]
            ts = torch.zeros([lat.shape[0], lat.shape[1]], device=lat.device, dtype=torch.float32)
            self.target(noisy_image_or_video=lat, conditional_dict=conditional_dict, timestep=ts, kv_cache=target_kv,
                        crossattn_cache=target_ca, current_start=blk * self.bt,
                        cache_start=(blk * self.bt) if cache_start is not None else None)
            self.counts[b]["T_rebuild"] += 1
        self.target_last = b - 1

    # ---- public ----------------------------------------------------------------------------------------------
    @contextlib.contextmanager
    def wrap(self, anchors, num_blocks):
        """Make the next `pipe.inference(...)` call a SpecAR run (for hosts that drive LongLive's pipeline
        themselves, such as LongLive's inference.py): `with engine.wrap(uniform_k(2, 24), 24): pipe.inference(...)`.
        Afterwards `engine.block_seconds` holds the per-block times."""
        self._begin(anchors, num_blocks)
        self._install()
        try:
            yield self
        finally:
            self._remove()
            self.block_seconds = self._finish()

    def generate(self, prompt, anchors, num_blocks, seed, decode=True):
        """anchors: the target's blocks (e.g. uniform_k(k, num_blocks)); an empty set drafts every block."""
        from utils.inference_utils import prepare_single_prompt_inputs, video_to_uint8
        self.models.config.num_output_frames = num_blocks * self.nfpb
        gen = torch.Generator(device=self.models.device).manual_seed(seed)
        noise, prompts = prepare_single_prompt_inputs(self.models.config, prompt, self.models.device, generator=gen)
        with self.wrap(anchors, num_blocks):
            latents = self.pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
        times = self.block_seconds
        frames = None
        if decode:
            video = self.pipe.vae.decode_to_pixel(latents)
            frames = video_to_uint8((video * 0.5 + 0.5).clamp(0, 1)).numpy()          # [T, 720, 1280, 3]
            del video
        return latents, frames, times
