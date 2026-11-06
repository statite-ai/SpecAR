# SPDX-License-Identifier: Apache-2.0
"""Attention backend selection and compatibility for the Krea backbone.

The generation engine calls one function, flash_attention(), with the signature Wan's code uses. Which kernel runs it
is chosen once per process from the GPU's compute capability and the kernels that are installed:

  fa3   FlashAttention 3   Hopper (H100, H200)       the backend behind the reference latents
  fa4   FlashAttention 4   Blackwell (B200, B300)    also runs on Hopper
  fa2   FlashAttention 2   any GPU it was built for  fallback

Override the choice with SPECAR_ATTENTION=fa2|fa3|fa4 or `specar generate --attention ...`. The backends do not share
one call signature; the differences are handled here and nowhere else.
"""
import importlib
import importlib.util
import os

BACKENDS = ("fa3", "fa4", "fa2")
SUPPORTED = {"fa3": (9,), "fa4": (9, 10), "fa2": None}        # compute-capability majors; None = any
PREFERENCE = {9: ("fa3", "fa4", "fa2"), 10: ("fa4", "fa2")}    # tried in order; any other GPU: fa2
HOW_TO_INSTALL = {
    "fa3": "pip install kernels  (prebuilt FlashAttention 3 from the Hugging Face kernel hub, Hopper GPUs)",
    "fa4": "pip install flash-attn-4  (FlashAttention 4, Blackwell and Hopper GPUs)",
    "fa2": "pip install flash-attn  (FlashAttention 2, compiled for your GPU)",
}
_state = {"name": None, "call": None}


def _has(module):
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:       # a parent package that fails to import
        return False


def installed():
    """Which backends can be loaded in this environment."""
    return {
        "fa3": _has("flash_attn_interface") or _has("kernels"),
        "fa4": _has("cutlass") and _has("flash_attn.cute"),
        "fa2": _has("flash_attn_2_cuda"),
    }


def choose(capability_major, available, override=None, gpu_name="this GPU"):
    """The backend to use, or a RuntimeError saying what is missing. Pure logic: needs no GPU."""
    override = (override or "auto").lower()
    if override != "auto":
        if override not in BACKENDS:
            raise ValueError(f"unknown attention backend {override!r}; choose auto, {', '.join(BACKENDS)}")
        majors = SUPPORTED[override]
        if majors is not None and capability_major not in majors:
            raise RuntimeError(f"attention backend {override} does not support {gpu_name} "
                               f"(compute capability {capability_major}.x)")
        if not available.get(override):
            raise RuntimeError(f"attention backend {override} is not installed: {HOW_TO_INSTALL[override]}")
        return override
    order = PREFERENCE.get(capability_major, ("fa2",))
    for name in order:
        if available.get(name):
            return name
    hints = "\n".join(f"  {name}: {HOW_TO_INSTALL[name]}" for name in order)
    raise RuntimeError(f"no attention backend is installed for {gpu_name} (compute capability {capability_major}.x). "
                       f"Install one of:\n{hints}")


def _only_plain(name, dropout_p, window_size, deterministic=False):
    if dropout_p or tuple(window_size) != (-1, -1) or deterministic:
        raise NotImplementedError(f"attention backend {name} does not support dropout, sliding windows or "
                                  f"deterministic mode; use SPECAR_ATTENTION=fa2")


def _tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _load(name):
    """A callable with one signature for every backend."""
    if name == "fa3":
        try:
            fa3 = importlib.import_module("flash_attn_interface")
        except ImportError:
            from kernels import get_kernel
            fa3 = get_kernel("kernels-community/flash-attn3")

        def call(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal, dropout_p, window_size, deterministic):
            _only_plain("fa3", dropout_p, window_size)
            return _tensor(fa3.flash_attn_varlen_func(
                q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_q, max_seqlen_k=max_k,
                softmax_scale=softmax_scale, causal=causal, deterministic=deterministic))
    elif name == "fa4":
        fa4 = importlib.import_module("flash_attn.cute")

        def call(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal, dropout_p, window_size, deterministic):
            _only_plain("fa4", dropout_p, window_size, deterministic)
            return _tensor(fa4.flash_attn_varlen_func(
                q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, softmax_scale=softmax_scale, causal=causal))
    else:
        fa2 = importlib.import_module("flash_attn")

        def call(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal, dropout_p, window_size, deterministic):
            return _tensor(fa2.flash_attn_varlen_func(
                q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_q, max_seqlen_k=max_k,
                dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, window_size=window_size,
                deterministic=deterministic))
    return call


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("SpecAR needs an NVIDIA GPU with CUDA; none is visible to PyTorch")
    return {"name": torch.cuda.get_device_name(0), "capability": tuple(torch.cuda.get_device_capability(0))}


def select(override=None):
    """Choose and load the backend (once per process; an explicit override re-selects). Returns its name."""
    override = override or os.environ.get("SPECAR_ATTENTION")
    if _state["name"] and (not override or override.lower() in ("auto", _state["name"])):
        return _state["name"]
    g = gpu()
    name = choose(g["capability"][0], installed(), override, g["name"])
    _state["call"], _state["name"] = _load(name), name
    return name


def active():
    return _state["name"]


def describe():
    """What a benchmark or bug report needs: GPU, backend and the versions behind them."""
    import torch
    from importlib import metadata
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for dist in ("flash-attn", "flash-attn-4", "nvidia-cutlass-dsl", "kernels", "diffusers", "transformers"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    g = gpu()
    return {"gpu": g["name"], "compute_capability": "%d.%d" % g["capability"], "attention": active(), "versions": versions}


def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None, causal=False,
                    window_size=(-1, -1), deterministic=False, dtype=None, version=None):
    """Drop-in for Wan's flash_attention. q [B, Lq, N, C], k and v [B, Lk, N, C]; q_lens, k_lens [B] or None.
    `version` is accepted for compatibility and ignored: the backend is the one select() chose."""
    import torch
    if _state["call"] is None:
        select()
    half = (torch.float16, torch.bfloat16)
    dtype = dtype or torch.bfloat16
    assert dtype in half and q.device.type == "cuda" and q.size(-1) <= 256
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def cast(x):
        return x if x.dtype in half else x.to(dtype)

    if q_lens is None:
        q = cast(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = cast(torch.cat([u[:n] for u, n in zip(q, q_lens)]))
    if k_lens is None:
        k = cast(k.flatten(0, 1))
        v = cast(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = cast(torch.cat([u[:n] for u, n in zip(k, k_lens)]))
        v = cast(torch.cat([u[:n] for u, n in zip(v, k_lens)]))
    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale
    cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
    cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
    x = _state["call"](q, k, v, cu_q, cu_k, lq, lk, softmax_scale, causal, dropout_p, window_size, deterministic)
    return x.unflatten(0, (b, lq)).type(out_dtype)
