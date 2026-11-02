# Vendored Krea Realtime code

The files in this folder are copied from [krea-ai/realtime-video](https://github.com/krea-ai/realtime-video)
at commit `acf4b7b4e1a3049a0e04782e751d12203334038d`, which builds on
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) and [Wan 2.1](https://github.com/Wan-Video/Wan2.1).
They are licensed under CC BY-NC-SA 4.0 (`LICENSE.md` in this folder). Only the files SpecAR imports are included.

One change from upstream, marked `SpecAR patch` in `utils/wan_wrapper.py`: the T5 text encoder is also read from
`models_t5_umt5-xxl-enc-bf16.pth`, the format published in `Wan-AI/Wan2.1-T2V-1.3B`, when no `.safetensors` file
is present. The loaded weights are the same.
