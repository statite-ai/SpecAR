# SpecAR

SpecAR speeds up block-autoregressive video generation by letting a small drafter generate most blocks and the
large target generate k blocks spread evenly over the video. This repository runs SpecAR on
[Krea Realtime 14B](https://huggingface.co/krea/krea-realtime-video) as the target with the
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) 1.3B model as the drafter.

Paper: *SpecAR: Distributed Target Blocks for Fast Autoregressive Video Generation*, Hudeifa Hassan, Statite.

## Install

Tested on one H100 80GB, CUDA 12.8, Python 3.11.

```bash
git clone https://github.com/statite-ai/SpecAR && cd SpecAR
pip install -e .
```

FlashAttention 3 is required on Hopper GPUs. Build it from the
[flash-attention](https://github.com/Dao-AILab/flash-attention) repository (`cd hopper && python setup.py install`);
the build takes 20 to 40 minutes.

## Download weights

```bash
specar download --weights weights
```

This fetches the Krea 14B checkpoint, the Self-Forcing 1.3B checkpoint, and the Wan 2.1 VAE, text encoder and 14B
config from Hugging Face (about 60 GB).

## Generate

```bash
specar generate --prompt "A papercraft coral reef full of colorful fish" --k 7 --out reef.mp4
```

`--k` is the number of blocks the target generates. The default video is 36 blocks (426 frames, 26.6 s at 16 fps);
the target generates blocks `floor(j * 36 / k + 1/2)` for `j = 0..k-1` and the drafter generates the rest.

From Python:

```python
from specar.model_krea import Models, SpecARSession
from specar.schedule import uniform_k

models = Models("weights")
session = SpecARSession(models, "A papercraft coral reef", seed=42, num_blocks=36, anchors=uniform_k(7, 36))
latents = session.generate()
```

## Check your setup

```bash
specar-benchmark --weights weights --k 2 7 9
```

This generates four videos at each k, prints the generation time of each, and compares the latents bit for bit with
the reference latents behind the paper's Krea Realtime results. A mismatch usually means a different attention
kernel (FlashAttention 2 instead of 3) or different library versions.

## License

SpecAR's own code (`specar/schedule.py`, `specar/cli.py`, `specar/benchmark.py`) is Apache-2.0 (`LICENSE`).
The Krea Realtime backend (`specar/model_krea.py`) is adapted from Krea's code, and `specar/_vendor/krea/` is copied
from it; both are CC BY-NC-SA 4.0 (`specar/_vendor/krea/LICENSE.md`), which allows non-commercial use only. See
`specar/_vendor/krea/NOTICE.md` for the source commit and the one change made. Model weights carry their own
licenses on Hugging Face.
