# SpecAR

SpecAR speeds up block-autoregressive video generation by letting a small drafter generate most blocks and the
large target generate k blocks spread evenly over the video. This repository runs SpecAR on two backbones:

| Backbone | Target | Drafter |
|---|---|---|
| `krea` | [Krea Realtime 14B](https://huggingface.co/krea/krea-realtime-video) | [Self-Forcing](https://github.com/guandeh17/Self-Forcing) 1.3B |
| `helios` | [Helios-Base 14B](https://huggingface.co/BestWishYsh/Helios-Base) | [Helios-Distilled 14B](https://huggingface.co/BestWishYsh/Helios-Distilled) |

Paper: *SpecAR: Distributed Target Blocks for Fast Autoregressive Video Generation*, Hudeifa Hassan, Statite.

## Install

Tested on one H100 80GB, CUDA 12.8, Python 3.11. Each backbone pins the library versions its reference outputs were
produced with, so use one environment per backbone.

```bash
git clone https://github.com/statite-ai/SpecAR && cd SpecAR
pip install -e ".[krea]"      # or: pip install -e ".[helios]"
```

Krea Realtime needs FlashAttention 3 on Hopper GPUs: build it from the
[flash-attention](https://github.com/Dao-AILab/flash-attention) repository (`cd hopper && python setup.py install`;
20 to 40 minutes). Helios loads FlashAttention 3 from the Hugging Face kernel hub through `kernels`.

## Download weights

```bash
specar download --backbone krea --weights weights        # about 50 GB
specar download --backbone helios --weights weights-helios  # about 150 GB
```

## Generate

```bash
specar generate --backbone krea --prompt "A papercraft coral reef full of colorful fish" --k 7 --out reef.mp4
specar generate --backbone helios --weights weights-helios --prompt "A papercraft coral reef full of colorful fish" --k 7 --out reef.mp4
```

`--k` is the number of blocks the target generates, and `--blocks` the video length (default 36). The target generates
blocks `floor(j * blocks / k + 1/2)` for `j = 0..k-1`; the drafter generates the rest. A Krea block is 3 latent
frames (36 blocks = 426 frames, 26.6 s at 16 fps); a Helios block is a 33-frame chunk (36 chunks = 1185 frames, 74 s).

From Python:

```python
from specar.model_krea import Models, SpecARSession
from specar.schedule import uniform_k

models = Models("weights")
session = SpecARSession(models, "A papercraft coral reef", seed=42, num_blocks=36, anchors=uniform_k(7, 36))
latents = session.generate()
```

```python
from specar.model_helios import HeliosSpecAR
from specar.schedule import uniform_k

model = HeliosSpecAR("weights-helios", prompts=["A papercraft coral reef"])
latents, frames, chunk_seconds = model.generate("A papercraft coral reef", uniform_k(7, 36), num_chunks=36, seed=42)
```

## Check your setup

```bash
specar-benchmark --backbone krea --weights weights                     # k = 2, 7, 9; four prompts
specar-benchmark --backbone helios --weights weights-helios            # k = 7; two prompts, about 11 min each
```

The benchmark prints the generation time of each video and compares its latents bit for bit with the reference
latents behind the paper's results. A mismatch usually means a different attention kernel or different library
versions.

## License

SpecAR's own code (`specar/schedule.py`, `specar/cli.py`, `specar/benchmark.py`, `specar/model_helios.py`) is
Apache-2.0 (`LICENSE`). `specar/_vendor/helios/` is copied unchanged from
[Helios](https://github.com/PKU-YuanGroup/Helios) (Apache-2.0, `specar/_vendor/helios/LICENSE.txt`).

The Krea Realtime backend (`specar/model_krea.py`) is adapted from Krea's code, and `specar/_vendor/krea/` is copied
from it; both are CC BY-NC-SA 4.0 (`specar/_vendor/krea/LICENSE.md`), which allows non-commercial use only. See
`specar/_vendor/krea/NOTICE.md` for the source commit and the one change made.

Model weights carry their own licenses on Hugging Face.
