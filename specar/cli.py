# SPDX-License-Identifier: Apache-2.0
"""specar download | generate  (--backbone krea | helios)."""
import argparse
import os
import sys
import time

H_DEFAULT = 36  # Krea: 426 frames (26.6 s at 16 fps); Helios: 1185 frames (74 s)


def download(weights, backbone="krea"):
    from huggingface_hub import hf_hub_download, snapshot_download
    os.makedirs(weights, exist_ok=True)
    if backbone == "helios":
        for name, skip in (("Helios-Base", "transformer_init/*"), ("Helios-Distilled", "transformer_ode/*")):
            snapshot_download(f"BestWishYsh/{name}", local_dir=os.path.join(weights, name), ignore_patterns=[skip])
        print(f"weights ready in {weights}")
        return
    hf_hub_download("krea/krea-realtime-video", "krea-realtime-video-14b.safetensors", local_dir=weights)
    p = hf_hub_download("gdhe17/Self-Forcing", "checkpoints/self_forcing_dmd.pt", local_dir=weights)
    os.replace(p, os.path.join(weights, "self_forcing_dmd.pt"))
    snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir=os.path.join(weights, "Wan2.1-T2V-1.3B"))
    hf_hub_download("Wan-AI/Wan2.1-T2V-14B", "config.json", local_dir=os.path.join(weights, "Wan2.1-T2V-14B"))
    print(f"weights ready in {weights}")


def write_mp4(frames, path, fps=16):
    import imageio
    import torch
    vid = (torch.cat(frames, dim=1)[0].permute(0, 2, 3, 1) * 255).to(torch.uint8).numpy()
    imageio.mimwrite(path, list(vid), fps=fps, codec="libx264", quality=7)
    return vid.shape[0]


def run(models, prompt, k, blocks, seed, embeds=None):
    """Generate one video; returns (session, frames, generation seconds)."""
    from .model_krea import SpecARSession
    from .schedule import uniform_k
    frames = []

    def on_frames(pixels, event):
        event.synchronize()
        frames.append(pixels.cpu().add_(1.0).mul_(0.5).clamp_(0.0, 1.0))

    sess = SpecARSession(models, prompt, seed, blocks, uniform_k(k, blocks), prompt_embeds=embeds, frame_callback=on_frames)
    t = time.perf_counter()
    sess.generate()
    return sess, frames, time.perf_counter() - t


def run_helios(model, prompt, k, chunks, seed):
    """Generate one Helios video; returns (latents, uint8 frames, generation seconds)."""
    from .schedule import uniform_k
    t = time.perf_counter()
    latents, frames, _ = model.generate(prompt, uniform_k(k, chunks), chunks, seed)
    return latents, frames, time.perf_counter() - t


def generate_helios(a):
    import imageio.v2 as imageio
    import torch
    from .model_helios import HeliosSpecAR
    from .schedule import max_draft_run, uniform_k
    anchors = uniform_k(a.k, a.blocks)
    print(f"target chunks (k={a.k}): {anchors}; longest drafter run {max_draft_run(anchors, a.blocks)}")
    model = HeliosSpecAR(a.weights, [a.prompt])
    latents, frames, secs = run_helios(model, a.prompt, a.k, a.blocks, a.seed)
    imageio.mimwrite(a.out, list(frames), fps=16, quality=7)
    print(f"{a.out}: {len(frames)} frames, generated in {secs:.1f} s")
    if a.save_latents:
        torch.save(latents.cpu(), os.path.splitext(a.out)[0] + ".latents.pt")


def generate(a):
    if a.backbone == "helios":
        return generate_helios(a)
    import torch
    from .model_krea import Models
    from .schedule import max_draft_run, uniform_k
    torch.set_grad_enabled(False)
    anchors = uniform_k(a.k, a.blocks)
    print(f"target blocks (k={a.k}): {anchors}; longest drafter run {max_draft_run(anchors, a.blocks)}")
    models = Models(a.weights)
    sess, frames, secs = run(models, a.prompt, a.k, a.blocks, a.seed)
    n = write_mp4(frames, a.out)
    print(f"{a.out}: {n} frames, generated in {secs:.1f} s")
    if a.save_latents:
        torch.save(sess.all_latents.cpu(), os.path.splitext(a.out)[0] + ".latents.pt")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="specar")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch the backbone's weights from Hugging Face")
    d.add_argument("--backbone", choices=["krea", "helios"], default="krea")
    d.add_argument("--weights", default="weights")
    g = sub.add_parser("generate", help="text to video with SpecAR")
    g.add_argument("--backbone", choices=["krea", "helios"], default="krea")
    g.add_argument("--prompt", required=True)
    g.add_argument("--k", type=int, default=7, help="number of target blocks (target budget)")
    g.add_argument("--blocks", type=int, default=H_DEFAULT,
                   help="video length in blocks (Krea: 3 latent frames each; Helios: 33-frame chunks)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--weights", default="weights")
    g.add_argument("--out", default="specar.mp4")
    g.add_argument("--save-latents", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "download":
        download(a.weights, a.backbone)
    else:
        generate(a)


if __name__ == "__main__":
    sys.exit(main())
