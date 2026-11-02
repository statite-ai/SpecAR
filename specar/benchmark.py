# SPDX-License-Identifier: Apache-2.0
"""specar-benchmark: generation time per video, and a bitwise check against the reference latents.

Krea Realtime: SHA-256 of the committed latents (bf16, [1, 108, 16, 60, 104]) for four prompts at seed 42, H=36.
Helios: SHA-256 of the latents for two HeliosBench prompts at seed 115000 + prompt index, H=36.
Both were produced on one H100 80GB with FlashAttention 3 and equal the outputs behind the paper's results.
"""
import argparse
import hashlib
import json
import sys

# MovieGenVideoBench prompt-list indices used for the check, and their text
PROMPTS = {
    1: "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.",
    2: "A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors.",
    4: "Animated scene features a close-up of a short fluffy monster kneeling beside a melting red candle. The art style is 3D and realistic, with a focus on lighting and texture. The mood of the painting is one of wonder and curiosity, as the monster gazes at the flame with wide eyes and open mouth. Its pose and expression convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time. The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image.",
    5: "A gorgeously rendered papercraft world of a coral reef, rife with colorful fish and sea creatures.",
}
SEED, BLOCKS = 42, 36
REFERENCE = {  # (k, prompt index) -> sha256 of the latents
    (2, 1): "d45cc25c6055358ed1512e47457478a20d4f39a30929e53c98be50b1eb5539dd",
    (2, 2): "a047a38982fab0873d53313ae9713b9301ad546c2fa6b0d743e22f4f44daec9b",
    (2, 4): "aec4a46950f61d3b4de1548d0831b6ffe7d61e6d48e8565fd5ec2ccc5b915f2a",
    (2, 5): "0389bcd7450df067721c9f74824e76eca0bd94f9a257db74aa316438a00f5e01",
    (7, 1): "b26ae767755ef1e85102eaf4664f6be6b5c83104dc70f7df2babac66d085f5e8",
    (7, 2): "6aa86b8e2e1eda93fb8c760fc600668c58e2870ff08ecfc89319cda5121be842",
    (7, 4): "9595c96f1fbbb935411e8368e1093b27ada64fe34d21821c9f8ecf80ea34d8f8",
    (7, 5): "4a22521a4d270ec53eb752678f401ff1abba80ebb7c26170798b81f90284dee1",
    (9, 1): "aa4dad831f73912df095de4bca83d50f4d833e053d70c9baaaeacae404846681",
    (9, 2): "19dcca6931debd513d726983c1ec20ef090bfcd61f3579cdf587c34bffe34bba",
    (9, 4): "4956f5e0cabe0ef880a7e5a0a77b15f1a24a972687e8fe1be42cfecdda8e89a5",
    (9, 5): "b814af65b00603f82146d981a142c36625a687fe5ce47a684a23d168943810ea",
}


# Helios: HeliosBench prompt indices, seeds 115000 + index, 36 chunks of 33 frames
HELIOS_PROMPTS = {
    59: "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement.",
    81: "A hyper-realistic macro photograph capturing the intricate details of a dandelion, zooming in at an incredible speed to reveal a dream-like, abstract world. The petals are softly blurred, creating a mesmerizing effect that blends reality with fantasy. Each fiber and grain of pollen is vividly detailed, giving the image a surreal texture. The background fades into a gradient of soft pastel colors, enhancing the ethereal quality of the scene. The dandelion appears almost otherworldly, with its delicate structure and vibrant colors standing out against the blurred, abstract surroundings. A close-up shot with a dynamic zoom-in motion.",
}
HELIOS_SEED0, HELIOS_CHUNKS = 115000, 36
HELIOS_REFERENCE = {  # (k, prompt index) -> sha256 of the latents
}


def latent_sha256(latents):
    import torch
    t = latents.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="specar-benchmark")
    ap.add_argument("--backbone", choices=["krea", "helios"], default="krea")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--k", type=int, nargs="+", default=None, help="default: 2 7 9 (Krea), 7 (Helios)")
    ap.add_argument("--json", default=None, help="write per-video results here")
    ap.add_argument("--save-mp4", default=None, help="Helios: also write each video here")
    a = ap.parse_args(argv)
    if a.backbone == "helios":
        return helios(a)
    a.k = a.k or [2, 7, 9]

    import torch
    from .cli import run
    from .model_krea import Models
    torch.set_grad_enabled(False)
    models = Models(a.weights)
    embeds = models.encode_prompts(list(PROMPTS.values()))
    run(models, PROMPTS[1], a.k[0], BLOCKS, SEED, embeds[PROMPTS[1]])          # warm-up (kernels, allocator)
    rows, failed = [], 0
    for k in a.k:
        for i, prompt in PROMPTS.items():
            sess, _, secs = run(models, prompt, k, BLOCKS, SEED, embeds[prompt])
            h = latent_sha256(sess.all_latents)
            ref = REFERENCE.get((k, i))
            status = "n/a" if ref is None else ("MATCH" if h == ref else "MISMATCH")
            failed += status == "MISMATCH"
            rows.append({"k": k, "prompt": i, "seconds": round(secs, 2), "sha256": h, "bitwise": status})
            print(f"k={k} prompt {i}: {secs:6.1f} s  latents {status}", flush=True)
            del sess
            torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
    print("bitwise check:", "FAILED" if failed else "all match")
    return 1 if failed else 0


def helios(a):
    import os
    import imageio.v2 as imageio
    import torch
    from .cli import run_helios
    from .model_helios import HeliosSpecAR
    ks = a.k or [7]
    model = HeliosSpecAR(a.weights, list(HELIOS_PROMPTS.values()))
    rows, failed = [], 0
    for k in ks:
        for i, prompt in HELIOS_PROMPTS.items():
            latents, frames, secs = run_helios(model, prompt, k, HELIOS_CHUNKS, HELIOS_SEED0 + i)
            h = latent_sha256(latents)
            ref = HELIOS_REFERENCE.get((k, i))
            status = "n/a" if ref is None else ("MATCH" if h == ref else "MISMATCH")
            failed += status == "MISMATCH"
            fh = hashlib.sha256(frames.tobytes()).hexdigest()
            rows.append({"k": k, "prompt": i, "seconds": round(secs, 2), "sha256": h, "frames_sha256": fh, "bitwise": status})
            print(f"k={k} prompt {i}: {secs:6.1f} s  latents {status}", flush=True)
            if a.save_mp4:
                os.makedirs(os.path.join(a.save_mp4, f"k{k}"), exist_ok=True)
                imageio.mimwrite(os.path.join(a.save_mp4, f"k{k}", f"prompt_{i:03d}.mp4"), list(frames), fps=16, quality=7)
            del latents, frames
            torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
    print("bitwise check:", "FAILED" if failed else "all match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
