# SPDX-License-Identifier: Apache-2.0
"""specar-benchmark: generation time per video, and a bitwise check against the reference latents.

The reference SHA-256 hashes are of the committed latents (bf16, [1, 108, 16, 60, 104]) for four prompts at seed 42,
H=36, on one H100 80GB with FlashAttention 3. They equal the latents behind the paper's Krea Realtime results.
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


def latent_sha256(latents):
    import torch
    t = latents.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="specar-benchmark")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 7, 9])
    ap.add_argument("--json", default=None, help="write per-video results here")
    a = ap.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
