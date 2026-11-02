# SPDX-License-Identifier: Apache-2.0
"""SpecAR target schedule: k anchors including block 0, spread as evenly as possible over H blocks.

a_j = floor(j * H / k + 1/2), j = 0..k-1. The longest draft run is ceil((H - k) / k) (Appendix B of the paper).
"""


def uniform_k(k: int, H: int) -> list[int]:
    if not (1 <= k <= H):
        raise ValueError(f"k={k} outside 1..{H}")
    anchors = sorted({int(j * H / k + 0.5) for j in range(k)})
    if len(anchors) != k or anchors[0] != 0:
        raise ValueError(f"degenerate anchor schedule k={k} H={H}: {anchors}")
    return anchors


def max_draft_run(anchors: list[int], H: int) -> int:
    a = sorted(anchors)
    gaps = [a[i + 1] - a[i] - 1 for i in range(len(a) - 1)] + [H - 1 - a[-1]]
    return max(gaps)
