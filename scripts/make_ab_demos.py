"""Generate a demo set with an explicit decoding configuration (for A/B listening).

Used by Step A: renders the same 10 demo specs with the interval-target model and
with the existing absolute-pitch model, under the *same* relaxed decoding settings,
so the two representation choices can be compared by ear and by the self-check.

Usage:
    python scripts/make_ab_demos.py --checkpoints runs/stepA_final_motif_s0/model.pt ... \
        --out-dir out/demo_stepA --w-leap 0.0 --w-bass 0.8 --w-range 0.2 --w-chord 1.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))
sys.path.insert(0, HERE)

from make_demos import DEMOS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--bpm", type=int, default=140)
    ap.add_argument("--w-chord", dest="w_chord", type=float, default=None)
    ap.add_argument("--w-template", dest="w_template", type=float, default=2.0)
    ap.add_argument("--w-repeat", dest="w_repeat", type=float, default=1.2)
    ap.add_argument("--w-leap", dest="w_leap", type=float, default=0.0)
    ap.add_argument("--w-range", dest="w_range", type=float, default=0.2)
    ap.add_argument("--w-bass", dest="w_bass", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import torch

    from trance_lora.evaluate import Ensemble
    from trance_lora.generate import GenerateSpec, describe, generate, render_midi
    from trance_lora.model import load_checkpoint
    from trance_lora.sample import MelodySampler

    torch.set_num_threads(args.threads)
    models = [load_checkpoint(p)[0] for p in args.checkpoints]
    sampler = MelodySampler(Ensemble(models) if len(models) > 1 else models[0])
    out_dir = os.path.join(PROJECT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    decode = dict(w_chord=args.w_chord, w_template=args.w_template, w_repeat=args.w_repeat,
                  w_leap=args.w_leap, w_range=args.w_range, w_bass=args.w_bass)
    manifest = {"checkpoints": args.checkpoints, "target_mode": sampler.target_mode,
                "decode": decode, "items": []}

    for name, key, prog, rhythm, mode, variation in DEMOS:
        best = None
        for k in range(max(1, args.candidates)):
            spec = GenerateSpec(key=key, bpm=args.bpm, progression=prog, rhythm=rhythm,
                                mode=mode, variation=variation, seed=1000 + k, **decode)
            result = generate(sampler, spec)
            st = result.stats
            if mode == "free":
                score = -abs(st.get("dup_rate", 0.0) - 0.30) + 0.04 * st.get("distinct_pitches", 0)
            else:
                motif = st.get("motif_consistency", 0.0)
                score = (-1.0 if motif != motif else -abs(motif - 0.55)
                         + 0.04 * st.get("distinct_pitches", 0))
            if best is None or score > best[0]:
                best = (score, spec, result, k)
        _, spec, result, k = best
        path = os.path.join(out_dir, f"{name}.mid")
        render_midi(result, path, spec)
        manifest["items"].append({
            "file": os.path.relpath(path, PROJECT).replace("\\", "/"),
            "name": name, "key": key, "progression": prog, "rhythm": rhythm,
            "mode": mode, "seed": spec.seed, "selected_candidate": k,
            "bpm": args.bpm, "anchor": result.condition.anchor,
            "stats": result.stats,
        })
        print(f"{name:22s} {describe(spec, result).splitlines()[-4]}")

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n{len(manifest['items'])} MIDI files -> {out_dir}  (target_mode={sampler.target_mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
