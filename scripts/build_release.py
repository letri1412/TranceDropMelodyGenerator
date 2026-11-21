"""Build a reproducible demo release for a registered version (v1 / v2).

    python scripts/build_release.py --version v2 --out out/demo
    python scripts/build_release.py --version v1 --out out/demo_v1

The version determines checkpoints, mixing weight and decoding parameters, so the
same command always regenerates the same 10 demos (byte-identical) — this is what
makes rollback safe.
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
    ap.add_argument("--version", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--bpm", type=int, default=140)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--check", action="store_true", help="run the output self-check afterwards")
    args = ap.parse_args()

    import torch

    from trance_lora import versions
    from trance_lora.evaluate import Ensemble
    from trance_lora.generate import GenerateSpec, describe, generate, render_midi
    from trance_lora.hybrid import HybridLogitModel
    from trance_lora.model import load_checkpoint
    from trance_lora.sample import MelodySampler

    ver = versions.get(args.version or versions.DEFAULT_VERSION)
    paths = ver.paths()
    decode = ver.decode()
    torch.set_num_threads(args.threads)
    if ver.mix is None:
        models = [load_checkpoint(p)[0] for p in paths["checkpoint"]]
        sampler = MelodySampler(Ensemble(models) if len(models) > 1 else models[0])
    else:
        pitch = Ensemble([load_checkpoint(p)[0] for p in paths["checkpoint"]])
        delta = Ensemble([load_checkpoint(p)[0] for p in paths["delta_checkpoint"]])
        sampler = MelodySampler(HybridLogitModel(pitch, delta, mix=ver.mix))

    out_dir = os.path.join(PROJECT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"version": ver.name, "description": ver.description,
                "mix": ver.mix, "decode": decode,
                "checkpoints": paths["checkpoint"],
                "delta_checkpoints": paths["delta_checkpoint"], "items": []}
    for name, key, prog, rhythm, mode, variation in DEMOS:
        best = None
        for k in range(max(1, args.candidates)):
            spec = GenerateSpec(key=key, bpm=args.bpm, progression=prog, rhythm=rhythm,
                                mode=mode, variation=variation,
                                seed=1000 + k, **{kk: vv for kk, vv in decode.items()
                                                  if kk != "mode"})
            result = generate(sampler, spec)
            st = result.stats
            # identical candidate-selection rule to scripts/make_demos.py, so a
            # release rebuild is byte-for-byte reproducible
            if mode == "free":
                score = -abs(st.get("dup_rate", 0.0) - 0.30) + 0.04 * st.get("distinct_pitches", 0)
            else:
                motif = st.get("motif_consistency", 0.0)
                if motif != motif:  # nan: bar-0 motif with zero variance, reject
                    score = -1.0
                else:
                    score = (-abs(motif - 0.55)
                             + 0.04 * st.get("distinct_pitches", 0)
                             - 0.3 * max(0.0, 0.22 - st.get("dup_rate", 0.0)))
            if best is None or score > best[0]:
                best = (score, spec, result, k)
        _, spec, result, k = best
        path = os.path.join(out_dir, f"{name}.mid")
        render_midi(result, path, spec)
        manifest["items"].append({
            "file": os.path.relpath(path, PROJECT).replace("\\", "/"),
            "name": name, "key": key, "progression": prog, "rhythm": rhythm,
            "seed": spec.seed, "selected_candidate": k, "bpm": args.bpm,
            "anchor": result.condition.anchor, "stats": result.stats,
        })
        print(f"[{ver.name}] {name:22s} {describe(spec, result).splitlines()[-4]}")
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n{len(manifest['items'])} MIDI -> {out_dir}  (version={ver.name})")
    if args.check:
        from check_outputs import main as check_main
        sys.argv = ["check_outputs.py", "--manifest",
                    os.path.join(out_dir, "manifest.json"), "--out",
                    os.path.join(PROJECT, "reports", f"output_check_{ver.name}.json")]
        check_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
