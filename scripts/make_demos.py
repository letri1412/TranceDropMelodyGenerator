"""Generate a batch of demo MIDI files from the final ensemble.

Writes ``out/demo/*.mid`` plus ``out/demo/manifest.json`` with the per-MIDI metrics
used by the summary report.

Usage:
    python scripts/make_demos.py --checkpoints runs/final_motif_s0/model.pt \
        runs/final_motif_s1/model.pt runs/final_motif_s2/model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

DEMOS = [
    # name, key, progression (4 chords = 2 bars each, or 8 = 1 bar each), rhythm, mode, variation
    ("01_Amin_plan", "Amin", "i,VI,III,VII", "12112112", "plan_root", 0.0),
    ("02_Cmin_plan", "Cmin", "i,i,VI,VI,iv,iv,v,v", "21212212", "plan_root", 0.0),
    ("03_Emin_free", "Emin", "i,VI,iv,VII", "12112121", "free", 0.0),
    ("04_Fmin_plan", "Fmin", "i,i,III,III,iv,iv,VII,VII", "12122121", "plan_root", 0.0),
    ("05_Gmin_motif", "Gmin", "i,III,VII,v", "12112112", "motif", 0.0),
    ("06_Bmin_plan", "Bmin", "i,i,VI,VI,III,III,VII,VII", "21212212", "plan_root", 0.0),
    ("07_Dmin_plan", "Dmin", "i,v,VI,III", "12112112", "plan_root", 0.0),
    ("08_Amin_full8", "Amin", "i,i,v,v,VI,VI,III,III", "12112112", "plan_root", 0.0),
    ("09_Gmin_planroot", "Gmin", "i,VI,III,VII", "12112112", "plan_root", 0.0),
    ("10_Dmin_free2", "Dmin", "i,i,VI,VI,III,III,v,v", "12122121", "free", 0.0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out-dir", default=os.path.join("out", "demo"))
    ap.add_argument("--bpm", type=int, default=140)
    ap.add_argument("--candidates", type=int, default=3,
                    help="sample N takes per spec and keep the best (by motif consistency)")
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
    manifest = {"checkpoints": args.checkpoints, "bpm": args.bpm,
                "selection": ("best of N by |motif-0.55| and pitch diversity (motif/plan modes); "
                              "corpus-like dup rate + diversity (free mode)"),
                "items": []}

    for name, key, prog, rhythm, mode, variation in DEMOS:
        best = None
        for k in range(max(1, args.candidates)):
            spec = GenerateSpec(key=key, bpm=args.bpm, progression=prog, rhythm=rhythm,
                                mode=mode, variation=variation, seed=1000 + k)
            result = generate(sampler, spec)
            st = result.stats
            if mode == "free":
                # free generation has no motif plan: prefer corpus-like repetition + diversity
                score = -abs(st.get("dup_rate", 0.0) - 0.30) + 0.04 * st.get("distinct_pitches", 0)
            else:
                motif = st.get("motif_consistency", 0.0)
                if motif != motif:  # nan: a bar-0 motif with zero variance, reject
                    score = -1.0
                else:
                    # target the corpus motif level (~0.55) instead of the most literal take,
                    # and reward pitch diversity so the lead does not turn into a 2-note loop
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
            "mode": mode, "variation": variation, "seed": spec.seed,
            "candidates": args.candidates, "selected_candidate": k,
            "bpm": args.bpm, "anchor": result.condition.anchor,
            "stats": result.stats,
        })
        print(f"{name:22s} {describe(spec, result).splitlines()[-3]}")

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n{len(manifest['items'])} MIDI files -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
