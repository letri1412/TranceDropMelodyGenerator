"""Batch "lottery" generator: sample many takes and pre-screen them for you.

The listener feedback was: the interval model is reliably corpus-like (mechanical),
while the pitch model needs many rolls but occasionally produces a more human line.
This tool rolls N takes in one call and prints a metric table, so auditioning is
cheap.  Two ranking modes:

  --rank novelty  (default)  reward takes that deviate from the corpus' mechanical
                             average while still being valid (in-key, 50-90% chord tones)
  --rank corpus               rank by closeness to the corpus melodic shape
  --rank none                 keep generation order

Families: pitch (existing model), delta (Step A), mix (logit-space hybrid).

Examples:
    python scripts/lottery.py --family mix --takes 8 --key Amin --progression i,VI,III,VII
    python scripts/lottery.py --family pitch --takes 16 --rank novelty --mode plan_root
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

CORPUS = {"mean_interval": 3.00, "bar_range": 5.30, "entropy": 2.36, "distinct": 8.7,
          "chord_tone": 0.676}


def load_sampler(args):
    import torch

    from trance_lora.evaluate import Ensemble
    from trance_lora.hybrid import HybridLogitModel
    from trance_lora.model import load_checkpoint
    from trance_lora.sample import MelodySampler

    runs = os.path.join(PROJECT, "runs")

    def ck(prefix, n=3):
        return [os.path.join(runs, f"{prefix}_s{i}", "model.pt") for i in range(n)]

    if args.family == "pitch":
        return MelodySampler(Ensemble([load_checkpoint(p)[0] for p in ck("final_motif")]))
    if args.family == "delta":
        return MelodySampler(Ensemble([load_checkpoint(p)[0] for p in ck("stepA_final_motif")]))
    pitch = Ensemble([load_checkpoint(p)[0] for p in ck("final_motif")])
    delta = Ensemble([load_checkpoint(p)[0] for p in ck("stepA_final_motif")])
    return MelodySampler(HybridLogitModel(pitch, delta, mix=args.mix))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mix", choices=["pitch", "delta", "mix"])
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--key", default="Amin")
    ap.add_argument("--bpm", type=int, default=140)
    ap.add_argument("--progression", default="i,VI,III,VII")
    ap.add_argument("--rhythm", default="12112112")
    ap.add_argument("--mode", default="plan_root")
    ap.add_argument("--takes", type=int, default=8)
    ap.add_argument("--out", default="out/lottery")
    ap.add_argument("--rank", default="novelty", choices=["novelty", "corpus", "none"])
    ap.add_argument("--w-chord", dest="w_chord", type=float, default=None)
    ap.add_argument("--w-template", dest="w_template", type=float, default=2.0)
    ap.add_argument("--w-repeat", dest="w_repeat", type=float, default=1.2)
    ap.add_argument("--w-leap", dest="w_leap", type=float, default=0.0)
    ap.add_argument("--w-range", dest="w_range", type=float, default=0.2)
    ap.add_argument("--w-bass", dest="w_bass", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    import torch

    from trance_lora import metrics as M
    from trance_lora.generate import GenerateSpec, _pedal_notes, generate, render_midi
    from trance_lora.sample import MelodySampler

    torch.set_num_threads(args.threads)
    sampler: MelodySampler = load_sampler(args)
    os.makedirs(os.path.join(PROJECT, args.out), exist_ok=True)
    items = []
    for k in range(args.takes):
        spec = GenerateSpec(key=args.key, bpm=args.bpm, progression=args.progression,
                            rhythm=args.rhythm, mode=args.mode, seed=5000 + k,
                            w_chord=args.w_chord, w_template=args.w_template,
                            w_repeat=args.w_repeat, w_leap=args.w_leap,
                            w_range=args.w_range, w_bass=args.w_bass)
        result = generate(sampler, spec)
        cond = result.condition
        mel = result.melody
        ped = _pedal_notes(cond, 15, mel)
        q = M.melody_quality_stats(mel, ped, cond.n_bars)
        g = M.evaluate_tokens(cond, result.tokens)
        valid = (g.get("in_key", 0) >= 0.999 and 0.50 <= g.get("chord_tone", 0) <= 0.90)
        novelty = (0.5 * abs(q.get("quality_mean_abs_interval", 0) - CORPUS["mean_interval"])
                   + 0.3 * (q.get("quality_interval_entropy", 0) - CORPUS["entropy"])
                   + 0.2 * (g.get("distinct_pitches", 0) - CORPUS["distinct"]))
        closeness = (abs(q.get("quality_mean_abs_interval", 0) - CORPUS["mean_interval"])
                     + abs(q.get("quality_bar_pitch_range", 0) - CORPUS["bar_range"]))
        path = os.path.join(PROJECT, args.out, f"take_{k:02d}.mid")
        render_midi(result, path, spec)
        items.append({
            "take": k, "seed": spec.seed, "file": os.path.relpath(path, PROJECT).replace("\\", "/"),
            "valid": bool(valid), "novelty": float(novelty), "closeness": float(closeness),
            "in_key": float(g.get("in_key", 0)), "chord_tone": float(g.get("chord_tone", 0)),
            "motif": float(g.get("motif_consistency", 0) or 0),
            "mean_interval": float(q.get("quality_mean_abs_interval", 0)),
            "bar_range": float(q.get("quality_bar_pitch_range", 0)),
            "corr_bass": float(q.get("quality_corr_melody_bass", 0) or 0),
            "jump7": float(q.get("quality_jump7_rate", 0)),
            "entropy": float(q.get("quality_interval_entropy", 0)),
            "distinct": float(g.get("distinct_pitches", 0)),
        })
    if args.rank == "novelty":
        items.sort(key=lambda r: (not r["valid"], -r["novelty"]))
    elif args.rank == "corpus":
        items.sort(key=lambda r: (not r["valid"], r["closeness"]))
    with open(os.path.join(PROJECT, args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"family": args.family, "mix": args.mix, "mode": args.mode,
                   "key": args.key, "progression": args.progression, "rhythm": args.rhythm,
                   "rank": args.rank, "items": items}, f, ensure_ascii=False, indent=1)
    print(f"family={args.family} mix={args.mix} mode={args.mode} rank={args.rank} "
          f"takes={len(items)}  (语料参考: meanΔ {CORPUS['mean_interval']:.2f}, "
          f"barRng {CORPUS['bar_range']:.2f}, entropy {CORPUS['entropy']:.2f})")
    print(f"{'#':>3s} {'valid':>5s} {'novelty':>8s} {'in_key':>6s} {'chord':>6s} {'motif':>6s} "
          f"{'meanΔ':>6s} {'barRng':>6s} {'corrB':>6s} {'jump7':>6s} {'entropy':>7s} {'dist':>4s}  file")
    for r in items:
        print(f"{r['take']:3d} {str(r['valid']):>5s} {r['novelty']:+8.2f} {r['in_key']:6.3f} "
              f"{r['chord_tone']:6.3f} {r['motif']:6.3f} {r['mean_interval']:6.2f} "
              f"{r['bar_range']:6.2f} {r['corr_bass']:+6.2f} {r['jump7']:6.3f} "
              f"{r['entropy']:7.2f} {r['distinct']:4.1f}  {r['file']}")
    print(f"\nmanifest -> {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
