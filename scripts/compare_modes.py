"""Compare decoding modes / protocols side by side for one checkpoint set.

Reports per (mode, protocol, decoding) the generation metrics vs the corpus
reference, on the held-out songs.  This is the main evidence table for the
"does motif conditioning help?" question.

Usage:
    python scripts/compare_modes.py --checkpoints runs/probe_motif/model.pt --split val
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--cache", default="data/cache/dataset.json")
    ap.add_argument("--modes", default="free,motif,plan")
    ap.add_argument("--protocols", default="continuation,free")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--w-chord", type=float, nargs="+", default=[0.0, 3.0])
    ap.add_argument("--w-template", type=float, nargs="+", default=[0.0])
    ap.add_argument("--variation", type=float, nargs="+", default=[0.0])
    ap.add_argument("--w-repeat", type=float, nargs="+", default=[0.0])
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--prime", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import numpy as np
    import torch

    from trance_lora import data as D
    from trance_lora import metrics as M
    from trance_lora.evaluate import Ensemble, _prime, _ref_abs, _ref_tokens
    from trance_lora.model import load_checkpoint
    from trance_lora.sample import MelodySampler, apply_variation, chord_tone_rate, condition_from_song

    torch.set_num_threads(args.threads)
    songs, _, _ = D.load_cache(os.path.join(PROJECT, args.cache))
    by_id = {s.song_id: s for s in songs}
    split = D.split_songs([s.song_id for s in songs], seed=42)
    ids = split[args.split]
    windows = [(sid, w) for sid in ids for w in D.build_windows(by_id[sid])]

    models = [load_checkpoint(p)[0] for p in args.checkpoints]
    sampler = MelodySampler(Ensemble(models) if len(models) > 1 else models[0])

    corpus = []
    for sid, win in windows:
        song = by_id[sid]
        cond = condition_from_song(song, win)
        corpus.append(M.evaluate_tokens(cond, _ref_tokens(song, win), _ref_abs(song, win)))
    cagg = M.aggregate(corpus)
    print(f"split={args.split}: {len(ids)} songs / {len(windows)} windows; "
          f"{len(models)} model(s)")
    print(f"corpus   : in_key {cagg['in_key']:.3f} chord {cagg['chord_tone']:.3f} "
          f"motif {cagg['motif_consistency']:.3f} dist_l1 {cagg.get('interval_dist_l1', float('nan')):.3f} "
          f"dup {cagg['dup_rate']:.3f} distinct {cagg['distinct_pitches']:.1f}")

    hdr = (f"{'mode/protocol/w':30s} {'in_key':>7s} {'chord':>7s} {'motif':>7s} "
           f"{'ival_l1':>8s} {'dist_l1':>8s} {'contour':>8s} {'dup':>6s} {'distinct':>8s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    results = {}
    for w_chord in args.w_chord:
        for w_template in args.w_template:
            for variation in args.variation:
                for w_repeat in args.w_repeat:
                    for protocol in args.protocols.split(","):
                        for mode in args.modes.split(","):
                            rows = []
                            for sid, win in windows:
                                song = by_id[sid]
                                cond = condition_from_song(song, win)
                                for seed in range(args.seeds):
                                    prime = _prime(cond, _ref_tokens(song, win), args.prime) \
                                        if protocol == "continuation" else []
                                    toks = sampler.generate(cond, mode=mode, seed=seed, prime=prime,
                                                            w_chord=w_chord, w_template=w_template,
                                                            w_repeat=w_repeat, greedy=args.greedy)
                                    if variation > 0:
                                        toks = apply_variation(sampler, cond, toks, variation, seed)
                                    rows.append(M.evaluate_tokens(
                                        cond, toks, _ref_abs(song, win) if protocol == "continuation" else None))
                            agg = M.aggregate(rows)
                            key = f"{mode}/w{w_chord:g}t{w_template:g}v{variation:g}r{w_repeat:g}"
                            results[key] = agg
                            print(f"{key:38s} {agg.get('in_key', 0):7.3f} {agg.get('chord_tone', 0):7.3f} "
                                  f"{agg.get('motif_consistency', 0):7.3f} {agg.get('interval_l1', 0):8.3f} "
                                  f"{agg.get('interval_dist_l1', float('nan')):8.3f} "
                                  f"{agg.get('contour_r', 0):8.3f} {agg.get('dup_rate', 0):6.3f} "
                                  f"{agg.get('distinct_pitches', 0):8.1f}")
    if args.out:
        with open(os.path.join(PROJECT, args.out), "w", encoding="utf-8") as f:
            json.dump({"corpus": cagg, "results": results}, f, ensure_ascii=False, indent=1)
        print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
