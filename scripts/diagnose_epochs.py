"""Diagnose how generation quality evolves with training epochs.

For every ``ckpt_e*.pt`` in a run directory, measure teacher-forced val loss and
constrained continuation metrics on the validation songs, so early stopping can
be aligned with generation quality instead of raw cross-entropy.

Usage:
    python scripts/diagnose_epochs.py --run runs/probe_base --split val
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--cache", default="data/cache/dataset.json")
    ap.add_argument("--modes", default="free")
    ap.add_argument("--protocol", default="continuation")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    import torch

    from trance_lora import data as D
    from trance_lora import metrics as M
    from trance_lora.evaluate import _prime, _ref_abs, _ref_tokens
    from trance_lora.model import load_checkpoint
    from trance_lora.sample import MelodySampler, condition_from_song
    from trance_lora.train import TrainConfig, evaluate_loss, prepare

    torch.set_num_threads(args.threads)
    songs, _, _ = D.load_cache(os.path.join(PROJECT, args.cache))
    by_id = {s.song_id: s for s in songs}
    split = D.split_songs([s.song_id for s in songs], seed=42)
    ids = split[args.split]
    windows = [(sid, w) for sid in ids for w in D.build_windows(by_id[sid])]

    cfg = TrainConfig(cache=os.path.join(PROJECT, args.cache))
    data = prepare(cfg)
    arrs = data[args.split]

    ckpts = sorted(glob.glob(os.path.join(PROJECT, args.run, "ckpt_e*.pt")))
    final = os.path.join(PROJECT, args.run, "model.pt")
    if os.path.exists(final):
        ckpts.append(final)

    # corpus reference
    corpus = []
    for sid, win in windows:
        song = by_id[sid]
        cond = condition_from_song(song, win)
        corpus.append(M.evaluate_tokens(cond, _ref_tokens(song, win), _ref_abs(song, win)))
    cagg = M.aggregate(corpus)
    print(f"{args.split}: {len(ids)} songs / {len(windows)} windows")
    print(f"corpus: in_key {cagg['in_key']:.3f} chord {cagg['chord_tone']:.3f} "
          f"motif {cagg['motif_consistency']:.3f} distinct {cagg['distinct_pitches']:.1f} "
          f"span {cagg['pitch_span']:.1f}")
    print(f"\n{'ckpt':16s} {'val_loss':>8s} {'in_key':>7s} {'chord':>7s} {'motif':>7s} "
          f"{'ival_l1':>8s} {'contour':>8s} {'dup':>6s} {'barRng':>7s} {'corrBass':>8s} {'j7':>5s}")
    rows = []
    for path in ckpts:
        model, _ = load_checkpoint(path)
        sampler = MelodySampler(model)
        val_loss, _ = evaluate_loss(model, arrs)
        mrows, qrows = [], []
        for sid, win in windows:
            song = by_id[sid]
            cond = condition_from_song(song, win)
            for mode in args.modes.split(","):
                prime = _prime(cond, _ref_tokens(song, win), 3) if args.protocol == "continuation" else []
                toks = sampler.generate(cond, mode=mode, seed=0, prime=prime,
                                        w_chord=1.5, w_template=2.0, w_repeat=1.2,
                                        w_leap=1.0, w_range=0.3)
                mrows.append(M.evaluate_tokens(cond, toks,
                                               _ref_abs(song, win) if args.protocol == "continuation" else None))
                from trance_lora.generate import _pedal_notes
                mel = sampler.melody_abs(cond, toks)
                qrows.append(M.melody_quality_stats(mel, _pedal_notes(cond, 15, mel), cond.n_bars))
        agg = M.aggregate(mrows)
        qagg = M.aggregate(qrows)
        name = os.path.basename(path).replace("ckpt_", "").replace(".pt", "")
        print(f"{name:16s} {val_loss:8.4f} {agg.get('in_key', float('nan')):7.3f} "
              f"{agg.get('chord_tone', float('nan')):7.3f} {agg.get('motif_consistency', float('nan')):7.3f} "
              f"{agg.get('interval_l1', float('nan')):8.3f} {agg.get('contour_r', float('nan')):8.3f} "
              f"{agg.get('dup_rate', float('nan')):6.3f} "
              f"{qagg.get('quality_bar_pitch_range', float('nan')):7.2f} "
              f"{qagg.get('quality_corr_melody_bass', float('nan')):+8.2f} "
              f"{qagg.get('quality_jump7_rate', float('nan')):5.3f}")
        rows.append({"ckpt": name, "val_loss": val_loss, **agg, **qagg})
    out = os.path.join(PROJECT, args.run, f"diagnose_{args.split}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
