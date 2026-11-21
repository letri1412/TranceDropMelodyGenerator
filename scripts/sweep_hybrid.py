"""Sweep the hybrid mixing weight on the held-out test songs.

mix = 0   -> pure pitch model   (more variance / occasionally more "human")
mix = 1   -> pure interval model (more faithful to the corpus, more reliable)
Reports both corpus-likeness and a few "novelty / non-mechanical" proxies.
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
    ap.add_argument("--mixes", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mode", default="plan_root")
    ap.add_argument("--w-chord", dest="w_chord", type=float, default=1.1)
    ap.add_argument("--w-template", dest="w_template", type=float, default=2.0)
    ap.add_argument("--w-repeat", dest="w_repeat", type=float, default=1.2)
    ap.add_argument("--w-leap", dest="w_leap", type=float, default=0.0)
    ap.add_argument("--w-range", dest="w_range", type=float, default=0.2)
    ap.add_argument("--w-bass", dest="w_bass", type=float, default=0.8)
    args = ap.parse_args()

    import numpy as np
    import torch

    from trance_lora import data as D
    from trance_lora import metrics as M
    from trance_lora.evaluate import Ensemble
    from trance_lora.generate import _pedal_notes
    from trance_lora.hybrid import HybridLogitModel
    from trance_lora.model import load_checkpoint
    from trance_lora.sample import MelodySampler, condition_from_song

    torch.set_num_threads(args.threads)
    songs, _, _ = D.load_cache(os.path.join(PROJECT, "data", "cache", "dataset.json"))
    by = {s.song_id: s for s in songs}
    split = D.split_songs([s.song_id for s in songs], seed=42)
    wins = [(sid, w) for sid in split["test"] for w in D.build_windows(by[sid])]

    pitch = Ensemble([load_checkpoint(os.path.join(PROJECT, "runs", f"final_motif_s{i}", "model.pt"))[0]
                      for i in range(3)])
    delta = Ensemble([load_checkpoint(os.path.join(PROJECT, "runs", f"stepA_final_motif_s{i}", "model.pt"))[0]
                      for i in range(3)])
    corpus_rows = []
    for sid, win in wins:
        song = by[sid]
        cond = condition_from_song(song, win)
        idx = win.indices
        mel = {i: song.melody_abs[idx[i]] for i in range(cond.L) if song.layer[idx[i]] == 2}
        ped = {i: song.pedal_abs[idx[i]] for i in range(cond.L)
               if song.layer[idx[i]] == 1 and song.pedal_abs[idx[i]] != D.NONE}
        corpus_rows.append(M.melody_quality_stats(mel, ped, cond.n_bars))
    corpora = M.aggregate(corpus_rows)

    def show(tag: str, q: dict, g: dict) -> None:
        print(f"{tag:14s} motif {g.get('motif_consistency',0):5.3f} chord {g.get('chord_tone',0):5.3f} "
              f"meanΔ {q.get('quality_mean_abs_interval',0):5.2f} "
              f"barRng {q.get('quality_bar_pitch_range',0):5.2f} "
              f"corrB {q.get('quality_corr_melody_bass',0):+5.2f} "
              f"j7 {q.get('quality_jump7_rate',0):5.3f} "
              f"distinct {g.get('distinct_pitches',0):4.1f} "
              f"ival-熵 {q.get('quality_interval_entropy',0):4.2f}")

    print(f"{'config':14s} {'motif':>5s} {'chord':>5s} {'meanΔ':>5s} {'barRng':>5s} "
          f"{'corrB':>5s} {'j7':>5s} {'distinct':>4s} ival-熵")
    show("CORPUS", corpora, {"motif_consistency": corpora.get("motif_consistency", 0),
                             "chord_tone": corpora.get("chord_tone", 0),
                             "distinct_pitches": corpora.get("distinct_pitches", 0)})
    rows = []
    for mix in [float(x) for x in args.mixes.split(",")]:
        if mix >= 0.999:
            sampler = MelodySampler(delta)
            tag = "mix 1.00"
        elif mix <= 0.001:
            sampler = MelodySampler(pitch)
            tag = "mix 0.00"
        else:
            sampler = MelodySampler(HybridLogitModel(pitch, delta, mix=mix))
            tag = f"mix {mix:.2f}"
        qrows, grows = [], []
        for sid, win in wins:
            song = by[sid]
            cond = condition_from_song(song, win)
            for seed in range(args.seeds):
                toks = sampler.generate(
                    cond, mode=args.mode, seed=seed, w_chord=args.w_chord,
                    w_template=args.w_template, w_repeat=args.w_repeat,
                    w_leap=args.w_leap, w_range=args.w_range, w_bass=args.w_bass)
                mel = sampler.melody_abs(cond, toks)
                ped = _pedal_notes(cond, 15, mel)
                qrows.append(M.melody_quality_stats(mel, ped, cond.n_bars))
                grows.append(M.evaluate_tokens(cond, toks, None))
        q, g = M.aggregate(qrows), M.aggregate(grows)
        rows.append({"mix": mix, "quality": q, "gen": {k: v for k, v in g.items()
                                                       if isinstance(v, (int, float))}})
        show(tag, q, g)

    out = os.path.join(PROJECT, "reports", "hybrid_sweep.json")
    json.dump({"corpus": corpora, "rows": rows}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
