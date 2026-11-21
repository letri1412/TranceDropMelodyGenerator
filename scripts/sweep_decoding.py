"""Grid-search decoding weights against the corpus target statistics.

Targets measured on the test split (corpus):
  mean_abs_interval ~3.0, per-bar pitch range ~6.5, corr(melody bar-mean, bass bar-mean) ~+0.5,
  chord-tone ~0.68, in-key 1.0.

Prints one line per config so the best compromise can be picked.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))
sys.path.insert(0, HERE)

from diagnose_quality import melody_stats, show  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mode", default="plan")
    ap.add_argument("--w-template", type=float, nargs="+", default=[2.0])
    ap.add_argument("--w-leap", type=float, nargs="+", default=[0.0, 0.8, 1.5, 2.5])
    ap.add_argument("--w-range", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    ap.add_argument("--w-repeat", type=float, nargs="+", default=[1.2])
    ap.add_argument("--w-chord", type=float, nargs="+", default=[1.5])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch

    from trance_lora import data as D
    from trance_lora import metrics as M
    from trance_lora.evaluate import Ensemble
    from trance_lora.generate import _pedal_notes
    from trance_lora.model import load_checkpoint
    from trance_lora.sample import MelodySampler, condition_from_song

    torch.set_num_threads(args.threads)
    songs, _, _ = D.load_cache(os.path.join(PROJECT, "data", "cache", "dataset.json"))
    by_id = {s.song_id: s for s in songs}
    split = D.split_songs([s.song_id for s in songs], seed=42)
    windows = [(sid, w) for sid in split["test"] for w in D.build_windows(by_id[sid])]
    if args.limit:
        windows = windows[: args.limit]
    models = [load_checkpoint(p)[0] for p in args.checkpoints]
    sampler = MelodySampler(Ensemble(models) if len(models) > 1 else models[0])

    corpus_rows, corpus_gen = [], []
    for sid, win in windows:
        song = by_id[sid]
        cond = condition_from_song(song, win)
        idx = win.indices
        mel = {i: song.melody_abs[idx[i]] for i in range(cond.L) if song.layer[idx[i]] == 2}
        ped = {i: song.pedal_abs[idx[i]] for i in range(cond.L)
               if song.layer[idx[i]] == 1 and song.pedal_abs[idx[i]] != D.NONE}
        corpus_rows.append(melody_stats(mel, ped, cond.n_bars))
        toks = M.evaluate_tokens(cond, __import__("numpy").full(cond.L, D.PAD), None)
        _ = toks
    corpora = M.aggregate(corpus_rows)
    show("CORPUS", corpora)
    print()

    results = {}
    for w_tmpl, w_leap, w_range, w_rep, w_chord in itertools.product(
            args.w_template, args.w_leap, args.w_range, args.w_repeat, args.w_chord):
        stat_rows, gen_rows = [], []
        for sid, win in windows:
            song = by_id[sid]
            cond = condition_from_song(song, win)
            for seed in range(args.seeds):
                toks = sampler.generate(cond, mode=args.mode, seed=seed, w_chord=w_chord,
                                        w_template=w_tmpl, w_repeat=w_rep,
                                        w_leap=w_leap, w_range=w_range)
                mel = sampler.melody_abs(cond, toks)
                ped = _pedal_notes(cond, 15, mel)
                stat_rows.append(melody_stats(mel, ped, cond.n_bars))
                gen_rows.append(M.evaluate_tokens(cond, toks, None))
        s, g = M.aggregate(stat_rows), M.aggregate(gen_rows)
        key = f"t{w_tmpl:g} L{w_leap:g} R{w_range:g} rep{w_rep:g} ch{w_chord:g}"
        score = (-abs(s.get("mean_abs_interval", 0) - 3.0)
                 - 0.35 * abs(s.get("bar_pitch_range_mean", 0) - 6.5)
                 + 1.2 * max(0.0, s.get("corr_melody_bass", 0))
                 + 0.8 * g.get("in_key", 0)
                 - abs(g.get("chord_tone", 0) - 0.80)
                 + 0.3 * min(1.0, g.get("motif_consistency", 0) or 0))
        results[key] = {"stats": s, "gen": g, "score": score}
        print(f"{key:34s} scr {score:+5.2f} ival {s.get('mean_abs_interval', 0):4.2f} "
              f"barRng {s.get('bar_pitch_range_mean', 0):4.2f} corrB {s.get('corr_melody_bass', 0):+4.2f} "
              f"j>7 {s.get('jump7_rate', 0):4.2f} | chord {g.get('chord_tone', 0):4.2f} "
              f"inkey {g.get('in_key', 0):4.2f} motif {g.get('motif_consistency', 0) or 0:4.2f} "
              f"dup {g.get('dup_rate', 0) or 0:4.2f}")

    best = sorted(results.items(), key=lambda kv: -kv[1]["score"])[:5]
    print("\nTOP 5:")
    for k, v in best:
        print(f"  {k:34s} score {v['score']:+5.2f}")
    out = os.path.join(PROJECT, "reports", "decoding_sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"corpus": corpora, "configs": results}, f, ensure_ascii=False, indent=1)
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
