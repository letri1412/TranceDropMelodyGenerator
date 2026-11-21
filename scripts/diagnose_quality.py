"""Diagnose the two reported musical problems quantitatively.

1. 大跳: per-bar max leap / fraction of leaps > 7 and > 10 semitones.
2. 起伏/跟随低音: within-bar pitch range, cross-bar movement, and the correlation
   between the bar-mean melody register and the bar-mean pedal (bass) register.

Compares the corpus against generated melodies under several decoding settings,
and reports teacher-forced PAD-prediction rates (a model that predicts PAD at
melody slots cannot produce intervals at all).

Usage:
    python scripts/diagnose_quality.py --checkpoints runs/final_motif_s0/model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

SPB = 16  # slots per bar


def melody_stats(bar_pitches: dict[int, int], bar_pedal: dict[int, int], n_bars: int) -> dict:
    """Per-bar interval / contour / bass-following statistics."""
    import numpy as np

    bar_pitch, bar_bass = [], []
    all_ints: list[float] = []
    ranges: list[float] = []
    leaps: list[float] = []
    jumps7 = jumps10 = 0
    for b in range(n_bars):
        seq = [bar_pitches[i] for i in range(b * SPB, (b + 1) * SPB) if i in bar_pitches]
        if len(seq) >= 2:
            ints = np.abs(np.diff(seq))
            all_ints.extend(ints.tolist())
            jumps7 += int((ints > 7).sum())
            jumps10 += int((ints > 10).sum())
            leaps.append(float(ints.max()))
            ranges.append(float(max(seq) - min(seq)))
        if seq:
            bar_pitch.append(float(np.mean(seq)))
        ped = [bar_pedal[i] for i in range(b * SPB, (b + 1) * SPB) if i in bar_pedal]
        bar_bass.append(float(np.mean(ped)) if ped else float("nan"))

    out = {
        "n_notes": len(bar_pitches),
        "mean_abs_interval": float(np.mean(all_ints)) if all_ints else float("nan"),
        "p90_interval": float(np.percentile(all_ints, 90)) if all_ints else float("nan"),
        "max_interval": float(np.max(all_ints)) if all_ints else float("nan"),
        "jump7_rate": jumps7 / max(1, len(all_ints)),
        "jump10_rate": jumps10 / max(1, len(all_ints)),
        "bar_pitch_range_mean": float(np.mean(ranges)) if ranges else float("nan"),
        "bar_max_leap_mean": float(np.mean(leaps)) if leaps else float("nan"),
    }
    bp = np.asarray(bar_pitch, dtype=float)
    bped = np.asarray(bar_bass, dtype=float)
    ok = ~np.isnan(bped)
    if len(bp) >= 3:
        out["bar_pitch_std"] = float(bp.std())
        out["bar_pitch_range"] = float(bp.max() - bp.min())
        if ok.sum() >= 3 and bp[ok].std() > 1e-9 and bped[ok].std() > 1e-9:
            out["corr_melody_bass"] = float(np.corrcoef(bp[ok], bped[ok])[0, 1])
    return out


def show(title: str, agg: dict) -> None:
    print(f"{title:32s} ival {agg.get('mean_abs_interval', float('nan')):5.2f} "
          f"p90 {agg.get('p90_interval', float('nan')):5.2f} "
          f"max {agg.get('max_interval', float('nan')):5.1f} "
          f"j>7 {agg.get('jump7_rate', float('nan')):5.3f} "
          f"j>10 {agg.get('jump10_rate', float('nan')):5.3f} "
          f"barRng {agg.get('bar_pitch_range_mean', float('nan')):5.2f} "
          f"barLeap {agg.get('bar_max_leap_mean', float('nan')):5.2f} "
          f"barStd {agg.get('bar_pitch_std', float('nan')):5.2f} "
          f"corrBass {agg.get('corr_melody_bass', float('nan')):+5.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--cache", default="data/cache/dataset.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    import torch

    from trance_lora import data as D
    from trance_lora import metrics as M
    from trance_lora.evaluate import Ensemble
    from trance_lora.generate import _pedal_notes
    from trance_lora.model import load_checkpoint
    from trance_lora.sample import MelodySampler, condition_from_song

    torch.set_num_threads(args.threads)
    songs, _, _ = D.load_cache(os.path.join(PROJECT, args.cache))
    by_id = {s.song_id: s for s in songs}
    split = D.split_songs([s.song_id for s in songs], seed=42)
    ids = split[args.split]
    windows = [(sid, w) for sid in ids for w in D.build_windows(by_id[sid])]
    if args.limit:
        windows = windows[: args.limit]

    # ---------------- corpus ----------------
    corpus_rows = []
    for sid, win in windows:
        song = by_id[sid]
        cond = condition_from_song(song, win)
        idx = win.indices
        mel = {i: song.melody_abs[idx[i]] for i in range(cond.L) if song.layer[idx[i]] == 2}
        ped = {i: song.pedal_abs[idx[i]] for i in range(cond.L)
               if song.layer[idx[i]] == 1 and song.pedal_abs[idx[i]] != D.NONE}
        corpus_rows.append(melody_stats(mel, ped, cond.n_bars))
    corpora = M.aggregate(corpus_rows)

    print(f"split={args.split}: {len(ids)} songs / {len(windows)} windows")
    show("CORPUS", corpora)
    print()

    models = [load_checkpoint(p)[0] for p in args.checkpoints]
    model = Ensemble(models) if len(models) > 1 else models[0]
    sampler = MelodySampler(model)

    configs = [
        ("greedy/free/no-bias", dict(mode="free", greedy=True, w_chord=0.0, w_template=0.0, w_repeat=0.0)),
        ("sample/free/no-bias", dict(mode="free", greedy=False, w_chord=0.0, w_template=0.0, w_repeat=0.0)),
        ("sample/plan/old-decode", dict(mode="plan", greedy=False, w_chord=1.5, w_template=2.0,
                                        w_repeat=1.2, w_leap=0.0, w_range=0.0)),
        ("sample/plan/new-decode", dict(mode="plan", greedy=False, w_chord=1.5, w_template=2.0,
                                        w_repeat=1.2, w_leap=1.0, w_range=0.3)),
        ("greedy/plan/new-decode", dict(mode="plan", greedy=True, w_chord=1.5, w_template=2.0,
                                        w_repeat=1.2, w_leap=1.0, w_range=0.3)),
        ("sample/free/new-decode", dict(mode="free", greedy=False, w_chord=1.5, w_template=0.0,
                                        w_repeat=1.2, w_leap=1.0, w_range=0.3)),
    ]
    results = {}
    for name, kw in configs:
        rows = []
        for sid, win in windows:
            song = by_id[sid]
            cond = condition_from_song(song, win)
            toks = sampler.generate(cond, seed=0, **kw)
            mel = sampler.melody_abs(cond, toks)
            ped = _pedal_notes(cond, 15, mel)
            rows.append(melody_stats(mel, ped, cond.n_bars))
        results[name] = M.aggregate(rows)
        show(name, results[name])
    print()

    # ---------------- teacher-forced PAD rate ----------------
    from trance_lora.features import IGNORE
    from trance_lora.train import TrainConfig, prepare

    cfg = TrainConfig(cache=os.path.join(PROJECT, args.cache))
    data = prepare(cfg)
    arrs = data["test"]
    print("teacher-forced diagnostics (test):")
    for path in args.checkpoints:
        m, _ = load_checkpoint(path)
        with torch.no_grad():
            logits = m(
                prev=torch.from_numpy(arrs["prev"]), chord=torch.from_numpy(arrs["chord"]),
                layer=torch.from_numpy(arrs["layer"]), root=torch.from_numpy(arrs["root"]),
                degree=torch.from_numpy(arrs["degree"]), barpos=torch.from_numpy(arrs["barpos"]),
                slotpos=torch.from_numpy(arrs["slotpos"]),
                template=torch.from_numpy(arrs["template"]))
        pred = logits.argmax(-1)
        target = torch.from_numpy(arrs["target"])
        mask = target != IGNORE
        pad_id = D.PAD
        pad_rate = float((pred[mask] == pad_id).float().mean())
        acc = float((pred[mask] == target[mask]).float().mean())
        notpad = pred[mask] != pad_id
        acc_nopad = float((pred[mask] == target[mask])[notpad].float().mean()) if notpad.any() else 0.0
        print(f"  {os.path.basename(os.path.dirname(path)):18s} token_acc {acc:.3f} "
              f"argmax==PAD {pad_rate:.3f} acc|notPAD {acc_nopad:.3f}")

    out = os.path.join(PROJECT, "reports", "quality_diagnosis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"corpus": corpora, "generated": results}, f, ensure_ascii=False, indent=1)
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
