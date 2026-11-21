"""Evaluation metrics for generated melodies."""

from __future__ import annotations

import math

import numpy as np

from . import theory
from .data import PAD, SLOTS_PER_BAR, NONE
from .features import token_to_rel
from .sample import Condition


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def melody_rel_list(cond: Condition, tokens: np.ndarray) -> list[int]:
    return [token_to_rel(int(tokens[i])) for i in range(cond.L)
            if cond.layer[i] == 2 and tokens[i] != PAD]


def ref_rel_list(cond: Condition, melody_abs: np.ndarray | list[int]) -> list[int]:
    return [int(melody_abs[i]) - cond.anchor for i in range(cond.L)
            if cond.layer[i] == 2 and melody_abs[i] != NONE]


def interval_metrics(pred: list[int], ref: list[int]) -> dict:
    if len(pred) < 2 or len(ref) < 2 or len(pred) != len(ref):
        return {"interval_l1": float("nan"), "contour_r": float("nan"),
                "dup_rate": float("nan"), "ref_dup_rate": float("nan"),
                "interval_dist_l1": float("nan")}
    p, r = np.asarray(pred, float), np.asarray(ref, float)
    dp, dr = np.diff(p), np.diff(r)
    return {
        "interval_l1": float(np.mean(np.abs(dp - dr))),
        "interval_dist_l1": _interval_dist_l1(dp, dr),
        "contour_r": _corr(p, r),
        "dup_rate": float(np.mean(dp == 0)),
        "ref_dup_rate": float(np.mean(dr == 0)),
    }


def _interval_dist_l1(dp: np.ndarray, dr: np.ndarray, lo: int = -12, hi: int = 12) -> float:
    """L1 distance between the interval histograms, in [0, 2] (same scale as the
    round-2 documentation's ``interval_l1`` ≈ 0.85)."""
    bins = np.arange(lo, hi + 2) - 0.5
    hp, _ = np.histogram(np.clip(dp, lo, hi), bins=bins)
    hr, _ = np.histogram(np.clip(dr, lo, hi), bins=bins)
    hp = hp / max(1, hp.sum())
    hr = hr / max(1, hr.sum())
    return float(np.abs(hp - hr).sum())


def distribution_metrics(pred: list[int], ref: list[int] | None = None) -> dict:
    if not pred:
        return {}
    p = np.asarray(pred, float)
    out = {
        "distinct_pitches": int(len(np.unique(p))),
        "pitch_span": int(p.max() - p.min()),
        "mean_abs_interval": float(np.mean(np.abs(np.diff(p)))) if len(p) > 1 else 0.0,
        "jump_rate": float(np.mean(np.abs(np.diff(p)) > 10)) if len(p) > 1 else 0.0,
        "octave_range_hi": int(p.max()),
        "octave_range_lo": int(p.min()),
    }
    if ref is not None and len(ref) == len(pred):
        r = np.asarray(ref, float)
        out["ref_distinct_pitches"] = int(len(np.unique(r)))
        out["ref_pitch_span"] = int(r.max() - r.min())
        out["ref_mean_abs_interval"] = float(np.mean(np.abs(np.diff(r)))) if len(r) > 1 else 0.0
    return out


def motif_consistency(cond: Condition, tokens: np.ndarray, max_steps: int = 7) -> float:
    """Mean correlation between bar-0 motif (best in-scale transposition) and later bars.

    Mirrors the corpus metric described in the round-2 documentation (~0.40).
    """
    melody = {}
    for i in range(cond.L):
        if cond.layer[i] == 2 and tokens[i] != PAD:
            melody[i] = token_to_rel(int(tokens[i]))
    motif = {r: melody[r] for r in range(SLOTS_PER_BAR) if r in melody}
    if len(motif) < 3:
        return float("nan")
    scores: list[float] = []
    for b in range(1, cond.n_bars):
        rows_x, rows_y = [], []
        matched = []
        for r in range(SLOTS_PER_BAR):
            i = b * SLOTS_PER_BAR + r
            if r in motif and i in melody:
                matched.append(r)
        if len(matched) < 3:
            continue
        best = -math.inf
        for s in range(-max_steps, max_steps + 1):
            xs, ys = [], []
            for r in matched:
                t = theory.transpose_rel(motif[r], s, cond.anchor, cond.tonic, cond.mode)
                if t is None:
                    continue
                xs.append(t)
                ys.append(melody[b * SLOTS_PER_BAR + r])
            if len(xs) >= 3:
                c = _corr(np.asarray(xs, float), np.asarray(ys, float))
                if not math.isnan(c) and c > best:
                    best = c
        if best > -math.inf:
            scores.append(best)
    return float(np.mean(scores)) if scores else float("nan")


def motif_consistency_abs(cond: Condition, melody_abs: dict[int, int], max_steps: int = 7) -> float:
    """Same metric but from absolute pitches (used for corpus reference)."""
    melody = {i: p - cond.anchor for i, p in melody_abs.items()
              if cond.layer[i] == 2 and p != NONE}
    tokens = np.full(cond.L, PAD, dtype=np.int64)
    from .features import rel_to_token
    for i, rel in melody.items():
        tokens[i] = rel_to_token(rel)
    return motif_consistency(cond, tokens, max_steps=max_steps)


def evaluate_tokens(cond: Condition, tokens: np.ndarray,
                    ref_abs: np.ndarray | list[int] | None = None) -> dict:
    pred_rel = melody_rel_list(cond, tokens)
    out = {
        "n_melody": len(pred_rel),
        "in_key": _key_rate(cond, tokens),
        "chord_tone": _chord_rate(cond, tokens),
        "motif_consistency": motif_consistency(cond, tokens),
        **distribution_metrics(pred_rel),
    }
    if ref_abs is not None:
        ref = ref_rel_list(cond, ref_abs)
        out.update(interval_metrics(pred_rel, ref))
    return out


def _key_rate(cond: Condition, tokens: np.ndarray) -> float:
    total = hit = 0
    for i in range(cond.L):
        if cond.layer[i] != 2 or tokens[i] == PAD:
            continue
        total += 1
        if theory.is_in_key(cond.anchor + token_to_rel(int(tokens[i])), cond.tonic, cond.mode):
            hit += 1
    return hit / max(1, total)


def _chord_rate(cond: Condition, tokens: np.ndarray) -> float:
    total = hit = 0
    for i in range(cond.L):
        if cond.layer[i] != 2 or tokens[i] == PAD:
            continue
        total += 1
        if cond.is_chord_tone(i, token_to_rel(int(tokens[i]))):
            hit += 1
    return hit / max(1, total)


def melody_quality_stats(melody_pitches: dict[int, int], pedal_pitches: dict[int, int],
                         n_bars: int, slots_per_bar: int = 16) -> dict:
    """Musical-shape statistics used to diagnose the round-3 listener feedback.

    * big leaps (fraction of intervals > 7 semitones)
    * per-bar pitch range (melodic undulation)
    * correlation between the bar-mean melody register and the bar-mean pedal
      (does the melody follow the bass?)
    """
    bar_pitch, bar_bass = [], []
    all_ints: list[float] = []
    ranges: list[float] = []
    jumps7 = jumps10 = 0
    for b in range(n_bars):
        seq = [melody_pitches[i] for i in range(b * slots_per_bar, (b + 1) * slots_per_bar)
               if i in melody_pitches]
        if len(seq) >= 2:
            ints = np.abs(np.diff(seq))
            all_ints.extend(ints.tolist())
            jumps7 += int((ints > 7).sum())
            jumps10 += int((ints > 10).sum())
            ranges.append(float(ints.max()))
        if seq:
            bar_pitch.append(float(np.mean(seq)))
        ped = [pedal_pitches[i] for i in range(b * slots_per_bar, (b + 1) * slots_per_bar)
               if i in pedal_pitches]
        bar_bass.append(float(np.mean(ped)) if ped else float("nan"))
    out = {
        "quality_mean_abs_interval": float(np.mean(all_ints)) if all_ints else float("nan"),
        "quality_p90_interval": float(np.percentile(all_ints, 90)) if all_ints else float("nan"),
        "quality_jump7_rate": jumps7 / max(1, len(all_ints)),
        "quality_jump10_rate": jumps10 / max(1, len(all_ints)),
        "quality_bar_pitch_range": float(np.mean(ranges)) if ranges else float("nan"),
    }
    # interval-distribution entropy (bits): diversity / "non-mechanical" proxy
    if all_ints:
        signed = []
        for b in range(n_bars):
            seq = [melody_pitches[i] for i in range(b * slots_per_bar, (b + 1) * slots_per_bar)
                   if i in melody_pitches]
            signed.extend(np.diff(seq).tolist())
        bins = np.arange(-13, 14) - 0.5
        hp, _ = np.histogram(np.clip(signed, -12, 12), bins=bins)
        pr = hp / max(1, hp.sum())
        pr = pr[pr > 0]
        out["quality_interval_entropy"] = float(-(pr * np.log2(pr)).sum())
    bp = np.asarray(bar_pitch, dtype=float)
    bb = np.asarray(bar_bass, dtype=float)
    ok = ~np.isnan(bb)
    if len(bp) >= 3:
        out["quality_bar_pitch_std"] = float(bp.std())
        out["quality_bar_pitch_spread"] = float(bp.max() - bp.min())
        if ok.sum() >= 3 and bp[ok].std() > 1e-9 and bb[ok].std() > 1e-9:
            out["quality_corr_melody_bass"] = float(np.corrcoef(bp[ok], bb[ok])[0, 1])
    return out


def aggregate(rows: list[dict]) -> dict:
    keys = set()
    for r in rows:
        keys |= set(r)
    out = {}
    for k in sorted(keys):
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))
                and not (isinstance(r[k], float) and math.isnan(r[k]))]
        if vals:
            out[k] = float(np.mean(vals))
    return out
