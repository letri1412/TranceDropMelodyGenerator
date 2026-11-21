"""High-level generation: condition spec -> constrained melody -> MIDI file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import theory
from .data import SLOTS_PER_BAR, PAD, WINDOW_SLOTS
from .features import token_to_rel
from .midi_io import write_chord_track, write_midi
from .sample import (Condition, MelodySampler, calibrate_chord_bias, condition_from_spec)


@dataclass
class GenerateSpec:
    key: str = "Amin"
    bpm: int = 140
    progression: str = "i,VI,III,VII"
    rhythm: str = "12112112"
    bars: int = 8
    anchor: int | None = None
    mode: str = "plan_root"       # free | motif | plan | plan_root (default)
    variation: float = 0.0        # 0..1, deviation from the motif plan
    w_chord: float | None = None  # None = auto-calibrate to 50-90 %
    w_template: float = 2.0
    w_repeat: float = 1.2
    w_leap: float = 0.4           # interval penalty beyond 5 semitones
    w_range: float = 0.3          # keep the melody near the song register
    w_root: float = 0.0           # extra bias toward the chord root
    w_bass: float = 1.5           # pull the bar register toward the bass contour
    temperature: float = 1.0
    top_k: int = 0
    greedy: bool = False
    seed: int = 0
    pedal: bool = True
    pedal_offset: int = 15
    chord_track: bool = False


@dataclass
class GenerateResult:
    melody: dict[int, int]        # slot -> absolute pitch
    pedal: dict[int, int]         # slot -> absolute pitch
    tokens: np.ndarray
    condition: Condition
    stats: dict = field(default_factory=dict)


def build_condition(spec: GenerateSpec) -> Condition:
    tonic, mode = theory.parse_key(spec.key)
    progression = theory.parse_progression(spec.progression, n_bars=spec.bars, mode=mode)
    rhythm = theory.parse_rhythm(spec.rhythm, n_bars=spec.bars)
    anchor = spec.anchor
    if anchor is None:
        anchor = theory.nearest_in_key(64, tonic, mode)
    return condition_from_spec(tonic, mode, spec.bpm, progression, rhythm, anchor=anchor)


def _pedal_notes(cond: Condition, offset: int, melody: dict[int, int],
                 min_gap: int = 0) -> dict[int, int]:
    """Pedal = chord root inside a fixed band below the melody (anchor-22 .. anchor-12).

    The melody register is masked to >= anchor-8, so the two layers stay >=4
    semitones apart and the pedal line is a deterministic function of the chords
    (the same line the melody is asked to follow at decoding time).

    ``min_gap``: if the rendered melody dips below the band anyway (e.g. primed
    external motifs), push the whole pedal line down so that
    min(melody) - max(pedal) >= min_gap.
    """
    lo, hi = cond.anchor - 22, cond.anchor - 12
    target = cond.anchor - offset
    out = {}
    for i in range(cond.L):
        if cond.layer[i] != 1:
            continue
        pc = (cond.tonic + int(cond.root[i])) % 12
        cands = [p for p in range(0, 128) if p % 12 == pc and lo <= p <= hi]
        if not cands:
            cands = [p for p in range(0, 128) if p % 12 == pc and p <= cond.anchor - 12] \
                or [p for p in range(0, 128) if p % 12 == pc]
        out[i] = min(cands, key=lambda p: (abs(p - target), p))
    if min_gap and out and melody:
        deficit = min_gap - (min(melody.values()) - max(out.values()))
        if deficit > 0:
            shift = int(deficit + 1)
            out = {i: p - shift for i, p in out.items()}
    return out


def _stats(cond: Condition, tokens: np.ndarray) -> dict:
    from .metrics import evaluate_tokens
    s = evaluate_tokens(cond, tokens)
    s["anchor"] = cond.anchor
    s["progression"] = cond.progression
    s["rhythm"] = cond.rhythm_pattern
    return s


def generate(sampler: MelodySampler, spec: GenerateSpec) -> GenerateResult:
    cond = build_condition(spec)
    w_chord = spec.w_chord
    tokens = None
    if w_chord is None:
        # auto-calibrate chord-tone usage to the 50-90 % contract
        mode = spec.mode if spec.mode in ("free", "motif", "plan", "plan_root") else "free"
        tokens, used_w, _rate = calibrate_chord_bias(
            sampler, cond, mode=mode, seed=spec.seed, temperature=spec.temperature,
            w_template=spec.w_template, w_repeat=spec.w_repeat,
            w_leap=spec.w_leap, w_range=spec.w_range, w_root=spec.w_root,
            w_bass=spec.w_bass, greedy=spec.greedy)
        w_chord = used_w
    if tokens is None:
        tokens = sampler.generate(
            cond, mode=spec.mode, greedy=spec.greedy, temperature=spec.temperature,
            w_chord=w_chord, w_template=spec.w_template, w_repeat=spec.w_repeat,
            w_leap=spec.w_leap, w_range=spec.w_range, w_root=spec.w_root,
            w_bass=spec.w_bass, top_k=spec.top_k, seed=spec.seed)
    tokens = _apply_variation(cond, sampler, tokens, spec.variation, spec.seed)
    melody = sampler.melody_abs(cond, tokens)
    pedal = _pedal_notes(cond, spec.pedal_offset, melody) if spec.pedal else {}
    stats = _stats(cond, tokens)
    stats["w_chord"] = w_chord
    stats["variation"] = spec.variation
    return GenerateResult(melody=melody, pedal=pedal, tokens=tokens, condition=cond, stats=stats)


def _apply_variation(cond: Condition, sampler: MelodySampler, tokens: np.ndarray,
                     variation: float, seed: int) -> np.ndarray:
    from .sample import apply_variation
    return apply_variation(sampler, cond, tokens, variation, seed)


def render_midi(result: GenerateResult, path: str, spec: GenerateSpec) -> str:
    cond = result.condition
    notes = [(slot, pitch) for slot, pitch in {**result.melody, **result.pedal}.items()]
    notes.sort(key=lambda x: (x[0], x[1]))
    extra = []
    if spec.chord_track:
        chords = _reference_chords(cond)
        extra.append(("Chords", chords))
    write_midi(path, bpm=spec.bpm, notes=notes, track_name="Lead", extra_tracks=extra)
    return path


def _reference_chords(cond: Condition) -> list[tuple[int, int, list[int]]]:
    """Voicing of the reference chord track (3 notes, centre near anchor-7)."""
    out = []
    for b, label in enumerate(cond.progression or []):
        root, pcs = theory.triad_by_label(label, cond.mode)
        centre = cond.anchor - 7
        pitches = []
        for pc in pcs:
            target_pc = (cond.tonic + pc) % 12
            pitches.append(min((p for p in range(0, 128) if p % 12 == target_pc),
                               key=lambda p: abs(p - centre)))
        out.append((b * SLOTS_PER_BAR, SLOTS_PER_BAR, sorted(set(pitches))))
    return out


def _chord_spans(progression: list[str] | None) -> list[int]:
    """Length in bars of each maximal run of identical chord labels."""
    if not progression:
        return []
    spans, run = [], 1
    for b in range(1, len(progression)):
        if progression[b] == progression[b - 1]:
            run += 1
        else:
            spans.append(run)
            run = 1
    spans.append(run)
    return spans


def describe(spec: GenerateSpec, result: GenerateResult) -> str:
    st = result.stats
    spans = _chord_spans(result.condition.progression)
    warn = ""
    if spans and max(spans) > 2:
        warn = (f"  ← 警告：有和弦持续 {max(spans)} 小节（语料 89% 是 2 小节换一次，"
                "想每 2 小节换请写 4 个不同和弦，想每 1 小节换请写 8 个）")
    lines = [
        f"key={spec.key} bpm={spec.bpm} bars={spec.bars} anchor={result.condition.anchor}",
        f"progression={'-'.join(result.condition.progression or [])}  rhythm={result.condition.rhythm_pattern}"
        f"  chord_spans={spans}{warn}",
        f"mode={spec.mode} variation={spec.variation} w_chord={st.get('w_chord')}",
        f"melody notes={len(result.melody)}  pedal notes={len(result.pedal)}",
        f"in_key={st.get('in_key', float('nan')):.3f}  chord_tone={st.get('chord_tone', float('nan')):.3f}"
        f"  motif={st.get('motif_consistency', float('nan')):.3f}",
        f"distinct={st.get('distinct_pitches')}  span={st.get('pitch_span')}"
        f"  mean_abs_interval={st.get('mean_abs_interval', 0):.2f}",
    ]
    return "\n".join(lines)


def ascii_piano_roll(cond: Condition, result: GenerateResult) -> str:
    """Text piano roll (melody only) for quick inspection in a terminal."""
    pitches = sorted(result.melody.values())
    if not pitches:
        return "(empty)"
    lo, hi = min(pitches) - 1, max(pitches) + 1
    rows = []
    for p in range(hi, lo - 1, -1):
        cells = "".join("X" if result.melody.get(i) == p else
                        ("." if cond.layer[i] == 2 else " ") for i in range(cond.L))
        rows.append(f"{p:3d} |{cells}|")
    rows.append("    +" + "-" * cond.L + "+")
    return "\n".join(rows)
