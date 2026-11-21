"""Adapter: any external MIDI -> our contract (analysis + motif seed).

Step B needs a bridge between open-source models (their own tokenisations, their own
rhythm) and our delivery contract (key, BPM, chord progression, 16th-note rhythm mask,
melody + pedal layers).  This script is that bridge:

  analyze  score an arbitrary MIDI against our metrics (in-key, chord tones against
           the per-bar pedal root, melody/pedal layers, interval statistics)
  seed     extract the bar-0 motif as our rel-pitch tokens and let the v2 model
           develop it into a full 8-bar drop melody under the contract

Usage:
    python scripts/adapt_external_midi.py analyze --midi some.mid
    python scripts/adapt_external_midi.py seed --midi some.mid --version v2 --out out/adapted
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

SLOTS_PER_BAR = 16


def read_slots_any(path: str):
    """Read any MIDI (any PPQ) and quantise to 16th-note slots (our convention)."""
    from trance_lora.data import infer_layer_mask
    from trance_lora.midi_io import read_midi

    data = read_midi(path)
    step = max(1.0, data.ticks_per_beat / 4.0)   # one 16th in the file's own ticks
    slots: dict[int, list[int]] = {}
    for n in data.notes:
        s = int(round(n.start_tick / step))
        slots.setdefault(s, []).append(int(n.pitch))
    n = (max(slots) + 1) if slots else 0
    layer, _ = infer_layer_mask(slots, n)
    mel = {s: max(v) for s, v in slots.items() if v and layer[s] == 2}
    ped = {s: min(v) for s, v in slots.items() if v and layer[s] == 1}
    return data, slots, n, mel, ped, layer


def read_melody_any(path: str, bars: int, offset_bars: int = 0):
    """Per-bar top-voice events, one 16th grid, any PPQ."""
    data, slots, n, mel, ped, layer = read_slots_any(path)
    bars_events = []
    for b in range(offset_bars, offset_bars + bars):
        ev = []
        for r in range(16):
            s = b * 16 + r
            if s in slots and slots[s]:
                ev.append((r, max(slots[s])))
        bars_events.append(ev)
    return bars_events, data


def infer_key(pitches: list[int]):
    from trance_lora import theory

    pcs = [p % 12 for p in pitches]
    best = None
    for tonic in range(12):
        sc = theory.scale_pcs(tonic, "minor")
        frac = sum(1 for pc in pcs if pc in sc) / max(1, len(pcs))
        if best is None or frac > best[0]:
            best = (frac, tonic, "minor")
    return best[1], best[2], best[0]



def skyline(path: str):
    """Top sounding voice of a polyphonic piano part, one value per 16th slot."""
    from trance_lora.midi_io import read_midi

    data = read_midi(path)
    step = max(1.0, data.ticks_per_beat / 4.0)
    sounding: dict[int, list[int]] = {}
    for n in data.notes:
        s0 = int(round(n.start_tick / step))
        s1 = max(s0, int(round((n.start_tick + max(n.dur_tick, 1)) / step)))
        for s in range(s0, min(s1 + 1, s0 + 64)):
            sounding.setdefault(s, []).append(int(n.pitch))
    top = {s: max(v) for s, v in sounding.items() if v}
    return top, data


def analyze_poly(path: str) -> dict:
    """Metrics for a polyphonic (piano-style) part using the skyline voice."""
    from trance_lora import metrics as M
    from trance_lora import theory

    # melodic top line = highest note of each *onset* group (chord top), ignoring
    # slots where only sustained notes remain (otherwise the line jumps to the
    # held left-hand note and the interval stats become meaningless)
    from trance_lora.midi_io import read_midi

    data = read_midi(path)
    step = max(1.0, data.ticks_per_beat / 4.0)
    onsets: dict[int, list[int]] = {}
    for n in data.notes:
        onsets.setdefault(int(round(n.start_tick / step)), []).append(int(n.pitch))
    slots = sorted(onsets)
    pitches = [max(onsets[s]) for s in slots]
    tonic, mode, fit = infer_key(pitches) if pitches else (9, "minor", 0.0)
    n_bars = max(1, (max(slots) + 1) // 16) if slots else 0
    seq = list(pitches)
    ivs = [seq[i] - seq[i - 1] for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
    scale = theory.scale_pcs(tonic, mode)
    q = {"bars": n_bars, "notes": len(data.notes), "key": theory.key_text(tonic, mode),
         "in_key": round(sum(1 for p in pitches if p % 12 in scale) / max(1, len(pitches)), 4),
         "mean_abs_interval": round(sum(abs(x) for x in ivs) / max(1, len(ivs)), 3),
         "pitch_span": max(pitches) - min(pitches) if pitches else 0,
         "n_intervals": len(ivs)}
    if ivs:
        q["arp_frac"] = round(sum(1 for x in ivs if abs(x) in (3, 4, 5)) / len(ivs), 3)
        q["step_frac"] = round(sum(1 for x in ivs if abs(x) in (1, 2)) / len(ivs), 3)
        q["leap7_frac"] = round(sum(1 for x in ivs if abs(x) > 7) / len(ivs), 3)
    return q


def analyze(path: str) -> dict:
    from trance_lora import metrics as M
    from trance_lora import theory

    data, slots, n, mel, ped, layer = read_slots_any(path)
    pitches = [p for notes in slots.values() for p in notes]
    tonic, mode, key_fit = infer_key(pitches)
    n_bars = max(1, n // 16)
    scale = theory.scale_pcs(tonic, mode)
    in_key = sum(1 for p in pitches if p % 12 in scale) / max(1, len(pitches))

    triads = {root: pcs for _label, root, pcs in theory.TRIADS[mode]}
    chord_hits = tot = 0
    for b in range(n_bars):
        ss = [s for s in range(b * 16, (b + 1) * 16) if s in ped]
        if not ss:
            continue
        lows = sorted(ped[s] for s in ss if s in ped)
        root_pc = (lows[len(lows) // 2] - tonic) % 12
        pcs = triads.get(root_pc)
        if pcs is None:
            continue
        chord_pcs = {(tonic + x) % 12 for x in pcs}
        for s in range(b * 16, (b + 1) * 16):
            for p in slots.get(s, []):
                if layer[s] == 2 or s in ped:
                    tot += 1
                    chord_hits += int(p % 12 in chord_pcs)

    quality = M.melody_quality_stats(mel, ped, n_bars)
    info = {
        "file": os.path.relpath(path, PROJECT).replace("\\", "/"),
        "notes": len(data.notes), "bars": n_bars,
        "bpm": round(float(data.tempo_bpm or 0), 1),
        "key": theory.key_text(tonic, mode), "tonic": tonic, "mode": mode,
        "key_fit": round(key_fit, 4),
        "in_key": round(in_key, 4),
        "chord_tone": round(chord_hits / max(1, tot), 4) if tot else None,
        "melody_notes": len(mel), "pedal_notes": len(ped),
        "layer_separation": (min(mel.values()) - max(ped.values())) if mel and ped else None,
    }
    for k in ("quality_mean_abs_interval", "quality_bar_pitch_range",
              "quality_corr_melody_bass", "quality_jump7_rate",
              "quality_interval_entropy"):
        v = quality.get(k)
        info[k.replace("quality_", "")] = None if v is None or v != v else round(float(v), 4)
    return info


def extract_motif(path: str):
    """Bar-0 melody -> our rel-pitch token array + inferred key/anchor."""
    from trance_lora import theory
    from trance_lora.data import PAD, PITCH_OFFSET

    _data, slots, _n, mel, _ped, _layer = read_slots_any(path)
    # melody-only inputs can confuse the two-layer inference: if bar 0 yields fewer
    # than 3 melody slots, treat every note as melody
    if sum(1 for s in range(16) if s in mel) < 3:
        mel = {s: max(v) for s, v in slots.items() if v}
    pitches = list(mel.values())
    tonic, mode, _fit = infer_key(pitches) if pitches else (9, "minor", 0.0)
    first = [mel[s] for s in range(16) if s in mel]
    anchor = int(round(sum(first) / len(first))) if first else 64
    anchor = int(theory.nearest_in_key(anchor, tonic, mode))
    motif = []
    for s in range(16):
        if s not in mel:
            motif.append(PAD)
            continue
        rel = mel[s] - anchor
        motif.append(int(max(-18, min(18, rel))) + PITCH_OFFSET)
    return motif, tonic, mode, anchor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["analyze", "analyze-poly", "seed"])
    ap.add_argument("--midi", required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--bpm", type=int, default=140)
    ap.add_argument("--version", default="v2")
    ap.add_argument("--progression", default="i,VI,III,VII")
    ap.add_argument("--rhythm", default="12112112")
    ap.add_argument("--w-chord", dest="w_chord", type=float, default=None)
    ap.add_argument("--out", default="out/adapted")
    args = ap.parse_args()

    if args.cmd == "analyze-poly":
        info = analyze_poly(args.midi)
        print(json.dumps(info, ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "analyze":
        info = analyze(args.midi)
        print(json.dumps(info, ensure_ascii=False, indent=1))
        p = os.path.join(PROJECT, "reports", "external_midi_analysis.json")
        rows = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []
        rows.append(info)
        json.dump(rows, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return 0

    import numpy as np
    import torch

    from trance_lora import theory, versions
    from trance_lora.evaluate import Ensemble
    from trance_lora.generate import (GenerateResult, GenerateSpec, _pedal_notes, _stats,
                                      build_condition, describe, render_midi)
    from trance_lora.hybrid import HybridLogitModel
    from trance_lora.model import load_checkpoint
    from trance_lora.sample import MelodySampler

    motif, tonic, mode, anchor = extract_motif(args.midi)
    ver = versions.get(args.version)
    paths, decode = ver.paths(), ver.decode()
    torch.set_num_threads(4)
    if ver.mix is None:
        models = [load_checkpoint(p)[0] for p in paths["checkpoint"]]
        sampler = MelodySampler(Ensemble(models) if len(models) > 1 else models[0])
    else:
        pitch = Ensemble([load_checkpoint(p)[0] for p in paths["checkpoint"]])
        delta = Ensemble([load_checkpoint(p)[0] for p in paths["delta_checkpoint"]])
        sampler = MelodySampler(HybridLogitModel(pitch, delta, mix=ver.mix))

    key_text = args.key or theory.key_text(tonic, mode)
    spec = GenerateSpec(key=key_text, bpm=args.bpm, progression=args.progression,
                        rhythm=args.rhythm, mode=decode["mode"], seed=0,
                        **{k: v for k, v in decode.items() if k != "mode"})
    cond = build_condition(spec)
    w_chord = spec.w_chord
    if w_chord is None:
        from trance_lora.sample import calibrate_chord_bias
        _tok, w_chord, _rate = calibrate_chord_bias(
            sampler, cond, mode=spec.mode, seed=spec.seed,
            w_template=spec.w_template, w_repeat=spec.w_repeat,
            w_leap=spec.w_leap, w_range=spec.w_range, w_root=spec.w_root,
            w_bass=spec.w_bass)
    tokens = sampler.generate(cond, mode=spec.mode, w_chord=w_chord,
                              w_template=spec.w_template, w_repeat=spec.w_repeat,
                              w_leap=spec.w_leap, w_range=spec.w_range,
                              w_bass=spec.w_bass, seed=spec.seed,
                              template_tokens=np.asarray(motif, dtype=np.int64))
    melody = sampler.melody_abs(cond, tokens)
    pedal = _pedal_notes(cond, spec.pedal_offset, melody) if spec.pedal else {}
    stats = _stats(cond, tokens)
    stats["w_chord"] = w_chord
    result = GenerateResult(melody=melody, pedal=pedal, tokens=tokens, condition=cond, stats=stats)
    os.makedirs(os.path.join(PROJECT, args.out), exist_ok=True)
    path = os.path.join(PROJECT, args.out, "adapted.mid")
    render_midi(result, path, spec)
    print(f"external motif (bar-0 rel tokens): {motif}")
    print(f"inferred key: {theory.key_text(tonic, mode)}  anchor={anchor}  version={ver.name}  w_chord={w_chord}")
    print(describe(spec, result))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
