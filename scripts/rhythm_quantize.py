"""Quantise any melody line onto one of our rhythm patterns (16th-note grid).

The contract concession agreed for Step B: open-source models will not honour our
rhythm mask, so we take their melody and re-time it onto a chosen existing pattern.
This script does exactly that, and can also copy a pattern from a corpus song.

    # quantise an external melody onto a corpus rhythm pattern
    python scripts/rhythm_quantize.py --midi external.mid --pattern 12112112 --out out/q.mid

    # copy the rhythm of a training song, force its key
    python scripts/rhythm_quantize.py --midi external.mid --pattern from:untitled7__140__G#min__drop \
        --key G#min --bars 8 --out out/q.mid --with-pedal
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

SPB = 16  # 16th slots per bar


def load_song(song_id: str):
    from trance_lora import data as D

    songs, _, _ = D.load_cache(os.path.join(PROJECT, "data", "cache", "dataset.json"))
    for s in songs:
        if s.song_id == song_id:
            return s
    raise KeyError(f"song not found: {song_id}")


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


def infer_key(pitches):
    from trance_lora import theory

    pcs = [p % 12 for p in pitches]
    best = None
    for tonic in range(12):
        sc = theory.scale_pcs(tonic, "minor")
        frac = sum(1 for pc in pcs if pc in sc) / max(1, len(pcs))
        if best is None or frac > best[0]:
            best = (frac, tonic, "minor")
    return best[1], best[2], best[0]


def quantise_bar(events, target_slots):
    """Map an ordered list of (slot, pitch) events onto the target 16th slots."""
    if not target_slots:
        return []
    if not events:
        return []
    pitches = [p for _s, p in events]
    if len(pitches) >= len(target_slots):
        # pick evenly spaced indices, preserving order
        idx = [round(i * (len(pitches) - 1) / max(1, len(target_slots) - 1))
               for i in range(len(target_slots))]
        chosen = [pitches[i] for i in idx]
    else:
        # cycle the motif to fill the pattern
        chosen = [pitches[i % len(pitches)] for i in range(len(target_slots))]
    return list(zip(target_slots, chosen))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True)
    ap.add_argument("--pattern", required=True,
                    help="8/16-char pattern, or 'from:<song_id>' to copy a corpus song")
    ap.add_argument("--bars", type=int, default=8)
    ap.add_argument("--offset-bars", dest="offset_bars", type=int, default=0)
    ap.add_argument("--key", default=None, help="force key, e.g. G#min; default: infer")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--with-pedal", dest="with_pedal", action="store_true",
                    help="also write a pedal layer from the inferred per-bar root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import mido

    from trance_lora import theory
    from trance_lora.data import NONE

    pattern = args.pattern
    song = None
    if pattern.startswith("from:"):
        song = load_song(pattern[5:])
        pattern = song.rhythm_pattern
    bar_mask = theory.parse_rhythm(pattern, 1)
    target_slots = [i for i, v in enumerate(bar_mask) if v == 2]
    pedal_slots = [i for i, v in enumerate(bar_mask) if v == 1]

    bars_events, data = read_melody_any(args.midi, args.bars, args.offset_bars)
    pitches = [p for ev in bars_events for _s, p in ev]
    if args.key:
        tonic, mode = theory.parse_key(args.key)
    elif song is not None:
        tonic, mode = song.tonic, song.mode
    else:
        tonic, mode, _fit = infer_key(pitches) if pitches else (9, "minor", 0.0)
    bpm = args.bpm or (song.bpm if song is not None else (data.tempo_bpm or 140.0))

    out_bars = [quantise_bar(ev, target_slots) for ev in bars_events]
    # pedal: per-bar root from the highest-coverage triad match of the input bar
    pedal_notes = []
    if args.with_pedal:
        anchor = 64
        for b, ev in enumerate(bars_events):
            if not ev:
                continue
            root_pc = min(p for _s, p in ev) % 12
            pitch = theory.pedal_pitch_for_root(root_pc, tonic, anchor, 15)
            for r in pedal_slots:
                pedal_notes.append((b * SPB + r, pitch))

    from trance_lora.midi_io import GRID_TICKS
    mid = mido.MidiFile(ticks_per_beat=GRID_TICKS * 4)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    step = GRID_TICKS
    # build absolute tick events: melody note lasts one 16th
    abs_events = []
    for b, notes in enumerate(out_bars):
        for r, pitch in notes:
            t = (b * SPB + r) * step
            abs_events.append((t, "on", pitch))
            abs_events.append((t + step - 1, "off", pitch))
    for slot, pitch in pedal_notes:
        t = slot * step
        abs_events.append((t, "on", pitch))
        abs_events.append((t + step - 1, "off", pitch))
    abs_events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))
    last = 0
    for t, kind, pitch in abs_events:
        msg = mido.Message("note_on" if kind == "on" else "note_off", note=pitch,
                           velocity=100 if kind == "on" else 0, time=t - last)
        track.append(msg)
        last = t
    out = os.path.join(PROJECT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    mid.save(out)

    scale = theory.scale_pcs(tonic, mode)
    result = {
        "input": os.path.relpath(args.midi, PROJECT).replace("\\", "/"),
        "output": os.path.relpath(out, PROJECT).replace("\\", "/"),
        "pattern": pattern, "bar_mask": "".join(str(v) for v in bar_mask),
        "melody_slots_per_bar": target_slots, "pedal_slots_per_bar": pedal_slots,
        "key": theory.key_text(tonic, mode), "bpm": bpm,
        "in_key": round(sum(1 for notes in out_bars for _r, p in notes if p % 12 in scale)
                        / max(1, sum(len(n) for n in out_bars)), 4),
        "input_bars": [len(ev) for ev in bars_events],
        "output_bars": [len(n) for n in out_bars],
        "with_pedal": args.with_pedal,
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.json:
        json.dump(result, open(os.path.join(PROJECT, args.json), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
