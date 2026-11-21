"""Self-check generated MIDI files against the round-2 generation contract.

Reads each generated MIDI (melody + pedal mixed on one lead track, exactly like the
corpus), re-derives the two layers with the same algorithm used for training data
and verifies the §8 self-check table of ``Trance旋律生成规范.md``:

  * length / tempo / 4-4 / 1-16 quantisation / monophonic per slot
  * rhythm mask == requested pattern
  * melody in-key (target 100%)
  * melody chord-tone rate (target 50-90%)
  * pedal follows the chord root (target 100%)
  * pedal register below the melody
  * motif consistency (corpus ≈ 0.4)

Usage:
    python scripts/check_outputs.py out/demo/*.mid
    python scripts/check_outputs.py --manifest out/demo/manifest.json
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
    ap.add_argument("files", nargs="*")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out", default="reports/output_check.json")
    args = ap.parse_args()

    import numpy as np

    from trance_lora import theory
    from trance_lora.data import (NONE, REL_MAX, REL_MIN, SLOTS_PER_BAR, infer_layer_mask)
    from trance_lora.features import rel_to_token, token_to_rel
    from trance_lora.midi_io import GRID_TICKS, notes_to_slots, read_midi
    from trance_lora.metrics import motif_consistency
    from trance_lora.sample import Condition

    files = list(args.files)
    specs: dict[str, dict] = {}
    if args.manifest and os.path.exists(os.path.join(PROJECT, args.manifest)):
        man = json.load(open(os.path.join(PROJECT, args.manifest), encoding="utf-8"))
        for it in man["items"]:
            files.append(os.path.join(PROJECT, it["file"]))
            specs[os.path.abspath(os.path.join(PROJECT, it["file"]))] = it
    if not files:
        print("no files", file=sys.stderr)
        return 2

    rows = []
    for path in files:
        abspath = os.path.abspath(path)
        spec = specs.get(abspath, {})
        data = read_midi(abspath)
        slots = notes_to_slots(data)
        poly = {s: v for s, v in slots.items() if len(set(v)) > 1}
        last_end = max((n.start_tick + n.dur_tick for n in data.notes), default=0)
        n_slots = int(round(last_end / GRID_TICKS))
        n_bars = n_slots // SLOTS_PER_BAR
        layer, diag = infer_layer_mask(slots, n_slots)
        pattern = "".join(str(layer[s]) for s in range(SLOTS_PER_BAR))
        melody = {s: max(slots[s]) for s in range(n_slots) if layer[s] == 2 and s in slots}
        pedal = {s: min(slots[s]) for s in range(n_slots) if layer[s] == 1 and s in slots}

        # condition from the requested spec (falls back to the recovered pattern)
        tonic, mode = theory.parse_key(spec.get("key", "Amin"))
        progression = theory.parse_progression(spec.get("progression", "i,VI,III,VII"),
                                               n_bars=n_bars, mode=mode)
        anchor = spec.get("anchor")
        if anchor is None:
            vals = sorted(melody.values())
            anchor = theory.nearest_in_key(vals[len(vals) // 2] if vals else 64, tonic, mode)
        n_mel = len(melody)
        med = int(np.median(sorted(melody.values()))) if melody else anchor
        anchor = theory.nearest_in_key(med, tonic, mode)

        cond = Condition(
            layer=np.asarray(layer, dtype=np.int64),
            chord=np.zeros((n_slots, 12), dtype=np.float32),
            root=np.zeros(n_slots, dtype=np.int64),
            degree=np.zeros(n_slots, dtype=np.int64),
            barpos=np.arange(n_slots) % SLOTS_PER_BAR,
            slotpos=np.arange(n_slots),
            anchor=anchor, tonic=tonic, mode=mode, n_bars=n_bars,
            bpm=int(round(data.tempo_bpm or 140)), progression=progression,
        )
        chord_pcs_by_bar = []
        for b in range(n_bars):
            root_pc, pcs = theory.triad_by_label(progression[b], mode)
            chord_pcs_by_bar.append(set(pcs))
            for s in range(b * SLOTS_PER_BAR, (b + 1) * SLOTS_PER_BAR):
                cond.root[s] = root_pc
                for pc in pcs:
                    cond.chord[s, pc] = 1.0

        in_key = sum(1 for p in melody.values() if theory.is_in_key(p, tonic, mode)) / max(1, n_mel)
        chord_hit = sum(1 for s, p in melody.items()
                        if (p - tonic) % 12 in chord_pcs_by_bar[s // SLOTS_PER_BAR])
        chord_rate = chord_hit / max(1, n_mel)
        pedal_root = 0
        for s, p in pedal.items():
            b = s // SLOTS_PER_BAR
            if (p - tonic) % 12 == (tonic + theory.triad_by_label(progression[b], mode)[0] - tonic) % 12:
                pedal_root += 1
        pedal_root_rate = pedal_root / max(1, len(pedal))
        sep = (min(melody.values()) - max(pedal.values())) if melody and pedal else None

        toks = np.full(n_slots, 0, dtype=np.int64)
        # build token stream for the motif metric
        from trance_lora.data import PAD, PITCH_OFFSET
        toks[:] = PAD
        for s, p in melody.items():
            rel = p - anchor
            if REL_MIN <= rel <= REL_MAX:
                toks[s] = rel + PITCH_OFFSET
        motif = motif_consistency(cond, toks)
        rhythm_expected = None
        if spec.get("rhythm"):
            rhythm_expected = "".join(
                str(x) for x in theory.parse_rhythm(spec["rhythm"], n_bars=1))[:16]

        # --- pedal root vs requested progression (chord change rate) -----------
        expected_roots = []
        for b in range(n_bars):
            root_pc, _pcs = theory.triad_by_label(progression[b], mode)
            expected_roots.append((tonic + root_pc) % 12)
        actual_roots = []
        for b in range(n_bars):
            ps = [p for s2, p in pedal.items() if b * SLOTS_PER_BAR <= s2 < (b + 1) * SLOTS_PER_BAR]
            actual_roots.append(min(ps) % 12 if ps else None)
        root_ok = sum(1 for e, a in zip(expected_roots, actual_roots) if a is not None and a == e)
        bar_changes = sum(1 for b in range(1, n_bars) if expected_roots[b] != expected_roots[b - 1])
        actual_changes = sum(1 for b in range(1, n_bars)
                             if actual_roots[b] is not None and actual_roots[b - 1] is not None
                             and actual_roots[b] != actual_roots[b - 1])

        rows.append({
            "file": os.path.relpath(abspath, PROJECT).replace("\\", "/"),
            "n_bars": n_bars, "bpm": round(data.tempo_bpm or 0),
            "slot_notes_ok": len(poly) == 0,
            "melody_notes": n_mel, "pedal_notes": len(pedal),
            "rhythm_pattern": pattern,
            "rhythm_matches_request": (pattern == rhythm_expected) if rhythm_expected else None,
            "in_key": in_key, "chord_tone": chord_rate,
            "pedal_follows_root": pedal_root_rate,
            "layer_separation": sep,
            "motif_consistency": motif,
            "pedal_root_matches_chord": root_ok / max(1, n_bars),
            "expected_bar_changes": bar_changes,
            "actual_bar_changes": actual_changes,
            "duration_ok": all(round(n.dur_tick / GRID_TICKS) == 1 for n in data.notes),
            "grid_ok": all(abs(n.start_tick - round(n.start_tick / GRID_TICKS) * GRID_TICKS) < 1
                           for n in data.notes),
        })

    ok = True
    print(f"{'file':42s} {'bars':>4s} {'in_key':>7s} {'chord':>6s} {'pedalRoot':>9s} "
          f"{'sep':>5s} {'motif':>6s} {'rhythm':>7s}")
    for r in rows:
        checks = []
        if r["n_bars"] != 8:
            checks.append("bars!=8")
        if r["in_key"] < 0.999:
            checks.append("in_key<1")
        if not (0.50 <= r["chord_tone"] <= 0.90):
            checks.append("chord_tone_out_of_50-90")
        if r["pedal_follows_root"] < 1.0:
            checks.append("pedal_not_root")
        if r["pedal_root_matches_chord"] < 0.999:
            checks.append("pedal_root_vs_progression")
        if r["actual_bar_changes"] != r["expected_bar_changes"]:
            checks.append("chord_change_rate")
        if not r["slot_notes_ok"]:
            checks.append("polyphonic_slot")
        if r["rhythm_matches_request"] is False:
            checks.append("rhythm_mismatch")
        if not r["duration_ok"] or not r["grid_ok"]:
            checks.append("quantisation")
        if checks:
            ok = False
        print(f"{os.path.basename(r['file']):42s} {r['n_bars']:4d} {r['in_key']:7.3f} "
              f"{r['chord_tone']:6.3f} {r['pedal_follows_root']:9.3f} "
              f"{(r['layer_separation'] if r['layer_separation'] is not None else 0):5d} "
              f"{r['motif_consistency']:6.3f} "
              f"{('OK' if r['rhythm_matches_request'] in (True, None) else 'BAD'):>7s}"
              f"{'  !! ' + ','.join(checks) if checks else ''}")

    agg_keys = ("in_key", "chord_tone", "pedal_follows_root", "motif_consistency",
                "layer_separation", "melody_notes")
    agg = {}
    for k in agg_keys:
        vals = [r[k] for r in rows if r[k] is not None and not (isinstance(r[k], float) and r[k] != r[k])]
        if vals:
            agg[k] = float(np.mean(vals))
    print("\naggregate:", {k: round(v, 4) for k, v in agg.items()})
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    os.makedirs(os.path.dirname(os.path.abspath(os.path.join(PROJECT, args.out))), exist_ok=True)
    with open(os.path.join(PROJECT, args.out), "w", encoding="utf-8") as f:
        json.dump({"files": rows, "aggregate": agg, "all_passed": ok}, f,
                  ensure_ascii=False, indent=1)
    print(f"written {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
