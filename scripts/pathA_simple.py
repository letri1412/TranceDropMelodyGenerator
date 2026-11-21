"""Path A variant study: keep the "simple repetition" character of the source.

Listener feedback: the developed melody loses the original's charm because the first
four notes of every bar are no longer consistent with each other (the originals are
built on repeated-note cells / literal transposition), and forcing 70-90% chord tones
plus the anti-repeat penalty turns the line into a busy arpeggio.

Variants (same model, decoding only):
  current  : bar-0 prime only, auto chord calibration (70-90%), w_repeat 1.2
  simple   : prime the first K melody notes of EVERY bar with the transposed motif
             cell, w_chord 0.3, w_repeat 0.0, w_template 3.0
  simple+  : same but w_template 5.0 (harder adherence)
  literal  : no model freedom for the cell (pure transposition); reference upper bound
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

DATA_ROOT = r"F:\DSHWorkSpace\Trance旋律训练集"
SONGS = [
    "Life_Without_The_Truth__140__Dmin__drop",
    "untitled7__140__G#min__drop",
    "untitled14__140__Amin__drop",
]
SHORT = {SONGS[0]: "dmin_life", SONGS[1]: "g#min_untitled7", SONGS[2]: "amin_untitled14"}

VARIANTS = {
    "current": dict(prime_k=0, w_chord="auto", w_repeat=1.2, w_template=2.0, w_leap=0.0),
    "simple": dict(prime_k=4, w_chord=0.3, w_repeat=0.0, w_template=3.0, w_leap=0.0),
    "simple_hi": dict(prime_k=4, w_chord=0.2, w_repeat=0.0, w_template=5.0, w_leap=0.0),
    "free2": dict(prime_k=2, w_chord=0.4, w_repeat=0.0, w_template=2.5, w_leap=0.0),
    "free3": dict(prime_k=3, w_chord=0.4, w_repeat=0.0, w_template=2.5, w_leap=0.0),
    "literal": dict(prime_k=16, w_chord=0.0, w_repeat=0.0, w_template=9.0, w_leap=0.0),
}


def load_v2_sampler():
    import torch

    from trance_lora import versions
    from trance_lora.evaluate import Ensemble
    from trance_lora.hybrid import HybridLogitModel
    from trance_lora.model import load_checkpoint
    from trance_lora.sample import MelodySampler

    ver = versions.get("v2")
    paths = ver.paths()
    pitch = Ensemble([load_checkpoint(p)[0] for p in paths["checkpoint"]])
    delta = Ensemble([load_checkpoint(p)[0] for p in paths["delta_checkpoint"]])
    torch.set_num_threads(4)
    return MelodySampler(HybridLogitModel(pitch, delta, mix=ver.mix))


def cell_consistency(tokens, cond, n_notes: int = 4) -> float:
    """Fraction of bars whose first N melody notes equal the bar-0 cell (any transposition)."""
    from trance_lora.sample import PAD, token_to_rel

    seqs = []
    for b in range(cond.n_bars):
        seq = [token_to_rel(int(tokens[b * 16 + r])) for r in range(16)
               if tokens[b * 16 + r] != PAD and cond.layer[b * 16 + r] == 2]
        seqs.append(seq[:n_notes])
    base = seqs[0]
    if len(base) < n_notes:
        return float("nan")
    base_c = [x - base[0] for x in base]
    hits = 0
    for seq in seqs[1:]:
        if len(seq) < n_notes:
            continue
        c = [x - seq[0] for x in seq]
        hits += int(c == base_c)
    return hits / max(1, len(seqs) - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/pathA_simple")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--freedom", type=float, default=None,
                    help="0 = simple (prime first 4 notes of every bar), 1 = no priming; "
                         "intermediate values prime round((1-freedom)*4) notes; when set, "
                         "only the matching freedom variant is rendered")
    args = ap.parse_args()

    import numpy as np

    from trance_lora import data as D
    from trance_lora import metrics as M
    from trance_lora import theory
    from trance_lora.generate import (GenerateResult, GenerateSpec, _pedal_notes, _stats,
                                      build_condition, render_midi)
    from trance_lora.sample import PAD, token_to_rel

    songs, _, _ = D.load_cache(os.path.join(PROJECT, "data", "cache", "dataset.json"))
    by = {s.song_id: s for s in songs}
    sampler = load_v2_sampler()
    out_root = os.path.join(PROJECT, args.out)
    os.makedirs(out_root, exist_ok=True)
    manifest = {"items": []}

    for sid in SONGS:
        song = by[sid]
        tag = SHORT[sid]
        out_dir = os.path.join(out_root, tag)
        os.makedirs(out_dir, exist_ok=True)
        mel_pitches = [song.melody_abs[s] for s in range(len(song.layer)) if song.layer[s] == 2]
        anchor = int(theory.nearest_in_key(round(sum(mel_pitches) / len(mel_pitches)),
                                          song.tonic, song.mode))
        spec = GenerateSpec(key=theory.key_text(song.tonic, song.mode), bpm=song.bpm,
                            progression=",".join(song.chord_labels), rhythm=song.rhythm_pattern,
                            mode="plan_root", anchor=anchor, seed=0)
        cond = build_condition(spec)
        motif = [int(max(-18, min(18, int(song.melody_abs[s]) - cond.anchor))) + 18
                 if song.layer[s] == 2 else PAD for s in range(16)]
        motif_arr = np.asarray(motif, dtype=np.int64)
        template = sampler.build_plan_template(cond, motif_arr, variant="root",
                                               bass_line=sampler._bass_register_line(cond))
        # transposed cell per bar from the plan template
        cell = {}
        for b in range(cond.n_bars):
            slots = [r for r in range(16) if template[b * 16 + r] != PAD]
            cell[b] = [(b * 16 + r, token_to_rel(int(template[b * 16 + r]))) for r in slots]

        print(f"=== {tag} ({theory.key_text(song.tonic, song.mode)}) "
              f"orig chord_tone {song.stats['chord_tone_melody']:.3f}")
        variant_items = list(VARIANTS.items())
        if args.freedom is not None:
            f = max(0.0, min(1.0, args.freedom))
            k = int(round((1.0 - f) * 4))
            if f >= 0.999:
                k = 0
            w_chord = 0.4 if k < 4 else 0.3
            name = f"freedom{f:.2f}"
            variant_items = [(name, dict(prime_k=k, w_chord=w_chord, w_repeat=0.0,
                                         w_template=2.5, w_leap=0.0))]
        for name, cfg in variant_items:
            prime = []
            for b in range(cond.n_bars):
                k = cfg["prime_k"]
                for slot, rel in cell.get(b, [])[:k]:
                    if cond.layer[slot] == 2:
                        prime.append((slot, rel))
            w_chord = cfg["w_chord"]
            if w_chord == "auto":
                best = None
                for w in (0.75, 1.5, 2.5, 4.0):
                    tok = sampler.generate(cond, mode="plan_root", seed=0, w_chord=w,
                                           w_template=cfg["w_template"],
                                           w_repeat=cfg["w_repeat"], w_leap=cfg["w_leap"],
                                           w_range=0.2, w_bass=0.8,
                                           template_tokens=motif_arr, prime=prime)
                    r = M.evaluate_tokens(cond, tok).get("chord_tone", 0)
                    if 0.50 <= r <= 0.90:
                        best = (w, tok)
                        break
                w_chord, _ = best if best else (0.75, None)
            for seed in range(args.seeds):
                tok = sampler.generate(cond, mode="plan_root", seed=seed, w_chord=w_chord,
                                       w_template=cfg["w_template"], w_repeat=cfg["w_repeat"],
                                       w_leap=cfg["w_leap"], w_range=0.2, w_bass=0.8,
                                       template_tokens=motif_arr, prime=prime)
                mel = sampler.melody_abs(cond, tok)
                ped = _pedal_notes(cond, spec.pedal_offset, mel, min_gap=8)
                path = os.path.join(out_dir, f"{name}_seed{seed}.mid")
                res = GenerateResult(melody=mel, pedal=ped, tokens=tok, condition=cond,
                                     stats={**_stats(cond, tok)})
                render_midi(res, path, spec)
                g = M.evaluate_tokens(cond, tok)
                q = M.melody_quality_stats(mel, ped, cond.n_bars)
                cc = cell_consistency(tok, cond, 4)
                manifest["items"].append({
                    "song": sid, "tag": tag, "variant": name, "seed": seed,
                    "w_chord": w_chord, "w_template": cfg["w_template"],
                    "w_repeat": cfg["w_repeat"], "prime_k": cfg["prime_k"],
                    "file": os.path.relpath(path, PROJECT).replace("\\", "/"),
                    "cell_consistency_4": None if cc != cc else round(cc, 3),
                    "motif": round(g.get("motif_consistency", 0) or 0, 3),
                    "chord_tone": round(g.get("chord_tone", 0), 3),
                    "in_key": round(g.get("in_key", 0), 3),
                    "mean_interval": round(q.get("quality_mean_abs_interval", 0), 2),
                    "bar_range": round(q.get("quality_bar_pitch_range", 0), 2),
                    "corr_bass": round(q.get("quality_corr_melody_bass", 0) or 0, 2),
                })
                print(f"  {name:10s} seed{seed} cell4 {cc if cc==cc else 0:.3f} "
                      f"motif {g.get('motif_consistency',0):.3f} chord {g.get('chord_tone',0):.3f} "
                      f"meanΔ {q.get('quality_mean_abs_interval',0):.2f} "
                      f"barRng {q.get('quality_bar_pitch_range',0):.2f}")

    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n-> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
