"""Path A: take a melody's bar-0 motif and develop it into an 8-bar drop melody
with the v2 model, under the full contract (key / chords / rhythm / pedal).

The three songs requested by the listener are training-set songs, so this validates
the bridge and lets them hear "motif kept, development changed".  A held-out song
can be added with --heldout to see the same pipeline on unseen material.

    python scripts/pathA_develop.py
    python scripts/pathA_develop.py --heldout
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    sampler = MelodySampler(HybridLogitModel(pitch, delta, mix=ver.mix))
    return sampler, ver


def song_files(song) -> tuple[str, str]:
    lead = os.path.join(DATA_ROOT, song.folder, f"{song.song_id}__lead.mid")
    chord = os.path.join(DATA_ROOT, song.folder, f"{song.song_id}__chord.mid")
    return lead, chord


def motif_from_song(song, cond) -> list[int]:
    from trance_lora.data import PAD, PITCH_OFFSET

    motif = []
    for s in range(16):
        if song.layer[s] == 2:
            rel = int(song.melody_abs[s]) - cond.anchor
            motif.append(int(max(-18, min(18, rel))) + PITCH_OFFSET)
        else:
            motif.append(PAD)
    return motif


def original_quality(song) -> dict:
    from trance_lora import metrics as M
    from trance_lora.data import NONE

    mel = {s: song.melody_abs[s] for s in range(len(song.layer)) if song.layer[s] == 2}
    ped = {s: song.pedal_abs[s] for s in range(len(song.layer))
           if song.layer[s] == 1 and song.pedal_abs[s] != NONE}
    return M.melody_quality_stats(mel, ped, song.n_bars)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/pathA")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--heldout", action="store_true", help="also run one held-out (test) song")
    ap.add_argument("--calib", default="0,0.75,1.5,2.5,4")
    args = ap.parse_args()

    import numpy as np

    from trance_lora import data as D
    from trance_lora import metrics as M
    from trance_lora import theory
    from trance_lora.generate import (GenerateSpec, _pedal_notes, _stats, build_condition,
                                      describe, render_midi)
    from trance_lora.sample import PAD, token_to_rel

    songs, _, _ = D.load_cache(os.path.join(PROJECT, "data", "cache", "dataset.json"))
    by = {s.song_id: s for s in songs}
    targets = list(SONGS)
    if args.heldout:
        split = D.split_songs([s.song_id for s in songs], seed=42)
        targets.append(split["test"][0])

    sampler, ver = load_v2_sampler()
    decode = ver.decode()
    out_root = os.path.join(PROJECT, args.out)
    os.makedirs(out_root, exist_ok=True)
    manifest = {"version": ver.name, "decode": decode, "items": []}
    cands = tuple(float(x) for x in args.calib.split(","))

    for sid in targets:
        song = by[sid]
        tag = SHORT.get(sid, sid.split("__")[0])
        out_dir = os.path.join(out_root, tag)
        os.makedirs(out_dir, exist_ok=True)
        lead, _chord = song_files(song)
        mel_pitches = [song.melody_abs[s] for s in range(len(song.layer)) if song.layer[s] == 2]
        src_anchor = int(round(sum(mel_pitches) / len(mel_pitches))) if mel_pitches else 64
        src_anchor = int(theory.nearest_in_key(src_anchor, song.tonic, song.mode))
        spec = GenerateSpec(key=theory.key_text(song.tonic, song.mode), bpm=song.bpm,
                            progression=",".join(song.chord_labels),
                            rhythm=song.rhythm_pattern, mode=decode["mode"],
                            anchor=src_anchor, seed=0,
                            **{k: v for k, v in decode.items() if k != "mode"})
        cond = build_condition(spec)
        motif = motif_from_song(song, cond)
        motif_arr = np.asarray(motif, dtype=np.int64)
        tokens, w_chord, rate = None, 0.0, 0.0
        best = None
        for w in cands:
            prime0 = [(slot, token_to_rel(int(t))) for slot, t in enumerate(motif)
                      if t != PAD and cond.layer[slot] == 2]
            tok = sampler.generate(cond, mode=spec.mode, seed=0, w_chord=w,
                                   w_template=decode["w_template"],
                                   w_repeat=decode["w_repeat"],
                                   w_leap=decode["w_leap"],
                                   w_range=decode["w_range"],
                                   w_bass=decode["w_bass"],
                                   template_tokens=motif_arr, prime=prime0)
            g = M.evaluate_tokens(cond, tok)
            r = g.get("chord_tone", 0.0)
            score = abs(r - 0.80)
            if best is None or score < best[0] or (0.50 <= r <= 0.90 and not (0.50 <= best[2] <= 0.90)):
                best = (score, tok, r, w)
        if best is not None:
            _s, tokens, rate, w_chord = best
        takes = []
        prime = [(slot, token_to_rel(int(tok))) for slot, tok in enumerate(motif)
                 if tok != PAD and cond.layer[slot] == 2]
        for seed in range(args.seeds):
            tok = sampler.generate(cond, mode=spec.mode, seed=seed, w_chord=w_chord,
                                   w_template=decode["w_template"],
                                   w_repeat=decode["w_repeat"], w_leap=decode["w_leap"],
                                   w_range=decode["w_range"], w_bass=decode["w_bass"],
                                   template_tokens=np.asarray(motif, dtype=np.int64),
                                   prime=prime)
            mel = sampler.melody_abs(cond, tok)
            ped = _pedal_notes(cond, spec.pedal_offset, mel)
            path = os.path.join(out_dir, f"developed_seed{seed}.mid")
            from trance_lora.generate import GenerateResult
            res = GenerateResult(melody=mel, pedal=ped, tokens=tok, condition=cond,
                                 stats={**_stats(cond, tok), "w_chord": w_chord})
            render_midi(res, path, spec)
            q = M.melody_quality_stats(mel, ped, cond.n_bars)
            g = M.evaluate_tokens(cond, tok)
            takes.append({"seed": seed, "file": os.path.relpath(path, PROJECT).replace("\\", "/"),
                          "quality": q, "gen": g, "describe": describe(spec, res)})
        shutil.copyfile(lead, os.path.join(out_dir, "original_lead.mid"))
        oq = original_quality(song)
        best = takes[0]
        manifest["items"].append({
            "song": sid, "tag": tag, "key": theory.key_text(song.tonic, song.mode),
            "bpm": song.bpm, "chords": song.chord_labels, "rhythm": song.rhythm_pattern,
            "motif_tokens": motif, "w_chord": w_chord,
            "original_lead": os.path.relpath(os.path.join(out_dir, "original_lead.mid"),
                                             PROJECT).replace("\\", "/"),
            "takes": takes,
        })
        o = oq
        t = best["quality"]
        gg = best["gen"]
        print(f"[{tag}] key={theory.key_text(song.tonic, song.mode)} "
              f"chords={','.join(song.chord_labels)} rhythm={song.rhythm_pattern}")
        print(f"   original : meanΔ {o.get('quality_mean_abs_interval',0):5.2f} "
              f"barRng {o.get('quality_bar_pitch_range',0):5.2f} "
              f"corrB {o.get('quality_corr_melody_bass',0):+5.2f} "
              f"jump7 {o.get('quality_jump7_rate',0):5.3f}")
        print(f"   developed: meanΔ {t.get('quality_mean_abs_interval',0):5.2f} "
              f"barRng {t.get('quality_bar_pitch_range',0):5.2f} "
              f"corrB {t.get('quality_corr_melody_bass',0):+5.2f} "
              f"jump7 {t.get('quality_jump7_rate',0):5.3f} | "
              f"motif {gg.get('motif_consistency',0):.3f} chord {gg.get('chord_tone',0):.3f} "
              f"in_key {gg.get('in_key',0):.3f} (w_chord={w_chord})")

    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n{len(targets)} songs -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
