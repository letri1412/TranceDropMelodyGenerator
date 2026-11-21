"""Evaluation: generation metrics on song-held-out test songs + report writing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from . import data as D
from . import metrics as M
from .data import N_DELTA_VOCAB, N_TOKENS, PAD, SLOTS_PER_BAR
from .features import build_arrays, rel_to_token, token_to_rel
from .model import load_checkpoint
from .generate import _pedal_notes
from .sample import (MelodySampler, calibrate_chord_bias, chord_tone_rate,
                     condition_from_song)


class Ensemble:
    """Average the logits of several checkpoints (keeps the MelodySampler API)."""

    def __init__(self, models: list) -> None:
        self.models = models

    def to(self, device: str):
        for m in self.models:
            m.to(device).eval()
        return self

    def eval(self):
        for m in self.models:
            m.eval()
        return self

    def __call__(self, *args, **kwargs):
        return torch.stack([m(*args, **kwargs) for m in self.models], dim=0).mean(dim=0)


def _ref_tokens(song: D.Song, window: D.Window) -> np.ndarray:
    toks = np.full(len(window.indices), PAD, dtype=np.int64)
    for i, s in enumerate(window.indices):
        if song.layer[s] == 2:
            toks[i] = rel_to_token(int(song.melody_rel[s]))
    return toks


def _ref_abs(song: D.Song, window: D.Window) -> np.ndarray:
    out = np.full(len(window.indices), D.NONE, dtype=np.int64)
    for i, s in enumerate(window.indices):
        out[i] = song.melody_abs[s]
    return out


def _prime(cond, tokens: np.ndarray, n: int = 3) -> list[tuple[int, int]]:
    slots = [i for i in range(cond.L) if cond.layer[i] == 2 and tokens[i] != PAD]
    return [(i, token_to_rel(int(tokens[i]))) for i in slots[:n]]


@dataclass
class EvalConfig:
    cache: str = "data/cache/dataset.json"
    checkpoints: list[str] = field(default_factory=list)
    split_seed: int = 42
    test_frac: float = 0.11
    val_frac: float = 0.10
    modes: tuple[str, ...] = ("free", "motif", "plan")
    protocols: tuple[str, ...] = ("continuation", "free")
    seeds: tuple[int, ...] = (0, 1, 2)
    w_chords: tuple[float, ...] = (0.0, 2.0)
    w_template: float = 2.0
    w_repeat: float = 1.2
    w_leap: float = 1.0
    w_range: float = 0.3
    w_root: float = 0.0
    w_bass: float = 1.5
    calib_candidates: tuple[float, ...] = (0.0, 0.75, 1.5, 2.5, 4.0)
    prime_notes: int = 3
    calibrate: bool = True
    threads: int = 4
    target_mode: str = "auto"   # auto | pitch | delta
    out_dir: str = "reports"


def evaluate(cfg: EvalConfig) -> dict:
    torch.set_num_threads(cfg.threads)
    songs, summary, _ = D.load_cache(cfg.cache)
    by_id = {s.song_id: s for s in songs}
    split = D.split_songs([s.song_id for s in songs], seed=cfg.split_seed,
                          val_frac=cfg.val_frac, test_frac=cfg.test_frac)
    windows_by_song = {sid: D.build_windows(by_id[sid]) for sid in by_id}
    test_ids = split["test"]

    models = []
    for path in cfg.checkpoints:
        model, ckpt = load_checkpoint(path)
        models.append(model)
    sampler = MelodySampler(Ensemble(models) if len(models) > 1 else models[0])
    if cfg.target_mode != "auto":
        sampler.target_mode = cfg.target_mode
        sampler.n_out = N_DELTA_VOCAB if cfg.target_mode == "delta" else N_TOKENS

    # corpus reference on the same windows
    corpus_rows = []
    for sid in test_ids:
        song = by_id[sid]
        for win in windows_by_song[sid]:
            cond = condition_from_song(song, win)
            idx = win.indices
            mel = {i: song.melody_abs[idx[i]] for i in range(cond.L) if song.layer[idx[i]] == 2}
            ped = {i: song.pedal_abs[idx[i]] for i in range(cond.L)
                   if song.layer[idx[i]] == 1 and song.pedal_abs[idx[i]] != D.NONE}
            row = M.evaluate_tokens(cond, _ref_tokens(song, win), _ref_abs(song, win))
            row.update(M.melody_quality_stats(mel, ped, cond.n_bars))
            corpus_rows.append(row)
    corpus_agg = M.aggregate(corpus_rows)

    windows_list = [(sid, win) for sid in test_ids for win in windows_by_song[sid]]

    def _calibrate_global(mode: str, protocol: str) -> float:
        """Pick one chord-tone weight for the whole test set (aggregate calibration)."""
        step = max(1, len(windows_list) // 8)
        probe = windows_list[::step][:8]
        best = None
        for w in cfg.calib_candidates:
            tot = hit = 0
            for sid, win in probe:
                song = by_id[sid]
                cond = condition_from_song(song, win)
                prime = _prime(cond, _ref_tokens(song, win), cfg.prime_notes) \
                    if protocol == "continuation" else []
                toks = sampler.generate(cond, mode=mode, seed=0, prime=prime, w_chord=w,
                                        w_template=cfg.w_template, w_repeat=cfg.w_repeat,
                                        w_leap=cfg.w_leap, w_range=cfg.w_range,
                                        w_root=cfg.w_root, w_bass=cfg.w_bass)
                for i in range(cond.L):
                    if cond.layer[i] != 2 or toks[i] == PAD:
                        continue
                    tot += 1
                    if cond.is_chord_tone(i, token_to_rel(int(toks[i]))):
                        hit += 1
            rate = hit / max(1, tot)
            if 0.50 <= rate <= 0.90:
                return w
            score = abs(rate - 0.80)
            if best is None or score < best[0]:
                best = (score, w, rate)
        return best[1]

    weights: dict[tuple[str, str], float] = {}
    for mode in cfg.modes:
        for protocol in cfg.protocols:
            weights[(mode, protocol)] = _calibrate_global(mode, protocol) if cfg.calibrate \
                else cfg.w_chords[0]

    rows: list[dict] = []
    for protocol in cfg.protocols:
        for mode in cfg.modes:
            w = weights[(mode, protocol)]
            for sid, win in windows_list:
                song = by_id[sid]
                cond = condition_from_song(song, win)
                ref_t = _ref_tokens(song, win)
                ref_a = _ref_abs(song, win)
                prime = _prime(cond, ref_t, cfg.prime_notes) if protocol == "continuation" else []
                for seed in cfg.seeds:
                    toks = sampler.generate(cond, mode=mode, seed=seed, prime=prime,
                                            w_chord=w, w_template=cfg.w_template,
                                            w_repeat=cfg.w_repeat,
                                            w_leap=cfg.w_leap, w_range=cfg.w_range,
                                            w_root=cfg.w_root, w_bass=cfg.w_bass)
                    m = M.evaluate_tokens(cond, toks,
                                          ref_a if protocol == "continuation" else None)
                    mel = sampler.melody_abs(cond, toks)
                    ped = _pedal_notes(cond, 15, mel)
                    m.update(M.melody_quality_stats(mel, ped, cond.n_bars))
                    m.update({"song": sid, "mode": mode, "protocol": protocol,
                              "seed": seed, "w_chord": w,
                              "n_rot": win.rotation_bars})
                    rows.append(m)
    # grouped aggregates
    groups: dict[str, dict] = {}
    for key_fn, label in [
        (lambda r: f"{r['mode']}/{r['protocol']}", None),
        (lambda r: "all", "all"),
    ]:
        acc: dict[str, list[dict]] = {}
        for r in rows:
            acc.setdefault(key_fn(r), []).append(r)
        for k, rs in acc.items():
            groups[k] = M.aggregate(rs)

    out = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "split": {k: len(v) for k, v in split.items()},
        "n_test_windows": len(corpus_rows),
        "corpus": corpus_agg,
        "groups": groups,
        "rows": rows,
    }
    return out


def teacher_forced_summary(checkpoint_paths: list[str], cache: str, split_seed: int = 42) -> dict:
    from .train import TrainConfig, evaluate_loss, prepare
    cfg = TrainConfig(cache=cache, split_seed=split_seed)
    data = prepare(cfg)
    out = {}
    for path in checkpoint_paths:
        model, _ = load_checkpoint(path)
        test_loss, test_m = evaluate_loss(model, data["test"])
        val_loss, val_m = evaluate_loss(model, data["val"])
        out[os.path.basename(os.path.dirname(path)) or path] = {
            "test_loss": test_loss, "val_loss": val_loss,
            **{k: v for k, v in test_m.items() if k not in ("pred_rel", "true_rel")},
        }
    return out


# ---------------------------------------------------------------------------
def _fmt(v, digits: int = 3) -> str:
    if isinstance(v, float):
        if v != v:
            return "  –  "
        return f"{v:.{digits}f}"
    return str(v)


def write_markdown(result: dict, tf: dict, path: str, title: str = "Evaluation") -> None:
    lines = [f"# {title}", ""]
    lines.append(f"test songs: {result['split']['test']}  "
                 f"test windows: {result['n_test_windows']}")
    lines.append("")
    lines.append("## Corpus reference (same windows, ground truth)")
    lines.append("")
    c = result["corpus"]
    for k in ("in_key", "chord_tone", "motif_consistency", "distinct_pitches",
              "pitch_span", "mean_abs_interval", "jump_rate", "dup_rate",
              "quality_bar_pitch_range", "quality_corr_melody_bass", "quality_jump7_rate"):
        if k in c:
            lines.append(f"- {k}: {_fmt(c[k])}")
    lines += ["", "## Generated", "",
              "| mode/protocol | in_key | chord_tone | motif | interval_l1 | dist_l1 | contour_r | dup_rate | distinct | span | bar_rng | corr_bass | jump>7 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = [f"{m}/{p}" for p in ("continuation", "free") for m in ("free", "motif", "plan")]
    for key in order:
        if key not in result["groups"]:
            continue
        g = result["groups"][key]
        lines.append("| " + " | ".join([
            key, _fmt(g.get("in_key"), 3), _fmt(g.get("chord_tone"), 3),
            _fmt(g.get("motif_consistency"), 3), _fmt(g.get("interval_l1"), 3),
            _fmt(g.get("interval_dist_l1"), 3), _fmt(g.get("contour_r"), 3),
            _fmt(g.get("dup_rate"), 3),
            _fmt(g.get("distinct_pitches"), 1), _fmt(g.get("pitch_span"), 1),
            _fmt(g.get("quality_bar_pitch_range"), 2),
            _fmt(g.get("quality_corr_melody_bass"), 2),
            _fmt(g.get("quality_jump7_rate"), 3)]) + " |")
    if tf:
        lines += ["", "## Teacher-forced (test songs, no generation)", "",
                  "| checkpoint | val_loss | test_loss | token_acc | interval_l1 | contour_r |",
                  "|---|---|---|---|---|---|"]
        for name, v in tf.items():
            lines.append(f"| {name} | {_fmt(v['val_loss'],3)} | {_fmt(v['test_loss'],3)} | "
                         f"{_fmt(v.get('token_acc'),3)} | {_fmt(v.get('interval_l1'),3)} | "
                         f"{_fmt(v.get('contour_r'),3)} |")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
