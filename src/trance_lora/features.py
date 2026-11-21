"""Turn extracted songs + windows into fixed-size numpy feature arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import (DELTA_OFFSET, DELTA_PAD, N_TOKENS, PAD, PITCH_OFFSET,
                   SLOTS_PER_BAR, Song, Window, WINDOW_SLOTS, NONE)

IGNORE = -100  # cross-entropy ignore index


def rel_to_token(rel: int) -> int:
    if rel == NONE:
        return PAD
    return int(rel) + PITCH_OFFSET


def token_to_rel(tok: int) -> int:
    return int(tok) - PITCH_OFFSET


@dataclass
class WindowData:
    song_id: str
    rotation_bars: int
    layer: np.ndarray       # (L,) int64   0 rest / 1 pedal / 2 melody
    melody: np.ndarray      # (L,) int64   melody token, PAD elsewhere
    prev: np.ndarray        # (L,) int64   previous melody token, PAD at start
    chord: np.ndarray       # (L,12) float32  chord tone mask rel. tonic
    root: np.ndarray        # (L,) int64   chord root pc rel. tonic
    degree: np.ndarray      # (L,) int64   triad index 0..6
    barpos: np.ndarray      # (L,) int64
    slotpos: np.ndarray     # (L,) int64
    template: np.ndarray    # (L,) int64   raw motif template token (PAD where none)
    template_plan: np.ndarray  # (L,) int64  transposed motif plan token
    target: np.ndarray      # (L,) int64   melody token or IGNORE
    delta: np.ndarray       # (L,) int64   interval-from-previous token or IGNORE
    prev_delta: np.ndarray  # (L,) int64   previous interval token (DELTA_PAD at start)


def _chord_matrix(masks: list[int], n: int) -> np.ndarray:
    m = np.zeros((n, 12), dtype=np.float32)
    for i in range(n):
        mask = masks[i]
        for pc in range(12):
            if mask & (1 << pc):
                m[i, pc] = 1.0
    return m


def harmony_features(chord_masks_rel_tonic: list[int], root_pcs_rel_tonic: list[int],
                     anchor: int, tonic: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the harmony inputs in BOTH coordinate frames.

    The melody tokens are ``rel = pitch - anchor``, so a chord mask expressed
    relative to the *tonic* cannot be matched against them unless the model also
    knows ``anchor - tonic``.  We therefore give the model
      * ``chord[:, 0:12]``  chord tones relative to the tonic (harmonic function),
      * ``chord[:, 12:24]`` chord tones relative to the anchor (directly comparable
        with the melody tokens),
      * ``root``            chord root pitch class relative to the anchor.
    """
    n = len(chord_masks_rel_tonic)
    chord = np.zeros((n, 24), dtype=np.float32)
    root = np.zeros(n, dtype=np.int64)
    for i in range(n):
        mask = chord_masks_rel_tonic[i]
        for pc in range(12):
            if mask & (1 << pc):
                chord[i, pc] = 1.0
                chord[i, 12 + (pc + tonic - anchor) % 12] = 1.0
        root[i] = (tonic + root_pcs_rel_tonic[i] - anchor) % 12
    return chord, root


def make_template(melody_tok: np.ndarray, bars_from: int = 1) -> np.ndarray:
    """Motif template = bar-0 melody copied to the same bar position of later bars."""
    L = len(melody_tok)
    motif = melody_tok[:SLOTS_PER_BAR].copy()
    tmpl = np.full(L, PAD, dtype=np.int64)
    for i in range(bars_from * SLOTS_PER_BAR, L):
        tmpl[i] = motif[i % SLOTS_PER_BAR]
    return tmpl


def window_features(song: Song, win: Window, template_override: np.ndarray | None = None) -> WindowData:
    idx = win.indices
    L = len(idx)
    layer = np.asarray([song.layer[i] for i in idx], dtype=np.int64)
    melody = np.asarray([rel_to_token(song.melody_rel[i]) for i in idx], dtype=np.int64)
    prev = np.full(L, PAD, dtype=np.int64)
    last = PAD
    for i in range(L):
        prev[i] = last
        if layer[i] == 2:
            last = melody[i]
    chord_masks = [song.chord_pcs[int(i)] for i in idx]
    roots_rel_tonic = [song.chord_root_pc[i] for i in idx]
    chord, root = harmony_features(chord_masks, roots_rel_tonic, song.anchor, song.tonic)
    degree = np.asarray([song.chord_degree[i] for i in idx], dtype=np.int64)
    barpos = np.arange(L, dtype=np.int64) % SLOTS_PER_BAR
    slotpos = np.arange(L, dtype=np.int64)
    n_bars = L // SLOTS_PER_BAR
    motif = melody[:SLOTS_PER_BAR]
    from . import plans
    template, template_plan, _steps = plans.build_templates(
        layer, chord_masks, song.mode, n_bars, song.anchor, song.tonic, motif)
    if template_override is not None:
        template = np.asarray(template_override, dtype=np.int64)
    target = np.where(layer == 2, melody, IGNORE).astype(np.int64)

    # interval (delta) targets: delta_i = rel_i - rel_{i-1}; the first melody note
    # of the window uses 0 (i.e. the anchor) as its reference.
    delta = np.full(L, IGNORE, dtype=np.int64)
    prev_delta = np.full(L, DELTA_PAD, dtype=np.int64)
    last_rel = 0
    last_delta = DELTA_PAD
    for i in range(L):
        if layer[i] == 2:
            rel = token_to_rel(int(melody[i]))
            prev_delta[i] = last_delta
            d = rel - last_rel
            delta[i] = d + DELTA_OFFSET
            last_delta = int(delta[i])
            last_rel = rel

    return WindowData(song.song_id, win.rotation_bars, layer, melody, prev, chord, root,
                      degree, barpos, slotpos, template, template_plan, target, delta,
                      prev_delta)


def _stack(wds: list[WindowData]) -> dict[str, np.ndarray]:
    return {
        "layer": np.stack([w.layer for w in wds]),
        "melody": np.stack([w.melody for w in wds]),
        "prev": np.stack([w.prev for w in wds]),
        "chord": np.stack([w.chord for w in wds]),
        "root": np.stack([w.root for w in wds]),
        "degree": np.stack([w.degree for w in wds]),
        "barpos": np.stack([w.barpos for w in wds]),
        "slotpos": np.stack([w.slotpos for w in wds]),
        "template": np.stack([w.template for w in wds]),
        "template_plan": np.stack([w.template_plan for w in wds]),
        "target": np.stack([w.target for w in wds]),
        "delta": np.stack([w.delta for w in wds]),
        "prev_delta": np.stack([w.prev_delta for w in wds]),
    }


def build_arrays(songs: list[Song], windows_by_song: dict[str, list[Window]],
                 song_ids: list[str] | None = None) -> dict[str, np.ndarray]:
    wds: list[WindowData] = []
    ids = song_ids if song_ids is not None else [s.song_id for s in songs]
    by_id = {s.song_id: s for s in songs}
    for sid in ids:
        song = by_id[sid]
        for win in windows_by_song[sid]:
            wds.append(window_features(song, win))
    if not wds:
        raise ValueError("empty window set")
    return _stack(wds)


def drop_template(batch: dict[str, np.ndarray], prob: float, rng: np.random.Generator) -> np.ndarray:
    """Return a template array with whole-window dropout applied."""
    tmpl = batch["template"].copy()
    if prob > 0:
        mask = rng.random(len(tmpl)) < prob
        tmpl[mask] = PAD
    return tmpl


__all__ = ["WindowData", "window_features", "build_arrays", "make_template",
           "rel_to_token", "token_to_rel", "IGNORE", "N_TOKENS"]
