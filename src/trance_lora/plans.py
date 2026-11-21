"""Motif development plan: in-scale transposition of the bar-0 motif, per bar.

The round-2 spec says: the motif should be transposed *within the scale* so that
its first note lands on a chord tone of the bar it is being repeated in, and then
varied.  This module is the single source of truth for that plan, shared by
training (plan-template channel), inference and analysis.
"""

from __future__ import annotations

import numpy as np

from . import theory
from .data import PAD, SLOTS_PER_BAR

# variation schedule from the round-2 spec: how many motif notes get rewritten
VARIATION_SCHEDULE = (0, 0, 1, 1, 2, 2, 3, 3)


def _chord_pcs(mask: int) -> set[int]:
    return {pc for pc in range(12) if mask & (1 << pc)}


def bar_first_melody_slot(layer: np.ndarray, b: int) -> int:
    sl = layer[b * SLOTS_PER_BAR:(b + 1) * SLOTS_PER_BAR]
    idx = np.where(sl == 2)[0]
    return b * SLOTS_PER_BAR + (int(idx[0]) if len(idx) else 0)


def choose_steps(layer: np.ndarray, chord_masks: list[int], motif_rels: list[int],
                 anchor: int, tonic: int, mode: str, n_bars: int,
                 max_steps: int = 7) -> list[int]:
    """Per-bar in-scale transposition steps (0 = untransposed)."""
    if not motif_rels:
        return [0] * n_bars
    first_rel = motif_rels[0]
    steps: list[int] = []
    for b in range(n_bars):
        i = bar_first_melody_slot(layer, b)
        pcs = _chord_pcs(chord_masks[i]) or set(theory.SCALES[mode])
        best, best_cost = 0, float("inf")
        for s in range(-max_steps, max_steps + 1):
            t = theory.transpose_rel(first_rel, s, anchor, tonic, mode)
            if t is None:
                continue
            pc = (anchor + t - tonic) % 12
            dist = min(min((pc - c) % 12, (c - pc) % 12) for c in pcs)
            cost = dist * 10 + abs(s) * 0.01
            if cost < best_cost:
                best, best_cost = s, cost
        steps.append(best)
    return steps


def transpose_motif_tokens(motif_tokens: np.ndarray, steps: int, mode: str,
                           anchor: int, tonic: int) -> np.ndarray:
    """Transpose a 16-slot motif token stream by ``steps`` scale steps.

    Transposed notes outside the model's rel-pitch window are folded by octaves
    (pitch class preserved) instead of dropped, so the template stays complete.
    """
    out = np.full_like(motif_tokens, PAD)
    for r in range(len(motif_tokens)):
        tok = int(motif_tokens[r])
        if tok == PAD:
            continue
        from .features import token_to_rel, rel_to_token
        rel = token_to_rel(tok)
        t = theory.transpose_rel(rel, steps, anchor, tonic, mode)
        if t is None:
            continue
        while t > 18:
            t -= 12
        while t < -18:
            t += 12
        out[r] = rel_to_token(t)
    return out


def raw_template(motif_tokens: np.ndarray, n_bars: int) -> np.ndarray:
    """Raw motif copied onto the same bar position of every later bar."""
    motif = np.asarray(motif_tokens[:SLOTS_PER_BAR])
    L = n_bars * SLOTS_PER_BAR
    tmpl = np.full(L, PAD, dtype=np.int64)
    for i in range(SLOTS_PER_BAR, L):
        tmpl[i] = motif[i % SLOTS_PER_BAR]
    return tmpl


def plan_template(motif_tokens: np.ndarray, steps: list[int], mode: str,
                  n_bars: int, anchor: int, tonic: int) -> np.ndarray:
    """Per-bar transposed motif, used as the 'plan' conditioning stream."""
    L = n_bars * SLOTS_PER_BAR
    tmpl = np.full(L, PAD, dtype=np.int64)
    for b in range(n_bars):
        bar = transpose_motif_tokens(np.asarray(motif_tokens[:SLOTS_PER_BAR]),
                                     steps[b] if b < len(steps) else 0, mode,
                                     anchor, tonic)
        tmpl[b * SLOTS_PER_BAR:(b + 1) * SLOTS_PER_BAR] = bar
    return tmpl


def triad_degree(chord_mask: int, mode: str) -> int | None:
    """Index (0..6) of the diatonic triad whose pitch-class set equals ``mask``."""
    pcs = _chord_pcs(chord_mask)
    for deg, (label, root, tri) in enumerate(theory.TRIADS[mode]):
        if pcs == set(tri):
            return deg
    # tolerate one foreign tone
    for deg, (label, root, tri) in enumerate(theory.TRIADS[mode]):
        if len(pcs & set(tri)) >= 2 and not (pcs - set(tri)) - set(tri):
            if len(pcs - set(tri)) <= 1:
                return deg
    return None


def choose_steps_by_root(chord_masks: list[int], mode: str, n_bars: int) -> list[int]:
    """Transpose the motif with the chord-root movement (i -> VI == 5 scale steps)."""
    degs = [triad_degree(chord_masks[b * SLOTS_PER_BAR], mode) for b in range(n_bars)]
    if not degs or degs[0] is None:
        return [0] * n_bars
    base = degs[0]
    steps = []
    for b in range(n_bars):
        d = degs[b]
        steps.append(0 if d is None else int(d) - int(base))
    # keep the plan inside a singable range
    return [max(-7, min(7, s)) for s in steps]


def build_templates(layer: np.ndarray, chord_masks: list[int], mode: str,
                    n_bars: int, anchor: int, tonic: int,
                    motif_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (raw_template, plan_template, steps) for one window."""
    motif_rels = []
    from .features import token_to_rel
    for t in motif_tokens[:SLOTS_PER_BAR]:
        if int(t) != PAD:
            motif_rels.append(token_to_rel(int(t)))
    steps = choose_steps(layer, chord_masks, motif_rels, anchor, tonic, mode, n_bars)
    return (raw_template(motif_tokens, n_bars),
            plan_template(motif_tokens, steps, mode, n_bars, anchor, tonic), steps)


def variation_counts(n_bars: int) -> list[int]:
    """How many motif notes to rewrite per bar (spec: 0,0,1,1,2,2,3,3)."""
    return [VARIATION_SCHEDULE[b % len(VARIATION_SCHEDULE)] for b in range(n_bars)]
