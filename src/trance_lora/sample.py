"""Constrained decoding: chord-conditioned melody sampling with in-key masking,
chord-tone weighting, motif-template conditioning and octave folding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import theory
from .data import (DELTA_MAX, DELTA_MIN, DELTA_OFFSET, DELTA_PAD, N_DELTA_VOCAB,
                   N_PITCH_TOKENS, N_TOKENS, PAD, PITCH_OFFSET, REL_MAX, REL_MIN,
                   SLOTS_PER_BAR, Song, Window, WINDOW_SLOTS, NONE)
from .features import rel_to_token, token_to_rel
from .model import TranceMelodyModel

NEG_INF = -1e9

# rel semitone values that are in-key (given that anchor itself is in key)
SCALE_RELS = {s + 12 * k for s in range(12) for k in (-2, -1, 0, 1)}


@dataclass
class Condition:
    layer: np.ndarray        # (L,) 0/1/2
    chord: np.ndarray        # (L,12) float32 chord-tone mask rel. tonic
    root: np.ndarray         # (L,) chord root pc rel. tonic
    degree: np.ndarray       # (L,)
    barpos: np.ndarray       # (L,)
    slotpos: np.ndarray      # (L,)
    anchor: int
    tonic: int
    mode: str
    n_bars: int = 8
    bpm: int = 140
    progression: list[str] | None = None   # per-bar labels
    rhythm_pattern: str = ""

    @property
    def L(self) -> int:
        return len(self.layer)

    def chord_tone_rels(self, i: int) -> set[int]:
        """Relative semitones (rel. anchor) that are chord tones at slot ``i``."""
        abs_mask = self.chord[i]
        rels = set()
        for rel in range(-24, 25):
            pc_rel_tonic = (self.anchor + rel - self.tonic) % 12
            if abs_mask[pc_rel_tonic] > 0.5:
                rels.add(rel)
        return rels

    def is_chord_tone(self, i: int, rel: int) -> bool:
        return self.chord[i, (self.anchor + rel - self.tonic) % 12] > 0.5

    def harmony(self) -> tuple[np.ndarray, np.ndarray]:
        """(24-dim chord matrix, root pc relative to the anchor) for the model."""
        cached = getattr(self, "_harmony", None)
        if cached is None:
            from .features import harmony_features
            masks = []
            for i in range(self.L):
                m = 0
                for pc in range(12):
                    if self.chord[i, pc] > 0.5:
                        m |= 1 << pc
                masks.append(m)
            cached = harmony_features(masks, [int(r) for r in self.root],
                                      self.anchor, self.tonic)
            self._harmony = cached
        return cached


def condition_from_song(song: Song, window: Window) -> Condition:
    idx = window.indices
    L = len(idx)
    chord = np.zeros((L, 12), dtype=np.float32)
    for i, s in enumerate(idx):
        mask = song.chord_pcs[s]
        for pc in range(12):
            if mask & (1 << pc):
                chord[i, pc] = 1.0
    return Condition(
        layer=np.asarray([song.layer[s] for s in idx], dtype=np.int64),
        chord=chord,
        root=np.asarray([song.chord_root_pc[s] for s in idx], dtype=np.int64),
        degree=np.asarray([song.chord_degree[s] for s in idx], dtype=np.int64),
        barpos=np.arange(L) % SLOTS_PER_BAR,
        slotpos=np.arange(L),
        anchor=song.anchor, tonic=song.tonic, mode=song.mode,
        n_bars=L // SLOTS_PER_BAR, bpm=song.bpm,
        progression=[song.chord_labels[b] for b in range(L // SLOTS_PER_BAR)],
        rhythm_pattern=song.rhythm_pattern,
    )


def condition_from_spec(tonic: int, mode: str, bpm: int, progression: list[str],
                        rhythm: list[int], anchor: int | None = None) -> Condition:
    n_bars = len(rhythm) // SLOTS_PER_BAR
    if anchor is None:
        anchor = theory.nearest_in_key(64, tonic, mode)
    chord = np.zeros((len(rhythm), 12), dtype=np.float32)
    root = np.zeros(len(rhythm), dtype=np.int64)
    degree = np.zeros(len(rhythm), dtype=np.int64)
    for b in range(n_bars):
        label = progression[b]
        r, pcs = theory.triad_by_label(label, mode)
        deg = theory.degree_of_triad(label, mode)
        for s in range(b * SLOTS_PER_BAR, (b + 1) * SLOTS_PER_BAR):
            root[s] = r
            degree[s] = deg
            for pc in pcs:
                chord[s, pc] = 1.0
    L = len(rhythm)
    return Condition(
        layer=np.asarray(rhythm, dtype=np.int64), chord=chord, root=root, degree=degree,
        barpos=np.arange(L) % SLOTS_PER_BAR, slotpos=np.arange(L),
        anchor=anchor, tonic=tonic, mode=mode, n_bars=n_bars, bpm=bpm,
        progression=list(progression), rhythm_pattern="".join(str(x) for x in rhythm[:16]),
    )


# ---------------------------------------------------------------------------

def _prime_rel(cond: Condition, rel: int) -> int:
    """Clamp a primed rel into [-8, 18] AND onto a diatonic degree of the key."""
    from . import theory

    lo, hi = -8, 18
    rel = max(lo, min(hi, int(rel)))
    for off in (0, 1, -1, 2, -2, 3, -3):
        r = rel + off
        if r < lo or r > hi:
            continue
        if (cond.anchor + r - cond.tonic) % 12 in theory.SCALES[cond.mode]:
            return r
    return rel


class MelodySampler:
    """Autoregressive sampler over the 128-slot grid (causal model)."""

    def __init__(self, model: TranceMelodyModel, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device
        self._allowed_cache: dict[tuple[str, int], np.ndarray] = {}
        cfg = getattr(model, "config", None)
        if cfg is None and hasattr(model, "models"):
            cfg = getattr(model.models[0], "config", None)
        cfg = cfg or {}
        self.target_mode = cfg.get("target_mode", "pitch")
        self.use_prev_delta = cfg.get("use_prev_delta", False)
        self.n_out = N_DELTA_VOCAB if self.target_mode == "delta" else N_TOKENS

    def _allowed_for(self, cond: Condition) -> np.ndarray:
        """In-key tokens, kept above the pedal band (>= -8 semitones from the anchor)."""
        key = (cond.mode, (cond.anchor - cond.tonic) % 12)
        if key not in self._allowed_cache:
            scale = set(theory.SCALES[cond.mode])
            mask = np.zeros(N_PITCH_TOKENS, dtype=bool)
            for tok in range(N_PITCH_TOKENS):
                rel = token_to_rel(tok)
                if rel < -8:
                    continue
                if (cond.anchor + rel - cond.tonic) % 12 in scale:
                    mask[tok] = True
            self._allowed_cache[key] = mask
        return self._allowed_cache[key]

    # -- tensor helpers -------------------------------------------------
    def _tensors(self, cond: Condition, prev: np.ndarray, template: np.ndarray,
                 prev_delta: np.ndarray | None = None) -> dict[str, torch.Tensor]:
        chord, root = cond.harmony()
        t = {
            "prev": torch.from_numpy(prev[None]).long().to(self.device),
            "chord": torch.from_numpy(chord[None]).float().to(self.device),
            "layer": torch.from_numpy(cond.layer[None]).long().to(self.device),
            "root": torch.from_numpy(root[None]).long().to(self.device),
            "degree": torch.from_numpy(cond.degree[None]).long().to(self.device),
            "barpos": torch.from_numpy(cond.barpos[None]).long().to(self.device),
            "slotpos": torch.from_numpy(cond.slotpos[None]).long().to(self.device),
            "template": torch.from_numpy(template[None]).long().to(self.device),
        }
        if self.use_prev_delta:
            if prev_delta is None:
                prev_delta = np.full(len(prev), DELTA_PAD, dtype=np.int64)
            t["prev_delta"] = torch.from_numpy(prev_delta[None]).long().to(self.device)
        return t

    @torch.no_grad()
    def _logits_at(self, cond: Condition, prev: np.ndarray, template: np.ndarray,
                   i: int, prev_delta: np.ndarray | None = None) -> np.ndarray:
        t = self._tensors(cond, prev, template, prev_delta)
        n = i + 1
        kw = {}
        if self.use_prev_delta:
            kw["prev_delta"] = t["prev_delta"][:, :n]
        logits = self.model(
            prev=t["prev"][:, :n], chord=t["chord"][:, :n], layer=t["layer"][:, :n],
            root=t["root"][:, :n], degree=t["degree"][:, :n],
            barpos=t["barpos"][:, :n], slotpos=t["slotpos"][:, :n],
            template=t["template"][:, :n], **kw,
        )
        return logits[0, i].cpu().numpy()

    # -- bias -----------------------------------------------------------
    def _bias(self, cond: Condition, i: int, template: np.ndarray,
              w_chord: float, w_template: float, prev_tok: int = PAD,
              w_leap: float = 0.0, w_range: float = 0.0, w_root: float = 0.0,
              target_reg: float | None = None, w_bass: float = 0.0,
              leap_free: int = 5, range_free: int = 8) -> np.ndarray:
        from .data import N_TOKENS
        bias = np.full(N_TOKENS, NEG_INF, dtype=np.float64)
        allowed = self._allowed_for(cond)
        bias[:N_PITCH_TOKENS][allowed] = 0.0
        prev_rel = None if prev_tok == PAD else token_to_rel(int(prev_tok))
        for tok in range(N_PITCH_TOKENS):
            if bias[tok] <= NEG_INF / 2:
                continue
            rel = token_to_rel(tok)
            if w_leap and prev_rel is not None:
                jump = abs(rel - prev_rel)
                if jump > leap_free:
                    bias[tok] -= w_leap * (jump - leap_free)
            if w_range:
                drift = abs(rel) - range_free
                if drift > 0:
                    bias[tok] -= w_range * drift
            if w_bass and target_reg is not None:
                pitch = cond.anchor + rel
                bias[tok] += w_bass * np.exp(-((pitch - target_reg) ** 2) / 18.0)
        if w_chord:
            for rel in cond.chord_tone_rels(i):
                tok = rel_to_token(rel)
                if 0 <= tok < N_PITCH_TOKENS and bias[tok] > NEG_INF / 2:
                    bias[tok] += w_chord
        if w_root:
            root_a = int(cond.harmony()[1][i])
            for tok in range(N_PITCH_TOKENS):
                if bias[tok] > NEG_INF / 2 and token_to_rel(tok) % 12 == root_a:
                    bias[tok] += w_root
        if w_template and template[i] != PAD:
            t_rel = token_to_rel(int(template[i]))
            for tok in range(N_PITCH_TOKENS):
                if bias[tok] <= NEG_INF / 2:
                    continue
                rel = token_to_rel(tok)
                d = abs(rel - t_rel)
                if d == 0:
                    bias[tok] += w_template
                elif d == 1:
                    bias[tok] += 0.65 * w_template
                elif d == 2:
                    bias[tok] += 0.35 * w_template
        return bias

    # -- motif plan -----------------------------------------------------
    def build_plan_template(self, cond: Condition, motif_tokens: np.ndarray,
                            variant: str = "chordtone",
                            bass_line: list[float] | None = None) -> np.ndarray:
        """Per-bar transposed motif plan (same rules as training, see plans.py).

        ``variant``: ``chordtone`` (spec rule: transposed first note hits a chord
        tone) or ``root`` (follow the chord-root movement, i -> VI == 5 steps).
        """
        from . import plans
        motif_rels = [token_to_rel(int(t)) for t in motif_tokens[:SLOTS_PER_BAR] if int(t) != PAD]
        chord_masks = []
        for i in range(cond.L):
            mask = 0
            for pc in range(12):
                if cond.chord[i, pc] > 0.5:
                    mask |= 1 << pc
            chord_masks.append(mask)
        if variant == "root":
            steps = plans.choose_steps_by_root(chord_masks, cond.mode, cond.n_bars)
            if bass_line and motif_rels:
                first = motif_rels[0]
                for b in range(cond.n_bars):
                    target = first + (bass_line[b] - bass_line[0])
                    best = None
                    for k in (-1, 0, 1):
                        cand = steps[b] + 7 * k
                        t = theory.transpose_rel(first, cand, cond.anchor, cond.tonic, cond.mode)
                        if t is None:
                            continue
                        d = abs(t - target)
                        if best is None or d < best[0]:
                            best = (d, cand)
                    if best is not None:
                        steps[b] = best[1]
        else:
            steps = plans.choose_steps(cond.layer, chord_masks, motif_rels, cond.anchor,
                                       cond.tonic, cond.mode, cond.n_bars)
        return plans.plan_template(motif_tokens, steps, cond.mode, cond.n_bars,
                                   cond.anchor, cond.tonic)

    # -- generation -----------------------------------------------------
    def generate(self, cond: Condition, *, mode: str = "free", greedy: bool = False,
                 temperature: float = 1.0, w_chord: float = 0.0, w_template: float = 0.0,
                 w_repeat: float = 0.0, w_leap: float = 0.0, w_range: float = 0.0,
                 w_root: float = 0.0, w_bass: float = 0.0, top_k: int = 0, seed: int = 0,
                 prime: list[tuple[int, int]] | None = None,
                 template_tokens: np.ndarray | None = None) -> np.ndarray:
        """Return melody tokens (PAD at non-melody slots).

        ``mode``: ``free`` (no template), ``motif`` (bar-0 motif fed back raw),
        ``plan`` (bar-0 motif transposed per bar by the harmonic plan),
        ``plan_root`` (transposed by the chord-root movement).
        ``w_bass`` pulls each bar's register toward the bass/pedal contour.
        ``prime``: list of (slot, rel_pitch) forced as already generated.
        """
        if getattr(self, "target_mode", "pitch") == "delta":
            return self._generate_delta(
                cond, mode=mode, greedy=greedy, temperature=temperature,
                w_chord=w_chord, w_template=w_template, w_repeat=w_repeat,
                w_leap=w_leap, w_range=w_range, w_root=w_root, w_bass=w_bass,
                top_k=top_k, seed=seed, prime=prime, template_tokens=template_tokens)

        rng = np.random.default_rng(seed)
        L = cond.L
        prev = np.full(L, PAD, dtype=np.int64)
        template = np.full(L, PAD, dtype=np.int64)
        out = np.full(L, PAD, dtype=np.int64)
        primed = dict(prime or [])
        target_reg = None
        bass_line = self._bass_register_line(cond) if (w_bass or mode == "plan_root") else None

        melody_slots = [i for i in range(L) if cond.layer[i] == 2]
        if not melody_slots:
            return out

        first_bar = [i for i in melody_slots if i < SLOTS_PER_BAR]
        template_ready = mode in ("free",)
        motif_modes = ("motif", "plan", "plan_root")

        if mode in motif_modes and template_tokens is not None:
            # caller already supplied the motif (e.g. harness / evaluation)
            template = self._template_from_motif(cond, template_tokens, mode, bass_line)
            template_ready = True

        def run_slots(slots: list[int]) -> None:
            nonlocal template, template_ready
            for i in slots:
                if i in primed:
                    # primed cells come from a transposed motif and can fall outside
                    # the melody register; clamp so they never collide with the pedal
                    tok = rel_to_token(_prime_rel(cond, primed[i]))
                else:
                    logits = self._logits_at(cond, prev, template, i)
                    treg = None if target_reg is None else \
                        target_reg[min(cond.n_bars - 1, i // SLOTS_PER_BAR)]
                    kw = dict(prev_tok=int(prev[i]) if i > 0 else PAD,
                              w_leap=w_leap, w_range=w_range, w_root=w_root,
                              target_reg=treg, w_bass=w_bass)
                    if not template_ready:
                        bias = self._bias(cond, i, template, w_chord, 0.0, **kw)
                    else:
                        bias = self._bias(cond, i, template, w_chord, w_template, **kw)
                    scores = logits + bias
                    if w_repeat and prev[i] != PAD:
                        scores[int(prev[i])] -= w_repeat
                    tok = int(np.argmax(scores)) if greedy else self._sample(scores, temperature, top_k, rng)
                out[i] = tok
                # prev[j] = last melody note strictly before j; it stays valid until
                # the next melody slot overwrites the tail again.
                prev[i + 1:] = tok

        if bass_line is not None and mode in motif_modes:
            bar0 = [token_to_rel(int(out[i])) for i in first_bar if out[i] != PAD]
            base = (cond.anchor + float(np.mean(bar0))) if bar0 else float(cond.anchor)
            target_reg = [base + (bass_line[b] - bass_line[0]) for b in range(cond.n_bars)]

        if mode in motif_modes and template_tokens is None:
            # stage 1: melody of bar 0 with no template
            run_slots(first_bar)
            motif = out[:SLOTS_PER_BAR].copy()
            if bass_line is not None:
                bar0 = [token_to_rel(int(motif[i])) for i in first_bar if motif[i] != PAD]
                base = (cond.anchor + float(np.mean(bar0))) if bar0 else float(cond.anchor)
                target_reg = [base + (bass_line[b] - bass_line[0]) for b in range(cond.n_bars)]
            template = self._template_from_motif(cond, motif, mode, bass_line)
            template_ready = True
            run_slots([i for i in melody_slots if i >= SLOTS_PER_BAR])
        else:
            run_slots(melody_slots)

        return self.fold_octaves(cond, out)

    # -- Step A: interval-target generation -----------------------------
    def _bias_delta(self, cond: Condition, i: int, template: np.ndarray,
                    prev_rel: float, w_chord: float, w_template: float,
                    w_repeat: float, w_leap: float, w_range: float, w_root: float,
                    target_reg: float | None, w_bass: float,
                    leap_free: int = 5, range_free: int = 8) -> np.ndarray:
        """Logit bias over the *interval* vocabulary (Step A models)."""
        bias = np.full(N_DELTA_VOCAB, NEG_INF, dtype=np.float64)
        allowed = self._allowed_for(cond)
        chord_rels = cond.chord_tone_rels(i) if w_chord else set()
        root_a = int(cond.harmony()[1][i]) if w_root else None
        t_rel = None
        if w_template and template[i] != PAD:
            t_rel = token_to_rel(int(template[i]))
        for d in range(DELTA_MIN, DELTA_MAX + 1):
            rel = prev_rel + d
            tok = rel + PITCH_OFFSET
            if tok < 0 or tok >= N_PITCH_TOKENS or not allowed[tok]:
                continue
            score = 0.0
            if w_leap and abs(d) > leap_free:
                score -= w_leap * (abs(d) - leap_free)
            if w_range:
                drift = abs(rel) - range_free
                if drift > 0:
                    score -= w_range * drift
            if chord_rels and rel in chord_rels:
                score += w_chord
            if root_a is not None and int(rel) % 12 == root_a:
                score += w_root
            if t_rel is not None:
                dd = abs(rel - t_rel)
                if dd == 0:
                    score += w_template
                elif dd == 1:
                    score += 0.65 * w_template
                elif dd == 2:
                    score += 0.35 * w_template
            if w_bass and target_reg is not None:
                score += w_bass * np.exp(-((cond.anchor + rel - target_reg) ** 2) / 18.0)
            if w_repeat and d == 0:
                score -= w_repeat
            bias[d + DELTA_OFFSET] = score
        return bias

    def _generate_delta(self, cond: Condition, *, mode: str = "free", greedy: bool = False,
                        temperature: float = 1.0, w_chord: float = 0.0, w_template: float = 0.0,
                        w_repeat: float = 0.0, w_leap: float = 0.0, w_range: float = 0.0,
                        w_root: float = 0.0, w_bass: float = 0.0, top_k: int = 0, seed: int = 0,
                        prime: list[tuple[int, int]] | None = None,
                        template_tokens: np.ndarray | None = None) -> np.ndarray:
        """Same contract as generate(), but the network emits intervals.

        Returns absolute relative-pitch tokens (PAD at non-melody slots), so every
        downstream metric/decoder keeps working unchanged.
        """
        rng = np.random.default_rng(seed)
        L = cond.L
        prev = np.full(L, PAD, dtype=np.int64)          # absolute rel token
        prev_delta = np.full(L, DELTA_PAD, dtype=np.int64)
        template = np.full(L, PAD, dtype=np.int64)
        out = np.full(L, PAD, dtype=np.int64)           # absolute rel token output
        primed = dict(prime or [])
        target_reg = None
        bass_line = self._bass_register_line(cond) if (w_bass or mode == "plan_root") else None

        melody_slots = [i for i in range(L) if cond.layer[i] == 2]
        if not melody_slots:
            return out
        first_bar = [i for i in melody_slots if i < SLOTS_PER_BAR]
        template_ready = mode in ("free",)
        motif_modes = ("motif", "plan", "plan_root")

        if mode in motif_modes and template_tokens is not None:
            template = self._template_from_motif(cond, template_tokens, mode, bass_line)
            template_ready = True

        def _apply(i: int, rel: int, delta_tok: int) -> None:
            out[i] = rel_to_token(int(rel))
            prev[i + 1:] = out[i]
            prev_delta[i + 1:] = delta_tok

        def run_slots(slots: list[int]) -> None:
            nonlocal template
            for i in slots:
                if i in primed:
                    rel = _prime_rel(cond, primed[i])
                    prev_rel = 0 if prev[i] == PAD else token_to_rel(int(prev[i]))
                    _apply(i, rel, rel - prev_rel + DELTA_OFFSET)
                    continue
                logits = self._logits_at(cond, prev, template, i, prev_delta)
                prev_rel = 0 if prev[i] == PAD else token_to_rel(int(prev[i]))
                treg = None if target_reg is None else \
                    target_reg[min(cond.n_bars - 1, i // SLOTS_PER_BAR)]
                bias = self._bias_delta(
                    cond, i, template, prev_rel, w_chord,
                    w_template if template_ready else 0.0, w_repeat, w_leap, w_range,
                    w_root, treg, w_bass)
                scores = logits + bias
                tok = int(np.argmax(scores)) if greedy else self._sample(scores, temperature, top_k, rng)
                d = tok - DELTA_OFFSET
                rel = int(np.clip(prev_rel + d, REL_MIN, REL_MAX))
                _apply(i, rel, rel - prev_rel + DELTA_OFFSET)

        if bass_line is not None and mode in motif_modes:
            target_reg = [float(cond.anchor)] * cond.n_bars
        if mode in motif_modes and template_tokens is None:
            run_slots(first_bar)
            motif = out[:SLOTS_PER_BAR].copy()
            if bass_line is not None:
                bar0 = [token_to_rel(int(motif[i])) for i in first_bar if motif[i] != PAD]
                base = (cond.anchor + float(np.mean(bar0))) if bar0 else float(cond.anchor)
                target_reg = [base + (bass_line[b] - bass_line[0]) for b in range(cond.n_bars)]
            template = self._template_from_motif(cond, motif, mode, bass_line)
            template_ready = True
            run_slots([i for i in melody_slots if i >= SLOTS_PER_BAR])
        else:
            run_slots(melody_slots)

        return self.fold_octaves(cond, out)

    @staticmethod
    def _template_raw(cond: Condition, motif: np.ndarray) -> np.ndarray:
        L = cond.L
        tmpl = np.full(L, PAD, dtype=np.int64)
        for i in range(SLOTS_PER_BAR, L):
            tok = int(motif[i % SLOTS_PER_BAR])
            if tok != PAD:
                tmpl[i] = tok
        return tmpl

    def _template_from_motif(self, cond: Condition, motif: np.ndarray, mode: str,
                             bass_line: list[float] | None = None) -> np.ndarray:
        if mode == "motif":
            return self._template_raw(cond, motif)
        return self.build_plan_template(cond, motif,
                                        variant="root" if mode == "plan_root" else "chordtone",
                                        bass_line=bass_line)

    def _sample(self, scores: np.ndarray, temperature: float, top_k: int,
                rng: np.random.Generator) -> int:
        x = scores / max(1e-6, temperature)
        finite = x > NEG_INF / 2
        if not finite.any():
            return PAD
        x = x - x[finite].max()
        p = np.zeros_like(x)
        p[finite] = np.exp(x[finite])
        if top_k and top_k < finite.sum():
            order = np.argsort(-p)
            keep = order[:top_k]
            mask = np.zeros_like(p, dtype=bool)
            mask[keep] = True
            p = np.where(mask, p, 0.0)
        p = p / p.sum()
        return int(rng.choice(len(p), p=p))

    # -- bass line ------------------------------------------------------
    @staticmethod
    def _bass_register_line(cond: Condition, offset: int = 15) -> list[float]:
        """Per-bar bass (pedal) mean register from the chord roots."""
        out = []
        target = cond.anchor - offset
        lo, hi = cond.anchor - 22, cond.anchor - 12   # pedal band, always below the melody
        for b in range(cond.n_bars):
            i = b * SLOTS_PER_BAR
            root_pc = (cond.tonic + int(cond.root[i])) % 12
            cands = [p for p in range(0, 128) if p % 12 == root_pc and lo <= p <= hi]
            if not cands:
                cands = [p for p in range(0, 128) if p % 12 == root_pc and p <= cond.anchor - 12] \
                    or [p for p in range(0, 128) if p % 12 == root_pc]
            out.append(float(min(cands, key=lambda p: (abs(p - target), p))))
        return out

    # -- post processing ------------------------------------------------
    def fold_octaves(self, cond: Condition, tokens: np.ndarray, max_jump: int = 10) -> np.ndarray:
        """Fold isolated octave jumps back toward the previous melody note."""
        out = tokens.copy()
        mel_slots = [i for i in range(cond.L) if cond.layer[i] == 2 and out[i] != PAD]
        prev_pitch = None
        lo, hi = cond.anchor + (-18), cond.anchor + 18
        for i in mel_slots:
            p = cond.anchor + token_to_rel(int(out[i]))
            if prev_pitch is not None and abs(p - prev_pitch) > max_jump:
                best, best_d = p, abs(p - prev_pitch)
                for shift in (-24, -12, 12, 24):
                    q = p + shift
                    if lo <= q <= hi and abs(q - prev_pitch) < best_d:
                        best, best_d = q, abs(q - prev_pitch)
                p = best
                rel = p - cond.anchor
                if -18 <= rel <= 18:
                    out[i] = rel_to_token(rel)
            prev_pitch = p
        return out

    def melody_abs(self, cond: Condition, tokens: np.ndarray) -> dict[int, int]:
        return {i: cond.anchor + token_to_rel(int(tokens[i]))
                for i in range(cond.L) if cond.layer[i] == 2 and tokens[i] != PAD}


def chord_tone_rate(cond: Condition, tokens: np.ndarray) -> float:
    total = hit = 0
    for i in range(cond.L):
        if cond.layer[i] != 2 or tokens[i] == PAD:
            continue
        total += 1
        if cond.is_chord_tone(i, token_to_rel(int(tokens[i]))):
            hit += 1
    return hit / max(1, total)


def in_key_rate(cond: Condition, tokens: np.ndarray) -> float:
    total = hit = 0
    for i in range(cond.L):
        if cond.layer[i] != 2 or tokens[i] == PAD:
            continue
        total += 1
        if theory.is_in_key(cond.anchor + token_to_rel(int(tokens[i])), cond.tonic, cond.mode):
            hit += 1
    return hit / max(1, total)


def apply_variation(sampler: MelodySampler, cond: Condition, tokens: np.ndarray,
                    variation: float, seed: int = 0) -> np.ndarray:
    """Break literal motif repetition: re-sample a few notes without the template bias."""
    if variation <= 0 or getattr(sampler, "target_mode", "pitch") == "delta":
        return tokens
    rng = np.random.default_rng(seed + 991)
    out = tokens.copy()
    melody_slots = [i for i in range(cond.L) if cond.layer[i] == 2]
    later = [i for i in melody_slots if i >= SLOTS_PER_BAR]
    n_mut = int(round(len(later) * min(1.0, variation)))
    if n_mut <= 0 or not later:
        return out
    chosen = sorted(rng.choice(later, size=min(n_mut, len(later)), replace=False).tolist())
    for i in chosen:
        prev = np.full(cond.L, PAD, dtype=np.int64)
        last = PAD
        for j in range(cond.L):
            prev[j] = last
            if j < i and cond.layer[j] == 2 and out[j] != PAD:
                last = out[j]
        tmpl = np.full(cond.L, PAD, dtype=np.int64)
        logits = sampler._logits_at(cond, prev, tmpl, i)
        from .data import N_TOKENS, N_PITCH_TOKENS
        allowed = sampler._allowed_for(cond)
        scores = logits.copy()
        scores[:N_PITCH_TOKENS][~allowed] = -1e9
        scores[N_PITCH_TOKENS:] = -1e9
        out[i] = int(np.argmax(scores))
    return sampler.fold_octaves(cond, out)


def calibrate_chord_bias(sampler: MelodySampler, cond: Condition, *, target=(0.50, 0.90),
                         candidates=(0.0, 0.3, 0.6, 0.9, 1.2, 1.6, 2.2, 3.0, 4.0),
                         seed: int = 0, mode: str = "free",
                         temperature: float = 1.0, w_template: float = 0.0,
                         w_repeat: float = 0.0, w_leap: float = 0.0, w_range: float = 0.0,
                         w_root: float = 0.0, w_bass: float = 0.0,
                         greedy: bool = False,
                         prime: list[tuple[int, int]] | None = None) -> tuple[np.ndarray, float, float]:
    """Sample with several chord-tone weights and pick the one landing in ``target``."""
    target_mid = 0.5 * (target[0] + target[1])
    trials = []
    for w in candidates:
        toks = sampler.generate(cond, mode=mode, temperature=temperature, w_chord=w,
                                w_template=w_template, w_repeat=w_repeat,
                                w_leap=w_leap, w_range=w_range, w_root=w_root,
                                w_bass=w_bass, seed=seed,
                                greedy=greedy, prime=prime)
        rate = chord_tone_rate(cond, toks)
        trials.append((abs(rate - target_mid), -int(target[0] <= rate <= target[1]),
                       toks, w, rate))
    trials.sort(key=lambda t: (t[1], t[0]))
    _score, _inrange, toks, w, rate = trials[0]
    return toks, w, rate
