"""Music theory helpers for the Trance melody model.

Everything is expressed relative to the tonic (pitch class 0 = tonic).
Only the seven diatonic triads of the key are modelled (matches the corpus).
"""

from __future__ import annotations

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NAME_TO_PC = {n: i for i, n in enumerate(NOTE_NAMES)}
# enharmonic aliases accepted when parsing keys
NAME_TO_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})

SCALES = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
}

# (roman label, root pc rel. tonic, triad pitch classes rel. tonic)
TRIADS = {
    "minor": [
        ("i", 0, (0, 3, 7)),
        ("ii\u00b0", 2, (2, 5, 8)),
        ("III", 3, (3, 7, 10)),
        ("iv", 5, (5, 8, 0)),
        ("v", 7, (7, 10, 2)),
        ("VI", 8, (8, 0, 3)),
        ("VII", 10, (10, 2, 5)),
    ],
    "major": [
        ("I", 0, (0, 4, 7)),
        ("ii", 2, (2, 5, 9)),
        ("iii", 4, (4, 7, 11)),
        ("IV", 5, (5, 9, 0)),
        ("V", 7, (7, 11, 2)),
        ("vi", 9, (9, 0, 4)),
        ("vii\u00b0", 11, (11, 2, 5)),
    ],
}

# Default progression used by the generator when the user gives none (most common
# per-bar progression of the corpus: i i VI VI iv iv v v).
DEFAULT_PROGRESSION = ["i", "i", "VI", "VI", "iv", "iv", "v", "v"]
# Default rhythm pattern (most common enhanced-training pattern, half-bar repeated).
DEFAULT_RHYTHM = "12112112"


def parse_key(text: str) -> tuple[int, str]:
    """``"Cmin"`` / ``"F#maj"`` -> (tonic pitch class, mode)."""
    t = text.strip()
    if t.endswith("min"):
        root, mode = t[:-3], "minor"
    elif t.endswith("maj"):
        root, mode = t[:-3], "major"
    elif t.endswith("m"):
        raise ValueError(f"key {text!r}: use {t[:-1]}min, not {t!r} (README forbids 'Am')")
    else:
        raise ValueError(f"key {text!r} must end with min/maj")
    if root not in NAME_TO_PC:
        raise ValueError(f"unknown note name {root!r}")
    return NAME_TO_PC[root], mode


def key_text(tonic: int, mode: str) -> str:
    return f"{NOTE_NAMES[tonic % 12]}{'min' if mode == 'minor' else 'maj'}"


def scale_pcs(tonic: int, mode: str) -> set[int]:
    return {(tonic + s) % 12 for s in SCALES[mode]}


def is_in_key(pitch: int, tonic: int, mode: str) -> bool:
    return (pitch - tonic) % 12 in SCALES[mode]


def nearest_in_key(pitch: float, tonic: int, mode: str) -> int:
    """Nearest absolute MIDI pitch that belongs to the key."""
    return min(
        (p for p in range(0, 128) if (p - tonic) % 12 in SCALES[mode]),
        key=lambda p: (abs(p - pitch), p),
    )


def degree_of_pitch(pitch: int, tonic: int, mode: str) -> tuple[int, int] | None:
    """Absolute pitch -> (scale degree 0..6, octave) inside the key, or None."""
    scale = SCALES[mode]
    pc = (pitch - tonic) % 12
    if pc not in scale:
        return None
    deg = scale.index(pc)
    octv = (pitch - tonic - pc) // 12
    return deg, octv


def pitch_of_degree(degree: int, octave: int, tonic: int, mode: str) -> int:
    scale = SCALES[mode]
    deg, octv = degree % 7, octave + degree // 7
    return tonic + scale[deg] + 12 * octv


def transpose_pitch_in_scale(pitch: int, steps: int, tonic: int, mode: str) -> int | None:
    """Diatonic transposition of an absolute pitch by ``steps`` scale steps."""
    d = degree_of_pitch(pitch, tonic, mode)
    if d is None:
        return None
    return pitch_of_degree(d[0] + steps, d[1], tonic, mode)


def transpose_rel(rel: int, steps: int, anchor: int, tonic: int, mode: str) -> int | None:
    """Transpose an *anchor-relative* pitch by scale steps (returns anchor-relative)."""
    p = transpose_pitch_in_scale(anchor + rel, steps, tonic, mode)
    return None if p is None else p - anchor


# --- legacy helpers: these assume ``rel`` is measured from the tonic ----------
def degree_of_rel(rel: int, mode: str) -> tuple[int, int] | None:
    scale = SCALES[mode]
    for octv in range(-12, 13):
        for deg, semi in enumerate(scale):
            if semi + 12 * octv == rel:
                return deg, octv
    return None


def rel_of_degree(degree: int, octave: int, mode: str) -> int:
    scale = SCALES[mode]
    deg, octv = degree % 7, octave + degree // 7
    return scale[deg] + 12 * octv


def transpose_rel_in_scale(rel: int, steps: int, mode: str) -> int | None:
    """Deprecated: only valid when ``rel`` is tonic-relative (anchor == tonic)."""
    d = degree_of_rel(rel, mode)
    if d is None:
        return None
    return rel_of_degree(d[0] + steps, d[1], mode)


def triad_for_pcs(rel_pcs: set[int], mode: str = "minor", min_coverage: float = 0.8):
    """Match a set of pitch classes (rel. tonic) to a diatonic triad.

    Returns ``(label, root_pc, triad_pcs)`` or ``None``.
    """
    best = None
    for label, root, pcs in TRIADS[mode]:
        inter = len(rel_pcs & set(pcs))
        cov = inter / float(len(pcs))
        extra = len(rel_pcs - set(pcs))
        if cov >= min_coverage and extra == 0:
            if best is None or cov > best[0]:
                best = (cov, label, root, pcs)
    return None if best is None else (best[1], best[2], best[3])


def triad_by_label(label: str, mode: str = "minor"):
    for lab, root, pcs in TRIADS[mode]:
        if lab == label:
            return root, pcs
    raise ValueError(f"unknown chord label {label!r}; valid: {[t[0] for t in TRIADS[mode]]}")


def degree_of_triad(label: str, mode: str = "minor") -> int:
    for i, (lab, _, _) in enumerate(TRIADS[mode]):
        if lab == label:
            return i
    raise ValueError(f"unknown chord label {label!r}")


def pedal_pitch_for_root(root_pc: int, tonic: int, anchor: int, offset: int = 15) -> int:
    """Pedal note: chord root in the register ``anchor - offset``."""
    target = anchor - offset
    cands = {"__pedal__": 0}
    target_pc = (tonic + root_pc) % 12
    return min(
        (p for p in range(0, 128) if p % 12 == target_pc),
        key=lambda p: (abs(p - target), p),
    )


def parse_progression(text: str, n_bars: int = 8, mode: str = "minor") -> list[str]:
    """Parse ``"i,VI,III,VII"`` or ``"i i VI VI iv iv v v"`` into per-bar labels.

    A chord listed once spans ``n_bars / len`` bars (must divide evenly);
    a bare list of 8 labels is used as-is.
    """
    seps = [",", " ", "-", ">"]
    parts = [text]
    for s in seps:
        parts = [p for chunk in parts for p in chunk.split(s)]
    labels = [p.strip() for p in parts if p.strip()]
    if not labels:
        raise ValueError("empty progression")
    for lab in labels:
        triad_by_label(lab, mode)  # validate
    if len(labels) == n_bars:
        return labels
    if n_bars % len(labels) != 0:
        raise ValueError(f"{len(labels)} chords cannot evenly span {n_bars} bars")
    span = n_bars // len(labels)
    return [lab for lab in labels for _ in range(span)]


def parse_rhythm(pattern: str, n_bars: int = 8, n_melody_per_bar: int | None = None) -> list[int]:
    """Expand a rhythm pattern into a per-16th layer mask.

    Accepted forms:
      * 8 chars (half-bar pattern, repeated twice per bar)  -> ``"12112112"``
      * 16 chars (one full bar)                             -> ``"1211211212112112"``
    Values: ``1`` pedal, ``2`` melody, ``0``/``-`` rest.
    """
    p = pattern.strip()
    if "-" in p:
        p = p.replace("-", "0")
    if len(p) == 8:
        bar = p + p
    elif len(p) == 16:
        bar = p
    else:
        raise ValueError(f"rhythm must be 8 or 16 chars, got {len(p)}: {pattern!r}")
    if any(c not in "012" for c in bar):
        raise ValueError(f"rhythm chars must be 0/1/2, got {pattern!r}")
    mask = [int(c) for c in bar] * n_bars
    if n_melody_per_bar is not None and bar.count("2") != n_melody_per_bar:
        raise ValueError(f"rhythm has {bar.count('2')} melody slots, expected {n_melody_per_bar}")
    return mask
