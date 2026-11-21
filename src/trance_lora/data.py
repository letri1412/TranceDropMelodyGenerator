"""Dataset discovery, validation and extraction.

The corpus is a flat set of folders; every sample is a *pair* of MIDI files

    <song>__<bpm>__<key>__<section>__<lead|chord>.mid

The ``lead`` track mixes two layers on one 16th-note grid: a low **pedal** that
follows the chord root and a high **melody**.  This module splits the two,
recovers the per-slot layer mask, decodes the chord track into per-slot chord
tones and normalises melody pitch as ``pitch - anchor`` so that the model learns
key-independent melodic shape.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import theory
from .midi_io import GRID_TICKS, read_midi

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SLOTS_PER_BAR = 16
DEFAULT_BARS = 8
WINDOW_SLOTS = DEFAULT_BARS * SLOTS_PER_BAR  # 128

REL_MIN, REL_MAX = -18, 18
PITCH_OFFSET = -REL_MIN                      # token = rel + PITCH_OFFSET
N_PITCH_TOKENS = REL_MAX - REL_MIN + 1       # 31
PAD = N_PITCH_TOKENS
N_TOKENS = N_PITCH_TOKENS + 1

# --- Step A: interval (delta) target representation ------------------------
# The corpus melody intervals live in [-22, +19] (p99 |delta| = 15), so a
# [-24, +24] vocabulary covers every observed transition.
DELTA_MIN, DELTA_MAX = -24, 24
DELTA_OFFSET = -DELTA_MIN                    # token = delta + 24
N_DELTA_TOKENS = DELTA_MAX - DELTA_MIN + 1   # 49
DELTA_PAD = N_DELTA_TOKENS                   # 49
N_DELTA_VOCAB = N_DELTA_TOKENS + 1           # 50

ALLOWED_SECTIONS = {"drop", "build", "breakdown", "intro", "outro", "full"}
ROLES = {"lead", "chord"}

NAME_RE = re.compile(
    r"^(?P<song>.+?)__(?P<bpm>\d+)__(?P<key>[A-G](?:#|b)?)(?P<mode>maj|min)"
    r"__(?P<section>[a-z]+)__(?P<role>lead|chord)$"
)

DEFAULT_ROOT = os.environ.get("TRANCE_DATASET", r"F:\DSHWorkSpace\Trance旋律训练集")


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedName:
    song: str
    bpm: int
    key: str
    tonic: int
    mode: str
    section: str
    role: str

    @property
    def stem(self) -> str:
        return f"{self.song}__{self.bpm}__{self.key}{'min' if self.mode == 'minor' else 'maj'}__{self.section}"

    @property
    def key_text(self) -> str:
        return f"{self.key}{'min' if self.mode == 'minor' else 'maj'}"


def parse_name(filename: str) -> ParsedName:
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = NAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"filename {filename!r} does not match <song>__<bpm>__<key><maj|min>__<section>__<lead|chord>.mid"
        )
    d = m.groupdict()
    tonic = theory.NAME_TO_PC[d["key"]]
    mode = "minor" if d["mode"] == "min" else "major"
    if d["section"] not in ALLOWED_SECTIONS:
        raise ValueError(f"section {d['section']!r} not in {sorted(ALLOWED_SECTIONS)}")
    return ParsedName(d["song"], int(d["bpm"]), d["key"], tonic, mode, d["section"], d["role"])


# ---------------------------------------------------------------------------
# pair discovery
# ---------------------------------------------------------------------------
@dataclass
class Pair:
    stem: str
    lead_path: str
    chord_path: str
    meta: ParsedName
    folder: str


def scan_pairs(root: str = DEFAULT_ROOT) -> tuple[list[Pair], list[dict]]:
    """Find lead/chord pairs under ``root``.  Folders starting with ``_`` are skipped."""
    found: dict[str, dict] = defaultdict(dict)
    folders: dict[str, str] = {}
    issues: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("_"))
        for fn in sorted(filenames):
            if not fn.lower().endswith(".mid"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                meta = parse_name(fn)
            except ValueError as exc:
                issues.append({"code": "name_invalid", "severity": "error", "file": path, "detail": str(exc)})
                continue
            key = meta.stem
            if meta.role in found[key]:
                issues.append({"code": "duplicate_role", "severity": "error", "file": path,
                               "detail": f"{meta.role} for {key} already seen"})
                continue
            found[key][meta.role] = path
            folders[key] = os.path.relpath(dirpath, root)

    pairs: list[Pair] = []
    for key, roles in sorted(found.items()):
        meta = parse_name(os.path.basename(roles.get("lead") or roles.get("chord")))
        if "lead" not in roles:
            issues.append({"code": "pair_missing", "severity": "error", "file": roles.get("chord", key),
                           "detail": "chord without lead"})
            continue
        if "chord" not in roles:
            issues.append({"code": "pair_missing", "severity": "error", "file": roles["lead"],
                           "detail": "lead without chord"})
            continue
        pairs.append(Pair(key, roles["lead"], roles["chord"], meta, folders[key]))
    return pairs, issues


# ---------------------------------------------------------------------------
# layer inference
# ---------------------------------------------------------------------------
def infer_layer_mask(slots: dict[int, list[int]], n_slots: int, period: int = 16,
                     min_gap: int = 3) -> tuple[list[int], dict]:
    """Split the lead track into pedal (1) / melody (2) / empty (0) per slot.

    The rhythm pattern is stable per bar position, so the pitch minimum of each
    position over all bars separates a low pedal layer from a high melody layer.
    """
    pos_pitches: dict[int, list[int]] = defaultdict(list)
    for s, ps in slots.items():
        if s < n_slots:
            pos_pitches[s % period].append(min(ps))
    if not pos_pitches:
        return [0] * n_slots, {"reason": "no notes"}
    mins = {r: min(v) for r, v in pos_pitches.items()}
    order = sorted(mins.items(), key=lambda kv: kv[1])
    best_gap, cut = 0, None
    for i in range(len(order) - 1):
        gap = order[i + 1][1] - order[i][1]
        if gap > best_gap:
            best_gap, cut = gap, i
    layer = [0] * n_slots
    if cut is None or best_gap < min_gap:
        # no separable pedal layer: everything is melody
        for s in slots:
            if s < n_slots:
                layer[s] = 2
        return layer, {"gap": best_gap, "mode": "all_melody",
                       "reason": f"best_gap={best_gap} < {min_gap}"}
    thresh = (order[cut][1] + order[cut + 1][1]) / 2.0
    pedal_pos = {r for r, m in mins.items() if m <= thresh}
    for s in slots:
        if s < n_slots:
            layer[s] = 1 if (s % period) in pedal_pos else 2
    return layer, {"gap": best_gap, "threshold": thresh,
                   "pedal_positions": sorted(pedal_pos)}


# ---------------------------------------------------------------------------
# song extraction
# ---------------------------------------------------------------------------
@dataclass
class Song:
    song_id: str
    stem: str
    song: str
    folder: str
    bpm: int
    tonic: int
    mode: str
    section: str
    n_bars: int
    anchor: int
    layer: list[int]
    melody_rel: list[int]        # sentinel NONE for non-melody slots
    melody_abs: list[int]
    pedal_abs: list[int]
    chord_pcs: list[int]         # 12-bit mask of chord tones rel. tonic
    chord_root_pc: list[int]     # rel. tonic
    chord_degree: list[int]      # 0..6 index into the key's triads
    chord_labels: list[str]      # per-bar roman label
    rhythm_pattern: str          # 16 chars
    stats: dict = field(default_factory=dict)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "song_id": self.song_id, "stem": self.stem, "song": self.song,
            "folder": self.folder, "bpm": self.bpm, "tonic": self.tonic,
            "mode": self.mode, "section": self.section, "n_bars": self.n_bars,
            "anchor": self.anchor, "layer": self.layer,
            "melody_rel": self.melody_rel, "melody_abs": self.melody_abs,
            "pedal_abs": self.pedal_abs, "chord_pcs": self.chord_pcs,
            "chord_root_pc": self.chord_root_pc, "chord_degree": self.chord_degree,
            "chord_labels": self.chord_labels, "rhythm_pattern": self.rhythm_pattern,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Song":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    @property
    def n_slots(self) -> int:
        return self.n_bars * SLOTS_PER_BAR

    def melody_positions(self) -> list[int]:
        return [i for i, l in enumerate(self.layer) if l == 2]


NONE = -999


def _chord_slots(chord_path: str, n_slots: int, tonic: int, mode: str) -> tuple[list[int], list[int], list[int], list[str], list[str]]:
    """Decode a chord track into per-slot chord tone masks / roots / degrees."""
    data = read_midi(chord_path)
    onsets: dict[int, list[int]] = defaultdict(list)
    for n in data.notes:
        slot = int(round(n.start_tick / GRID_TICKS))
        onsets[slot].append(n.pitch)

    segments = []  # (start_slot, pcs mask, root_pc, degree, label)
    unresolved = []
    for start in sorted(onsets):
        pitches = sorted(set(onsets[start]))
        rel_pcs = {(p - tonic) % 12 for p in pitches}
        match = theory.triad_for_pcs(rel_pcs, mode)
        if match is None:
            # tolerate one foreign tone: try dropping each pitch class
            best = None
            for pc in rel_pcs:
                m = theory.triad_for_pcs(rel_pcs - {pc}, mode)
                if m is not None:
                    best = m
                    break
            if best is None:
                unresolved.append((start, sorted(rel_pcs)))
                continue
            match = best
        label, root, pcs = match
        mask = 0
        for pc in pcs:
            mask |= 1 << pc
        degree = theory.degree_of_triad(label, mode)
        segments.append((start, mask, root, degree, label))
    if not segments:
        raise ValueError(f"no decodable chord in {chord_path}")

    chord_pcs = [0] * n_slots
    chord_root = [0] * n_slots
    chord_degree = [0] * n_slots
    labels = []
    cur = segments[0]
    for s in range(n_slots):
        for seg in segments:
            if seg[0] <= s:
                cur = seg
        chord_pcs[s] = cur[1]
        chord_root[s] = cur[2]
        chord_degree[s] = cur[3]
    for b in range(n_slots // SLOTS_PER_BAR):
        s0 = b * SLOTS_PER_BAR
        lab = segments[0][4]
        for seg in segments:
            if seg[0] <= s0:
                lab = seg[4]
        labels.append(lab)
    return chord_pcs, chord_root, chord_degree, labels, unresolved


def extract_song(pair: Pair, issues: list[dict] | None = None) -> Song:
    """Extract one fully normalised song from its lead/chord pair."""
    issues = issues if issues is not None else []
    meta = pair.meta
    lead = read_midi(pair.lead_path)
    chord = read_midi(pair.chord_path)
    if lead.ticks_per_beat != chord.ticks_per_beat:
        raise ValueError(f"ticks_per_beat differ: {lead.ticks_per_beat} vs {chord.ticks_per_beat}")
    if lead.tempo_bpm is None:
        issues.append({"code": "tempo_missing", "severity": "error", "file": pair.lead_path,
                       "detail": "no set_tempo event"})
    elif round(lead.tempo_bpm) != meta.bpm:
        issues.append({"code": "bpm_mismatch", "severity": "error", "file": pair.lead_path,
                       "detail": f"filename bpm {meta.bpm} != midi tempo {lead.tempo_bpm:.2f}"})
    if chord.tempo_bpm is not None and lead.tempo_bpm is not None and abs(chord.tempo_bpm - lead.tempo_bpm) > 0.01:
        issues.append({"code": "bpm_mismatch", "severity": "error", "file": pair.chord_path,
                       "detail": f"chord tempo {chord.tempo_bpm:.2f} != lead tempo {lead.tempo_bpm:.2f}"})

    # -- lead grid -------------------------------------------------------
    slots: dict[int, list[int]] = defaultdict(list)
    for n in lead.notes:
        slots[int(round(n.start_tick / GRID_TICKS))].append(n.pitch)
    poly = {s: sorted(set(ps)) for s, ps in slots.items() if len(set(ps)) > 1}
    if poly:
        issues.append({"code": "lead_polyphonic", "severity": "warn", "file": pair.lead_path,
                       "detail": f"{len(poly)} slots with >1 distinct pitch"})
    last_end = max((n.start_tick + n.dur_tick for n in lead.notes), default=0)
    n_slots_lead = int(round(last_end / GRID_TICKS))
    n_bars = n_slots_lead // SLOTS_PER_BAR
    if n_slots_lead % SLOTS_PER_BAR:
        issues.append({"code": "bars_not_aligned", "severity": "error", "file": pair.lead_path,
                       "detail": f"lead length {n_slots_lead} slots is not a whole number of bars"})
    if n_bars not in (8, 16):
        issues.append({"code": "bars_unexpected", "severity": "error", "file": pair.lead_path,
                       "detail": f"{n_bars} bars (expected 8 or 16)"})
    n_slots = n_bars * SLOTS_PER_BAR
    slots = {s: ps for s, ps in slots.items() if 0 <= s < n_slots}
    for s in slots:
        slots[s] = [max(set(slots[s]), key=slots[s].count)]

    if len(slots) < n_slots:
        issues.append({"code": "lead_sparse", "severity": "info", "file": pair.lead_path,
                       "detail": f"{n_slots - len(slots)} empty 16th slots"})

    layer, layer_diag = infer_layer_mask(slots, n_slots)
    melody_rel = [NONE] * n_slots
    melody_abs = [NONE] * n_slots
    pedal_abs = [NONE] * n_slots
    for s, ps in slots.items():
        if layer[s] == 2:
            melody_abs[s] = ps[0]
        elif layer[s] == 1:
            pedal_abs[s] = ps[0]

    mel_pitches = [p for p in melody_abs if p != NONE]
    if not mel_pitches:
        raise ValueError(f"{pair.lead_path}: no melody notes after layer split")
    med = float(np.median(mel_pitches))
    anchor = theory.nearest_in_key(med, meta.tonic, meta.mode)
    for s in range(n_slots):
        if melody_abs[s] != NONE:
            rel = melody_abs[s] - anchor
            if not (REL_MIN <= rel <= REL_MAX):
                raise ValueError(f"{pair.lead_path}: relative pitch {rel} outside [{REL_MIN},{REL_MAX}]")
            melody_rel[s] = rel

    # -- chord grid ------------------------------------------------------
    chord_pcs, chord_root, chord_degree, chord_labels, unresolved = _chord_slots(
        pair.chord_path, n_slots, meta.tonic, meta.mode)
    if unresolved:
        issues.append({"code": "chord_unresolved", "severity": "error", "file": pair.chord_path,
                       "detail": f"{len(unresolved)} chord(s) do not match a diatonic triad: {unresolved[:3]}"})

    # -- rhythm pattern --------------------------------------------------
    pattern = "".join(str(layer[s]) for s in range(SLOTS_PER_BAR))

    # -- statistics / quality checks -------------------------------------
    mel_pos = [s for s in range(n_slots) if layer[s] == 2]
    ped_pos = [s for s in range(n_slots) if layer[s] == 1]
    in_key_mel = sum(1 for s in mel_pos if theory.is_in_key(melody_abs[s], meta.tonic, meta.mode))
    in_key_ped = sum(1 for s in ped_pos if theory.is_in_key(pedal_abs[s], meta.tonic, meta.mode))

    def pcs_of(mask: int) -> set[int]:
        return {i for i in range(12) if mask & (1 << i)}

    chord_tone_mel = sum(1 for s in mel_pos
                         if (melody_abs[s] - meta.tonic) % 12 in pcs_of(chord_pcs[s]))
    chord_tone_ped = sum(1 for s in ped_pos
                         if (pedal_abs[s] - meta.tonic) % 12 in pcs_of(chord_pcs[s]))

    stats = {
        "n_melody": len(mel_pos),
        "n_pedal": len(ped_pos),
        "in_key_melody": in_key_mel / max(1, len(mel_pos)),
        "in_key_pedal": in_key_ped / max(1, len(ped_pos)),
        "chord_tone_melody": chord_tone_mel / max(1, len(mel_pos)),
        "chord_tone_pedal": chord_tone_ped / max(1, len(ped_pos)),
        "anchor": anchor,
        "layer_gap": layer_diag.get("gap"),
        "rhythm_pattern": pattern,
    }
    if stats["in_key_melody"] < 0.99:
        issues.append({"code": "melody_in_key_low", "severity": "warn", "file": pair.lead_path,
                       "detail": f"melody in-key {stats['in_key_melody']:.3f}"})
    if stats["chord_tone_pedal"] < 0.80:
        issues.append({"code": "pedal_chord_tone_low", "severity": "warn", "file": pair.lead_path,
                       "detail": f"pedal chord-tone {stats['chord_tone_pedal']:.3f}"})
    if ped_pos and mel_pos:
        sep = min(melody_abs[s] for s in mel_pos) - max(pedal_abs[s] for s in ped_pos)
        stats["register_separation"] = sep
        if sep < 4:
            issues.append({"code": "layer_separation_low", "severity": "info", "file": pair.lead_path,
                           "detail": f"min(melody)-max(pedal) = {sep} semitones"})

    return Song(
        song_id=pair.stem, stem=pair.stem, song=meta.song, folder=pair.folder,
        bpm=meta.bpm, tonic=meta.tonic, mode=meta.mode, section=meta.section,
        n_bars=n_bars, anchor=anchor, layer=layer, melody_rel=melody_rel,
        melody_abs=melody_abs, pedal_abs=pedal_abs, chord_pcs=chord_pcs,
        chord_root_pc=chord_root, chord_degree=chord_degree, chord_labels=chord_labels,
        rhythm_pattern=pattern, stats=stats,
    )


# ---------------------------------------------------------------------------
# full dataset build / validation
# ---------------------------------------------------------------------------
def build_dataset(root: str = DEFAULT_ROOT) -> tuple[list[Song], list[dict], dict]:
    pairs, issues = scan_pairs(root)
    songs: list[Song] = []
    for p in pairs:
        try:
            songs.append(extract_song(p, issues))
        except Exception as exc:  # noqa: BLE001 - report, don't crash the whole scan
            issues.append({"code": "extract_failed", "severity": "error", "file": p.lead_path,
                           "detail": f"{type(exc).__name__}: {exc}"})
    summary = {
        "root": os.path.abspath(root),
        "n_pairs": len(pairs),
        "n_songs": len(songs),
        "n_issues": len(issues),
        "n_errors": sum(1 for i in issues if i["severity"] == "error"),
        "n_warnings": sum(1 for i in issues if i["severity"] == "warn"),
        "n_bars": dict(Counter(s.n_bars for s in songs)),
        "keys": dict(Counter(theory.key_text(s.tonic, s.mode) for s in songs)),
        "bpm": dict(Counter(s.bpm for s in songs)),
        "n_melody_notes": int(sum(s.stats["n_melody"] for s in songs)),
        "n_pedal_notes": int(sum(s.stats["n_pedal"] for s in songs)),
        "progression_counts": dict(Counter(" ".join(s.chord_labels) for s in songs).most_common()),
        "rhythm_patterns": dict(Counter(s.rhythm_pattern for s in songs).most_common()),
        "mean_melody_chord_tone": float(np.mean([s.stats["chord_tone_melody"] for s in songs])),
        "mean_pedal_chord_tone": float(np.mean([s.stats["chord_tone_pedal"] for s in songs])),
        "mean_melody_in_key": float(np.mean([s.stats["in_key_melody"] for s in songs])),
    }
    return songs, issues, summary


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def save_cache(songs: list[Song], summary: dict, issues: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"version": 3, "summary": summary, "issues": issues,
               "songs": [s.to_dict() for s in songs]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def load_cache(path: str) -> tuple[list[Song], dict, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    songs = [Song.from_dict(d) for d in payload["songs"]]
    return songs, payload["summary"], payload["issues"]


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
@dataclass
class Window:
    song_id: str
    indices: np.ndarray   # slot indices into the song arrays
    rotation_bars: int
    cyclic: bool


def build_windows(song: Song, bars: int = DEFAULT_BARS) -> list[Window]:
    """Training windows: cyclic 2/4/6-bar rotations for 8-bar songs, sliding for 16-bar."""
    n = bars * SLOTS_PER_BAR
    L = song.n_slots
    out: list[Window] = []
    if L == n:
        for r in (0, 2, 4, 6):
            idx = (np.arange(n) + r * SLOTS_PER_BAR) % n
            out.append(Window(song.song_id, idx, r, True))
        return out
    step = SLOTS_PER_BAR * 4
    for off in range(0, L - n + 1, step):
        out.append(Window(song.song_id, np.arange(n) + off, off // SLOTS_PER_BAR, False))
    return out


def split_songs(song_ids: list[str], seed: int = 42, val_frac: float = 0.10, test_frac: float = 0.11):
    """Song-level split (windows of the same song must never straddle sets)."""
    rng = np.random.default_rng(seed)
    ids = sorted(song_ids)
    perm = rng.permutation(len(ids))
    n_test = max(1, int(round(len(ids) * test_frac)))
    n_val = max(1, int(round(len(ids) * val_frac)))
    test = [ids[i] for i in perm[:n_test]]
    val = [ids[i] for i in perm[n_test:n_test + n_val]]
    train = [ids[i] for i in perm[n_test + n_val:]]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
