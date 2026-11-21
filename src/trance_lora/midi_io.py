"""Minimal Standard-MIDI-File reading / writing (mido based).

Reading pairs note_on/note_off events into explicit notes and quantises them onto
the project's 16th-note grid.  Writing produces exactly the corpus format:
type-1 file, 96 ticks per beat, tempo track + one note track, 1/16 notes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import mido

TICKS_PER_BEAT = 96
STEPS_PER_BEAT = 4
GRID_TICKS = TICKS_PER_BEAT // STEPS_PER_BEAT  # 24 ticks = one 16th


@dataclass
class Note:
    start_tick: int
    dur_tick: int
    pitch: int
    vel: int = 100
    channel: int = 0
    track: int = 0


@dataclass
class MidiFileData:
    path: str
    ticks_per_beat: int
    tempo_bpm: float | None
    time_signature: tuple[int, int] | None
    notes: list[Note] = field(default_factory=list)
    n_tracks: int = 0


def read_midi(path: str) -> MidiFileData:
    """Read a MIDI file and pair note-on/note-off into notes (absolute ticks)."""
    mid = mido.MidiFile(path)
    tempo = None
    tsig = None
    notes: list[Note] = []
    pending: dict[tuple[int, int], list[tuple[int, int, int]]] = {}

    for ti, tr in enumerate(mid.tracks):
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo" and tempo is None:
                tempo = mido.tempo2bpm(msg.tempo)
            elif msg.type == "time_signature" and tsig is None:
                tsig = (msg.numerator, msg.denominator)
            elif msg.type == "note_on" and msg.velocity > 0:
                pending.setdefault((msg.channel, msg.note), []).append((t, msg.velocity, ti))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                q = pending.get(key)
                if q:
                    st, vel, tri = q.pop(0)
                    notes.append(Note(st, t - st, msg.note, vel, msg.channel, tri))
    # unterminated notes (malformed file): close at last event tick
    last = 0
    for tr in mid.tracks:
        last += sum(m.time for m in tr)
    for (ch, pitch), q in pending.items():
        for st, vel, tri in q:
            notes.append(Note(st, max(0, last - st), pitch, vel, ch, tri))

    notes.sort(key=lambda n: (n.start_tick, n.pitch))
    return MidiFileData(
        path=path,
        ticks_per_beat=mid.ticks_per_beat,
        tempo_bpm=tempo,
        time_signature=tsig,
        notes=notes,
        n_tracks=len(mid.tracks),
    )


def notes_to_slots(data: MidiFileData) -> dict[int, list[int]]:
    """Group note pitches by 16th-note slot index (absolute slot from tick 0)."""
    slots: dict[int, list[int]] = {}
    for n in data.notes:
        slot = int(round(n.start_tick / GRID_TICKS))
        slots.setdefault(slot, []).append(n.pitch)
    for v in slots.values():
        v.sort()
    return slots


def write_midi(
    path: str,
    bpm: float,
    notes: list[tuple[int, int]],
    ticks_per_beat: int = TICKS_PER_BEAT,
    duration_steps: float = 1.0,
    velocity: int = 100,
    channel: int = 0,
    track_name: str = "Lead",
    extra_tracks: list[tuple[str, list[tuple[int, int]]]] | None = None,
    watermark: str | None = None,
) -> None:
    """Write a type-1 MIDI file.

    ``notes`` is a list of ``(slot, pitch)`` on the 16th-note grid; every note gets
    ``duration_steps`` 16ths.  ``extra_tracks`` adds further ``(name, notes)``
    tracks (e.g. a reference chord track) after the lead track.
    """
    from .watermark import copyright_text, watermark_text

    wm = watermark if watermark is not None else watermark_text()
    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Conductor", time=0))
    meta.append(mido.MetaMessage("text", text=wm, time=0))
    meta.append(mido.MetaMessage("copyright", text=copyright_text(), time=0))
    meta.append(mido.MetaMessage("marker", text=wm, time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)

    def build_track(name: str, note_pairs: list[tuple[int, int]]) -> mido.MidiTrack:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        tr.append(mido.MetaMessage("text", text=wm, time=0))
        events: list[tuple[int, int, mido.Message]] = []
        dur_ticks = int(round(duration_steps * GRID_TICKS))
        for slot, pitch in note_pairs:
            start = int(slot) * GRID_TICKS
            events.append((start, 1, mido.Message("note_on", note=int(pitch),
                                                  velocity=velocity, channel=channel, time=0)))
            events.append((start + dur_ticks, 0, mido.Message("note_off", note=int(pitch),
                                                              velocity=0, channel=channel, time=0)))
        events.sort(key=lambda e: (e[0], e[1]))
        prev = 0
        for tick, _, msg in events:
            msg.time = tick - prev
            prev = tick
            tr.append(msg)
        tr.append(mido.MetaMessage("end_of_track", time=0))
        return tr

    mid.tracks.append(build_track(track_name, notes))
    for name, pairs in (extra_tracks or []):
        mid.tracks.append(build_track(name, pairs))

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not hasattr(mido, "MidiFile"):
        raise RuntimeError("mido not available")
    mid.save(path)


def write_chord_track(path: str, bpm: float, chords: list[tuple[int, int, list[int]]],
                      ticks_per_beat: int = TICKS_PER_BEAT) -> None:
    """Write a chord-only reference MIDI. ``chords`` = (start_slot, length_slots, pitches)."""
    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)
    tr = mido.MidiTrack()
    events: list[tuple[int, int, mido.Message]] = []
    for start_slot, length_slots, pitches in chords:
        start = int(start_slot) * GRID_TICKS
        end = start + int(length_slots) * GRID_TICKS
        for p in pitches:
            events.append((start, 1, mido.Message("note_on", note=int(p), velocity=90, channel=0, time=0)))
            events.append((end, 0, mido.Message("note_off", note=int(p), velocity=0, channel=0, time=0)))
    events.sort(key=lambda e: (e[0], e[1]))
    prev = 0
    for tick, _, msg in events:
        msg.time = tick - prev
        prev = tick
        tr.append(msg)
    tr.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(tr)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    mid.save(path)
