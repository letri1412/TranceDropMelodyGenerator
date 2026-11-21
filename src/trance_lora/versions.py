"""Version registry for the delivery pipeline.

v1 = absolute-pitch model only (the previous shipped default)
v2 = hybrid logit mixture (pitch + interval models), mix = 0.35 (new default)

Both versions share the same checkpoints repository, so rolling back is just a
matter of naming a version: no files are overwritten and no model is retrained.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_VERSION = "v2"


def _p(*parts: str) -> str:
    return os.path.join(PROJECT, *parts)


@dataclass(frozen=True)
class Version:
    name: str
    description: str
    target_mode: str                       # pitch | delta
    mix: float | None                      # None = single-model, else hybrid weight
    mode: str = "plan_root"
    w_chord: float | None = None            # None = auto-calibrate to 50-90%
    w_template: float = 2.0
    w_repeat: float = 1.2
    w_leap: float = 0.0
    w_range: float = 0.2
    w_bass: float = 0.8
    checkpoints: tuple[str, ...] = ()
    delta_checkpoints: tuple[str, ...] = ()

    def paths(self) -> dict:
        return {
            "checkpoint": [_p(*c.split("/")) for c in self.checkpoints],
            "delta_checkpoint": [_p(*c.split("/")) for c in self.delta_checkpoints],
        }

    def decode(self) -> dict:
        return {"mode": self.mode, "w_chord": self.w_chord,
                "w_template": self.w_template, "w_repeat": self.w_repeat,
                "w_leap": self.w_leap, "w_range": self.w_range, "w_bass": self.w_bass}


VERSIONS: dict[str, Version] = {
    "v1": Version(
        name="v1",
        description="绝对音高模型（pitch）+ 默认解码；上一版交付，保留用于回滚",
        target_mode="pitch",
        mix=None,
        mode="plan_root",
        w_chord=None,
        w_template=2.0,
        w_repeat=1.2,
        w_leap=0.4,
        w_range=0.3,
        w_bass=1.5,
        checkpoints=("runs/final_motif_s0/model.pt", "runs/final_motif_s1/model.pt",
                     "runs/final_motif_s2/model.pt"),
    ),
    "v2": Version(
        name="v2",
        description="混合解码 v2（pitch + interval，mix=0.35）；当前默认",
        target_mode="pitch",
        mix=0.35,
        mode="plan_root",
        w_chord=None,
        w_template=2.0,
        w_repeat=1.2,
        w_leap=0.0,
        w_range=0.2,
        w_bass=0.8,
        checkpoints=("runs/final_motif_s0/model.pt", "runs/final_motif_s1/model.pt",
                     "runs/final_motif_s2/model.pt"),
        delta_checkpoints=("runs/stepA_final_motif_s0/model.pt",
                           "runs/stepA_final_motif_s1/model.pt",
                           "runs/stepA_final_motif_s2/model.pt"),
    ),
}


def get(name: str) -> Version:
    if name not in VERSIONS:
        raise KeyError(f"unknown version {name!r}; available: {', '.join(VERSIONS)}")
    return VERSIONS[name]


def list_versions() -> list[Version]:
    return [VERSIONS[k] for k in sorted(VERSIONS)]


def find_by_checkpoints(checkpoint_paths: list[str], mix: float | None) -> str:
    """Best-effort: which registered version does an explicit checkpoint set match?"""
    norm = {os.path.normcase(os.path.abspath(p)) for p in checkpoint_paths}
    for v in list_versions():
        want = {os.path.normcase(os.path.abspath(p)) for p in v.paths()["checkpoint"]}
        if norm == want and (v.mix == mix or mix is None):
            return v.name
    return "custom"
