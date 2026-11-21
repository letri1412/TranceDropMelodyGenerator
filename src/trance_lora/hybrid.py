"""Hybrid decoding: mix the absolute-pitch model and the interval model in logit space.

The pitch model emits a distribution over absolute relative-pitch tokens; the
interval model emits one over intervals.  For each slot the interval distribution
can be projected into absolute space given the previous melody note, so the two
can be mixed:

    p_mix(rel) = (1 - mix) * p_pitch(rel) + mix * p_delta(rel)

``mix = 0`` is the pure pitch model (more variance, occasionally more "human"),
``mix = 1`` is the pure interval model (more faithful to the corpus, more reliable).
Everything downstream (in-key mask, chord-tone bias, motif template, bass-following,
leap/range penalties) is unchanged because the wrapper returns absolute-space logits.

The wrapper is a drop-in "model" for :class:`trance_lora.sample.MelodySampler`, so no
existing code path is modified.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data import (DELTA_OFFSET, DELTA_PAD, N_DELTA_TOKENS, N_PITCH_TOKENS, N_TOKENS,
                   PAD, PITCH_OFFSET)

MIX_PAD_IDX = N_PITCH_TOKENS  # PAD token id in the absolute vocabulary


def compute_prev_delta(prev: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
    """Recover the previous interval token from the ``prev``/``layer`` inputs.

    ``prev[i]`` is the last melody note before slot i, so for a melody slot i the
    interval of the *previous* transition is ``prev[i] - prev[j]`` where j is the
    previous melody slot (0 when that slot has no earlier note).
    """
    B, L = prev.shape
    out = torch.full((B, L), DELTA_PAD, dtype=torch.long, device=prev.device)
    prev_np = prev.cpu().numpy()
    layer_np = layer.cpu().numpy()
    out_np = out.cpu().numpy()
    for b in range(B):
        last_mel = -1
        for i in range(L):
            if layer_np[b, i] != 2:
                continue
            if prev_np[b, i] == PAD:
                out_np[b, i] = DELTA_PAD
            else:
                cur = int(prev_np[b, i]) - PITCH_OFFSET
                if last_mel < 0 or prev_np[b, last_mel] == PAD:
                    base = 0
                else:
                    base = int(prev_np[b, last_mel]) - PITCH_OFFSET
                d = cur - base
                if -24 <= d <= 24:
                    out_np[b, i] = d + DELTA_OFFSET
            last_mel = i
    return torch.from_numpy(out_np).to(prev.device)


class HybridLogitModel(nn.Module):
    """Return absolute-space logits that blend a pitch model and a delta model."""

    def __init__(self, pitch_model: nn.Module, delta_model: nn.Module, mix: float = 0.5):
        super().__init__()
        if not (0.0 <= mix <= 1.0):
            raise ValueError("mix must be in [0, 1]")
        self.pitch_model = pitch_model
        self.delta_model = delta_model
        self.mix = float(mix)
        # MelodySampler reads these to pick its (absolute-space) decoding path.
        self.config = {"target_mode": "pitch", "use_prev_delta": False,
                       "d_model": getattr(getattr(pitch_model, "config", {}), "get", lambda k, d=None: d)("d_model", 0)}

    def eval(self):
        self.pitch_model.eval()
        self.delta_model.eval()
        return self

    def to(self, device):
        self.pitch_model.to(device)
        self.delta_model.to(device)
        return self

    def param_count(self) -> int:
        f = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        return f(self.pitch_model) + f(self.delta_model)

    @torch.no_grad()
    def forward(self, prev, chord, layer, root, degree, barpos, slotpos,
                template=None, **kw):
        lp = self.pitch_model(prev=prev, chord=chord, layer=layer, root=root,
                             degree=degree, barpos=barpos, slotpos=slotpos,
                             template=template)
        prev_delta = compute_prev_delta(prev, layer).to(prev.device)
        ld = self.delta_model(prev=prev, chord=chord, layer=layer, root=root,
                              degree=degree, barpos=barpos, slotpos=slotpos,
                              template=template, prev_delta=prev_delta)
        p_pitch = torch.softmax(lp, dim=-1)                       # (B,L,N_TOKENS)
        p_delta = torch.softmax(ld, dim=-1)                       # (B,L,N_DELTA_VOCAB)

        B, L = prev.shape
        prev_rel = torch.where(prev == PAD, torch.zeros_like(prev), prev - PITCH_OFFSET)
        p_abs = torch.zeros((B, L, N_TOKENS), dtype=p_pitch.dtype, device=p_pitch.device)
        # scatter the interval probabilities to the absolute tokens they imply
        for d in range(N_DELTA_TOKENS):
            rel = prev_rel + (d - DELTA_OFFSET)
            tok = rel + PITCH_OFFSET
            ok = (tok >= 0) & (tok < N_PITCH_TOKENS)
            tok_safe = torch.where(ok, tok, torch.full_like(tok, MIX_PAD_IDX))
            p_abs.scatter_add_(2, tok_safe.unsqueeze(-1), p_delta[..., d].unsqueeze(-1))
        p_abs[..., MIX_PAD_IDX] = p_pitch[..., MIX_PAD_IDX]
        p = (1.0 - self.mix) * p_pitch + self.mix * p_abs
        return torch.log(p.clamp_min(1e-12))
