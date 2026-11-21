"""Causal transformer over the 128-slot 16th-note grid.

Every slot carries: the previous melody note, the chord tones of its bar, its
layer (rest / pedal / melody), its bar position, its absolute grid position, the
chord root + degree, and (optional) a motif-template token.  The model predicts
a relative-pitch token for every slot; only melody slots contribute to the loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .data import (DELTA_PAD, N_DELTA_VOCAB, N_TOKENS, PAD, WINDOW_SLOTS)

IGNORE = -100


class TranceMelodyModel(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        ff_mult: int = 4,
        use_template: bool = True,
        n_slots: int = WINDOW_SLOTS,
        n_barpos: int = 16,
        chord_dim: int = 24,
        target_mode: str = "pitch",
        use_prev_delta: bool = False,
    ) -> None:
        super().__init__()
        self.config = dict(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
            ff_mult=ff_mult, use_template=use_template, n_slots=n_slots,
            n_barpos=n_barpos, chord_dim=chord_dim, target_mode=target_mode,
            use_prev_delta=use_prev_delta,
        )
        self.target_mode = target_mode
        self.use_prev_delta = use_prev_delta
        n_out = N_DELTA_VOCAB if target_mode == "delta" else N_TOKENS
        d = d_model
        self.use_template = use_template
        self.emb_prev = nn.Embedding(N_TOKENS, d, padding_idx=PAD)
        self.emb_layer = nn.Embedding(3, d)
        self.emb_barpos = nn.Embedding(n_barpos, d)
        self.emb_slotpos = nn.Embedding(n_slots, d)
        self.emb_root = nn.Embedding(12, d)
        self.emb_degree = nn.Embedding(7, d)
        self.emb_template = nn.Embedding(N_TOKENS, d, padding_idx=PAD) if use_template else None
        self.emb_prev_delta = (nn.Embedding(N_DELTA_VOCAB, d, padding_idx=DELTA_PAD)
                               if use_prev_delta else None)
        self.chord_proj = nn.Linear(chord_dim, d)

        n_parts = 7 + int(use_template) + int(use_prev_delta)
        self.proj = nn.Linear(d * n_parts, d)
        self.in_norm = nn.LayerNorm(d)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * ff_mult, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    # ------------------------------------------------------------------
    @staticmethod
    def causal_mask(n: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((n, n), float("-inf"), device=device), diagonal=1)

    def forward(
        self,
        prev: torch.Tensor,
        chord: torch.Tensor,
        layer: torch.Tensor,
        root: torch.Tensor,
        degree: torch.Tensor,
        barpos: torch.Tensor,
        slotpos: torch.Tensor,
        template: torch.Tensor | None = None,
        prev_delta: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        parts = [
            self.emb_prev(prev),
            self.emb_layer(layer),
            self.emb_barpos(barpos),
            self.emb_slotpos(slotpos),
            self.emb_root(root),
            self.emb_degree(degree),
            self.chord_proj(chord),
        ]
        if self.use_template:
            if template is None:
                template = torch.full_like(prev, PAD)
            parts.append(self.emb_template(template))
        if self.use_prev_delta:
            if prev_delta is None:
                prev_delta = torch.full_like(prev, DELTA_PAD)
            parts.append(self.emb_prev_delta(prev_delta))
        x = self.proj(torch.cat(parts, dim=-1))
        x = self.in_norm(x)
        mask = self.causal_mask(x.shape[1], x.device) if causal else None
        x = self.encoder(x, mask=mask)
        return self.head(self.out_norm(x))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_loss(logits: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.05) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=IGNORE,
        label_smoothing=label_smoothing,
    )


def save_checkpoint(path: str, model: TranceMelodyModel, extra: dict) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"model_config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> tuple[TranceMelodyModel, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = TranceMelodyModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
