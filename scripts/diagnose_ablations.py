"""Which inputs does the trained model actually use?  Input-ablation + entropy ceiling.

For teacher-forced melody slots:
  * shuffle `prev` across windows      -> does token accuracy drop?
  * rotate chord mask by a random k    -> does accuracy drop?  (measures harmony use)
  * rotate chord mask by (anchor-tonic) -> is the model using the *wrong* framing?
  * shuffle barpos/layer
  * empirical conditional entropy ceilings from the training split.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

import numpy as np
import torch

from trance_lora import data as D
from trance_lora.features import IGNORE
from trance_lora.model import load_checkpoint
from trance_lora.train import TrainConfig, prepare


def accuracy(model, arrs, *, shuffle_prev=False, rotate_chord=0, shuffle_barpos=False,
             rotate_root=0, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    prev = arrs["prev"].copy()
    chord = arrs["chord"].copy()
    barpos = arrs["barpos"].copy()
    root = arrs["root"].copy()
    if shuffle_prev:
        perm = rng.permutation(len(prev))
        prev = arrs["prev"][perm]
    if rotate_chord:
        chord = np.roll(chord, rotate_chord, axis=-1)
    if rotate_root:
        root = (arrs["root"] + rotate_root) % 12
    if shuffle_barpos:
        perm = rng.permutation(len(barpos))
        barpos = arrs["barpos"][perm]
    with torch.no_grad():
        logits = model(
            prev=torch.from_numpy(prev), chord=torch.from_numpy(chord),
            layer=torch.from_numpy(arrs["layer"]), root=torch.from_numpy(root),
            degree=torch.from_numpy(arrs["degree"]), barpos=torch.from_numpy(barpos),
            slotpos=torch.from_numpy(arrs["slotpos"]),
            template=torch.from_numpy(arrs["template"]))
    pred = logits.argmax(-1)
    target = torch.from_numpy(arrs["target"])
    mask = target != IGNORE
    acc = float((pred[mask] == target[mask]).float().mean())
    # log-prob of the true token and chord-tone mass, for a finer view
    logp = torch.log_softmax(logits, dim=-1)
    nll = float(-logp[mask].gather(1, target[mask].unsqueeze(1)).mean())
    return acc, nll


def empirical_entropy(arrs: dict, max_order: int = 2) -> dict:
    """Count-based conditional entropy (bits/slot) of the next melody token."""
    layer = arrs["layer"]
    target = arrs["target"]
    prev = arrs["prev"]
    barpos = arrs["barpos"]
    # chord-tone indicator for the *target* slot is a property of the slot; build
    # a compact chord signature: sorted tuple of chord-tone rel pcs is not available
    # here, so use (chord mask hash) as context.
    chord_sig = [tuple(np.nonzero(row)[0]) for row in np.round(arrs["chord"]).astype(int)]
    rows = []
    for b in range(len(layer)):
        pos = np.where(layer[b] == 2)[0]
        for i in pos:
            if target[b, i] == IGNORE:
                continue
            ctx1 = (int(prev[b, i]),)
            ctx2 = (int(prev[b, i]), chord_sig[b][i])
            ctx3 = (int(prev[b, i]), chord_sig[b][i], int(barpos[b, i]))
            rows.append((ctx1, ctx2, ctx3, int(target[b, i])))
    keys = [0, 1, 2]
    names = ["prev", "prev+chord", "prev+chord+barpos"]
    out = {}
    for ki, name in zip(keys, names):
        # build joint/conditional counts with add-1 smoothing over the alphabet seen
        ctx_counts: dict = {}
        for r in rows:
            key = r[ki]
            d = ctx_counts.setdefault(key, {})
            d[r[3]] = d.get(r[3], 0) + 1
        n_total = 0
        ent = 0.0
        V = max(len(c) for c in ctx_counts.values()) + 2
        for key, d in ctx_counts.items():
            tot = sum(d.values())
            n_total += tot
            ce = 0.0
            for c, n in d.items():
                p = n / tot
                ce -= p * np.log2(p)
            # unseen mass penalty
            p_unseen = 1.0 - (V - len(d)) / V * 0  # noqa: F841
            ent += tot * ce
        out[name] = ent / max(1, n_total)
    out["n"] = len(rows)
    return out


def main() -> int:
    torch.set_num_threads(4)
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT, "runs", "final_motif_s0", "model.pt")
    cfg = TrainConfig(cache=os.path.join(PROJECT, "data", "cache", "dataset.json"))
    data = prepare(cfg)
    model, _ = load_checkpoint(ckpt)
    test = data["test"]
    train = data["train"]

    print(f"model: {ckpt}")
    acc, nll = accuracy(model, test)
    print(f"  baseline            acc {acc:.3f}  nll {nll:.3f}")
    for name, kw in [
        ("prev shuffled", dict(shuffle_prev=True)),
        ("chord rotated +1", dict(rotate_chord=1)),
        ("chord rotated +5", dict(rotate_chord=5)),
        ("chord rotated +6", dict(rotate_chord=6)),
        ("root rotated +5", dict(rotate_root=5)),
        ("barpos shuffled", dict(shuffle_barpos=True)),
    ]:
        a, n = accuracy(model, test, seed=1, **kw)
        print(f"  {name:20s} acc {a:.3f} (Δ{a-acc:+.3f})  nll {n:.3f} (Δ{n-nll:+.3f})")

    print("\nempirical conditional entropy (train split), bits/slot:")
    for k, v in empirical_entropy(train).items():
        print(f"  {k:18s} {v}")

    # how predictable is the *interval* given prev interval?
    print("\ncopy baseline vs model on intervals (teacher forced, test):")
    layer = test["layer"]
    target = test["target"]
    prev = test["prev"]
    diff_pred, diff_true = [], []
    copy_err, model_err = [], []
    with torch.no_grad():
        logits = model(prev=torch.from_numpy(test["prev"]), chord=torch.from_numpy(test["chord"]),
                       layer=torch.from_numpy(layer), root=torch.from_numpy(test["root"]),
                       degree=torch.from_numpy(test["degree"]),
                       barpos=torch.from_numpy(test["barpos"]),
                       slotpos=torch.from_numpy(test["slotpos"]),
                       template=torch.from_numpy(test["template"]))
    pred = logits.argmax(-1).numpy()
    for b in range(len(layer)):
        pos = np.where(layer[b] == 2)[0]
        p = pred[b, pos] - D.PITCH_OFFSET
        t = target[b, pos] - D.PITCH_OFFSET
        if len(p) < 2:
            continue
        dp, dt = np.diff(p), np.diff(t)
        model_err.append(np.abs(dp - dt))
        copy_err.append(np.abs(dt))
    me = np.concatenate(model_err)
    ce = np.concatenate(copy_err)
    print(f"  model argmax interval error: {me.mean():.3f}")
    print(f"  copy-previous-note error   : {ce.mean():.3f}")
    print(f"  -> model {'BETTER' if me.mean() < ce.mean() else 'WORSE'} than copy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
