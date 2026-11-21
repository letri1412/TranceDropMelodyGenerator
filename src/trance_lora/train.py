"""Training loop for the Trance melody model (CPU friendly)."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import features
from .data import (DELTA_MAX, DELTA_MIN, DELTA_OFFSET, PAD, PITCH_OFFSET, REL_MAX,
                   REL_MIN, SLOTS_PER_BAR, WINDOW_SLOTS, Song, build_windows,
                   load_cache, split_songs)
from .features import IGNORE, token_to_rel
from .model import TranceMelodyModel, masked_loss, save_checkpoint


@dataclass
class TrainConfig:
    cache: str = ""
    out_dir: str = ""
    seed: int = 0
    # model
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.25
    ff_mult: int = 4
    use_template: bool = True
    target_mode: str = "pitch"        # pitch (absolute rel token) | delta (interval token)
    use_prev_delta: bool = False      # feed the previous interval as an extra input
    # optimisation
    epochs: int = 400
    patience: int = 80
    batch_size: int = 8
    lr: float = 2e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    label_smoothing: float = 0.08
    prev_dropout: float = 0.10
    template_dropout: float = 0.30
    template_plan_prob: float = 0.50
    warmup_frac: float = 0.05
    select_metric: str = "token_acc"   # token_acc (maximise) | val_loss (minimise) | none
    restore_best: bool = True          # False = keep the weights of the last epoch
    # data
    bars: int = 8
    split_seed: int = 42
    val_frac: float = 0.10
    test_frac: float = 0.11
    train_song_ids: list[str] | None = None   # override: train on these song ids
    # runtime
    num_threads: int = 4
    log_every: int = 25
    save_every: int = 0
    device: str = "cpu"


def prepare(cfg: TrainConfig):
    songs, summary, _issues = load_cache(cfg.cache)
    by_id = {s.song_id: s for s in songs}
    all_ids = [s.song_id for s in songs]
    split = split_songs(all_ids, seed=cfg.split_seed, val_frac=cfg.val_frac, test_frac=cfg.test_frac)
    windows_by_song = {sid: build_windows(by_id[sid], bars=cfg.bars) for sid in all_ids}

    train_ids = cfg.train_song_ids if cfg.train_song_ids is not None else split["train"]
    arrays_train = features.build_arrays(songs, windows_by_song, train_ids)
    arrays_val = features.build_arrays(songs, windows_by_song, split["val"])
    arrays_test = features.build_arrays(songs, windows_by_song, split["test"])
    return {
        "songs": songs, "summary": summary, "split": split,
        "train": arrays_train, "val": arrays_val, "test": arrays_test,
        "windows_by_song": windows_by_song,
    }


# ---------------------------------------------------------------------------
# batch helpers
# ---------------------------------------------------------------------------
KEYS = ("prev", "chord", "layer", "root", "degree", "barpos", "slotpos",
        "template", "template_plan", "target", "melody", "delta", "prev_delta")


def _slice(arrs: dict, idx: np.ndarray) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(arrs[k][idx])) for k in KEYS if k in arrs}


def _forward_batch(model: TranceMelodyModel, b: dict[str, torch.Tensor],
                   template: torch.Tensor | None = None) -> torch.Tensor:
    kw = {}
    if getattr(model, "use_prev_delta", False):
        kw["prev_delta"] = b["prev_delta"]
    return model(
        prev=b["prev"], chord=b["chord"], layer=b["layer"], root=b["root"],
        degree=b["degree"], barpos=b["barpos"], slotpos=b["slotpos"],
        template=template if template is not None else b["template"], **kw,
    )


# ---------------------------------------------------------------------------
# metrics (teacher forced)
# ---------------------------------------------------------------------------
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def teacher_forced_metrics(logits: torch.Tensor, b: dict[str, torch.Tensor],
                           anchor: np.ndarray | None = None) -> dict:
    pred = logits.argmax(-1).numpy()
    target = b["target"].numpy()
    melody = b["melody"].numpy()
    layer = b["layer"].numpy()
    mask = target != IGNORE
    n = int(mask.sum())
    if n == 0:
        return {}
    acc = float((pred[mask] == target[mask]).mean())
    out = {"token_acc": acc, "n_melody": n}

    interval_preds, interval_refs = [], []
    dup_pred, dup_ref = 0, 0
    contour_rs = []
    distinct_preds = []
    pred_rel_all, true_rel_all = [], []
    for i in range(pred.shape[0]):
        pos = np.where(layer[i] == 2)[0]
        if len(pos) < 2:
            continue
        p = pred[i, pos].astype(np.int64) - PITCH_OFFSET
        t = melody[i, pos].astype(np.int64) - PITCH_OFFSET

        pred_rel_all.extend(p.tolist())
        true_rel_all.extend(t.tolist())
        dp, dt = np.diff(p), np.diff(t)
        interval_preds.extend(dp.tolist())
        interval_refs.extend(dt.tolist())
        dup_pred += int((dp == 0).sum())
        dup_ref += int((dt == 0).sum())
        distinct_preds.append(len(np.unique(p)))
        r = _corr(p.astype(float), t.astype(float))
        if not math.isnan(r):
            contour_rs.append(r)
    pred_rel_all = np.array(pred_rel_all)
    true_rel_all = np.array(true_rel_all)
    out.update({
        "interval_l1": float(np.mean(np.abs(np.array(interval_preds) - np.array(interval_refs))))
        if interval_preds else float("nan"),
        "contour_r": float(np.mean(contour_rs)) if contour_rs else float("nan"),
        "dup_rate_pred": dup_pred / max(1, len(interval_preds)),
        "dup_rate_ref": dup_ref / max(1, len(interval_refs)),
        "distinct_per_window": float(np.mean(distinct_preds)) if distinct_preds else float("nan"),
        # key / chord metrics are filled in by caller (needs anchor + chord masks)
        "pred_rel": pred_rel_all, "true_rel": true_rel_all,
    })
    return out


def teacher_forced_metrics_delta(logits: torch.Tensor, b: dict[str, torch.Tensor]) -> dict:
    """Teacher-forced metrics for the interval-target model.

    Each slot is scored against the *true* previous note (teacher forcing), i.e. the
    implied absolute pitch is ``true_prev_rel + predicted_delta``; this avoids the
    error accumulation of integrating the predicted sequence and is the direct
    analogue of the absolute-pitch model's metrics.
    """
    import math as _math

    pred = logits.argmax(-1).numpy()
    layer = b["layer"].numpy()
    melody = b["melody"].numpy()
    prev_tok = b["prev"].numpy()
    acc_hits = acc_tot = 0
    abs_err: list[float] = []
    interval_preds: list[float] = []
    interval_refs: list[float] = []
    dup_pred = dup_ref = 0
    distinct_preds: list[int] = []
    contour_rs: list[float] = []
    pred_rel_all: list[int] = []
    true_rel_all: list[int] = []
    for i in range(pred.shape[0]):
        pos = np.where(layer[i] == 2)[0]
        if len(pos) < 1:
            continue
        prev_rel = np.where(prev_tok[i, pos] == PAD, 0,
                            prev_tok[i, pos] - PITCH_OFFSET).astype(np.int64)
        d = pred[i, pos].astype(np.int64) - DELTA_OFFSET
        pred_rel = prev_rel + d
        true_rel = melody[i, pos].astype(np.int64) - PITCH_OFFSET
        acc_hits += int((pred_rel == true_rel).sum())
        acc_tot += len(pos)
        abs_err.extend(np.abs(pred_rel - true_rel).astype(float).tolist())
        if len(pos) >= 2:
            dp, dt = np.diff(pred_rel), np.diff(true_rel)
            interval_preds.extend(dp.astype(float).tolist())
            interval_refs.extend(dt.astype(float).tolist())
            dup_pred += int((dp == 0).sum())
            dup_ref += int((dt == 0).sum())
            distinct_preds.append(int(len(np.unique(pred_rel))))
            r = _corr(pred_rel.astype(float), true_rel.astype(float))
            if not _math.isnan(r):
                contour_rs.append(r)
            pred_rel_all.extend(pred_rel.tolist())
            true_rel_all.extend(true_rel.tolist())
    if acc_tot == 0:
        return {}
    return {
        "token_acc": acc_hits / acc_tot,
        "n_melody": acc_tot,
        "interval_l1": float(np.mean(abs_err)),
        "contour_r": float(np.mean(contour_rs)) if contour_rs else float("nan"),
        "dup_rate_pred": dup_pred / max(1, len(interval_preds)),
        "dup_rate_ref": dup_ref / max(1, len(interval_refs)),
        "distinct_per_window": float(np.mean(distinct_preds)) if distinct_preds else float("nan"),
        "pred_rel": np.asarray(pred_rel_all), "true_rel": np.asarray(true_rel_all),
    }


@torch.no_grad()
def evaluate_loss(model: TranceMelodyModel, arrs: dict[str, np.ndarray], batch_size: int = 32,
                  label_smoothing: float = 0.0,
                  template_key: str = "template") -> tuple[float, dict]:
    model.eval()
    n = len(arrs["layer"])
    total, count = 0.0, 0
    metrics_accum: list[dict] = []
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n))
        b = _slice(arrs, idx)
        tmpl = b[template_key] if model.use_template else None
        logits = _forward_batch(model, b, tmpl)
        if getattr(model, "target_mode", "pitch") == "delta":
            loss = masked_loss(logits, b["delta"], label_smoothing=label_smoothing)
            metrics_accum.append(teacher_forced_metrics_delta(logits, b))
        else:
            loss = masked_loss(logits, b["target"], label_smoothing=label_smoothing)
            metrics_accum.append(teacher_forced_metrics(logits, b))
        total += float(loss) * len(idx)
        count += len(idx)
    model.train()
    keys = ["token_acc", "interval_l1", "contour_r", "dup_rate_pred", "dup_rate_ref",
            "distinct_per_window"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_accum if k in m and not math.isnan(m[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    all_pred = np.concatenate([m["pred_rel"] for m in metrics_accum if "pred_rel" in m]) \
        if metrics_accum and "pred_rel" in metrics_accum[0] else np.array([])
    all_true = np.concatenate([m["true_rel"] for m in metrics_accum if "true_rel" in m]) \
        if metrics_accum and "true_rel" in metrics_accum[0] else np.array([])
    return total / max(1, count), {**agg, "pred_rel": all_pred, "true_rel": all_true}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig, data: dict | None = None) -> dict:
    torch.set_num_threads(cfg.num_threads)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = data or prepare(cfg)
    tr, va = data["train"], data["val"]
    n_train = len(tr["layer"])
    steps_per_epoch = max(1, math.ceil(n_train / cfg.batch_size))
    total_steps = cfg.epochs * steps_per_epoch

    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "train.log") if cfg.out_dir else None
    if log_path and os.path.exists(log_path):
        os.remove(log_path)

    def log(msg: str) -> None:
        print(msg, flush=True)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    model = TranceMelodyModel(
        d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        dropout=cfg.dropout, ff_mult=cfg.ff_mult, use_template=cfg.use_template,
        target_mode=cfg.target_mode,
        use_prev_delta=cfg.use_prev_delta or cfg.target_mode == "delta",
    ).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_steps = int(total_steps * cfg.warmup_frac)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return cfg.lr * (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))

    history: list[dict] = []
    best = {"val_loss": float("inf"), "epoch": -1, "state": None, "metrics": {},
            "select_value": float("inf") if cfg.select_metric == "val_loss" else -1e9}
    patience_left = cfg.patience
    t0 = time.time()
    step = 0
    model.train()
    for epoch in range(cfg.epochs):
        perm = rng.permutation(n_train)
        run_loss, run_count = 0.0, 0
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            b = _slice(tr, idx)
            # input token dropout on the previous-melody stream
            if cfg.prev_dropout > 0:
                drop = torch.from_numpy((rng.random(b["prev"].shape) < cfg.prev_dropout))
                b["prev"] = torch.where(drop, torch.full_like(b["prev"], features.PAD), b["prev"])
            template = None
            if cfg.use_template:
                use_plan = torch.from_numpy(rng.random(len(idx)) < cfg.template_plan_prob)[:, None]
                template = torch.where(use_plan, b["template_plan"], b["template"])
                if cfg.template_dropout > 0:
                    drop = torch.from_numpy(rng.random(len(idx)) < cfg.template_dropout)[:, None]
                    template = torch.where(drop, torch.full_like(template, features.PAD), template)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            logits = _forward_batch(model, b, template)
            loss_target = b["delta"] if cfg.target_mode == "delta" else b["target"]
            loss = masked_loss(logits, loss_target, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            step += 1
            run_loss += float(loss.detach()) * len(idx)
            run_count += len(idx)

        train_loss = run_loss / max(1, run_count)
        if model.use_template:
            val_loss_raw, _vm = evaluate_loss(model, va, label_smoothing=0.0, template_key="template")
            val_loss_plan, val_m = evaluate_loss(model, va, label_smoothing=0.0, template_key="template_plan")
            val_loss = 0.5 * (val_loss_raw + val_loss_plan)
        else:
            val_loss, val_m = evaluate_loss(model, va, label_smoothing=0.0)
        rec = {"epoch": epoch, "step": step, "train_loss": train_loss, "val_loss": val_loss,
               "lr": lr_at(step), **{k: v for k, v in val_m.items() if k not in ("pred_rel", "true_rel")}}
        if model.use_template:
            rec["val_loss_raw"] = val_loss_raw
            rec["val_loss_plan"] = val_loss_plan
        history.append(rec)
        # -- model selection: token accuracy peaks much later than val loss -------
        if cfg.select_metric == "none":
            improved = False
        elif cfg.select_metric == "token_acc":
            cur = float(rec.get("token_acc") or 0.0)
            improved = cur > best["select_value"] + 1e-4
        else:
            cur = val_loss
            improved = cur < best["select_value"] - 1e-5
        rec["select_value"] = cur if cfg.select_metric != "none" else None
        if improved:
            best = {"val_loss": val_loss, "epoch": epoch, "select_value": cur,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "metrics": rec}
            patience_left = cfg.patience
        elif cfg.select_metric != "none":
            patience_left -= 1
        if cfg.log_every and (epoch % cfg.log_every == 0 or epoch == 0):
            log(f"  ep {epoch:4d} train {train_loss:.4f} val {val_loss:.4f} "
                f"acc {rec.get('token_acc', float('nan')):.3f} "
                f"ival {rec.get('interval_l1', float('nan')):.3f} "
                f"r {rec.get('contour_r', float('nan')):.3f} ({time.time()-t0:.1f}s)")
        if cfg.out_dir and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            with open(os.path.join(cfg.out_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch, "best_epoch": best["epoch"],
                           "best_val_loss": best["val_loss"], "val_loss": val_loss,
                           "train_loss": train_loss, "elapsed_s": time.time() - t0}, f)
        if cfg.out_dir and cfg.save_every and epoch > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(os.path.join(cfg.out_dir, f"ckpt_e{epoch:03d}.pt"), model,
                            {"epoch": epoch, "val_loss": val_loss,
                             "train_loss": train_loss})
        if patience_left <= 0:
            log(f"  early stop at epoch {epoch} (best {best['epoch']})")
            break

    if cfg.restore_best and best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    if model.use_template:
        test_loss_raw, test_m = evaluate_loss(model, data["test"], template_key="template")
        test_loss_plan, _ = evaluate_loss(model, data["test"], template_key="template_plan")
        test_loss = 0.5 * (test_loss_raw + test_loss_plan)
    else:
        test_loss, test_m = evaluate_loss(model, data["test"])
    result = {
        "config": asdict(cfg),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "test_loss": test_loss,
        "test_teacher_forced": {k: v for k, v in test_m.items() if k not in ("pred_rel", "true_rel")},
        "history": history,
        "param_count": model.param_count(),
        "seconds": time.time() - t0,
        "split": data["split"],
    }
    if cfg.out_dir:
        os.makedirs(cfg.out_dir, exist_ok=True)
        save_checkpoint(os.path.join(cfg.out_dir, "model.pt"), model,
                        {"train_result": {k: v for k, v in result.items() if k != "history"}})
        with open(os.path.join(cfg.out_dir, "train_result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        with open(os.path.join(cfg.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=0)
        np.savez_compressed(os.path.join(cfg.out_dir, "test_preds.npz"),
                            pred_rel=test_m.get("pred_rel", np.array([])),
                            true_rel=test_m.get("true_rel", np.array([])))
    result["model"] = model
    return result
