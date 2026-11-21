"""Unified CLI: ``python -m trance_lora <command>``.

Commands
--------
validate   scan + validate the corpus (read-only)
extract    validate and write the parsed dataset cache
train      train one model
evaluate   evaluate checkpoints with generation metrics
generate   generate a MIDI melody from key/bpm/progression/rhythm
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import data as D
from . import theory


def _print_issues(issues: list[dict], limit: int = 60) -> None:
    if not issues:
        print("  no issues")
        return
    by_code: dict[str, list[dict]] = {}
    for i in issues:
        by_code.setdefault(f"{i['severity']}:{i['code']}", []).append(i)
    for key in sorted(by_code):
        rows = by_code[key]
        print(f"  [{key}] x{len(rows)}")
        for r in rows[: max(1, limit // max(1, len(by_code)))]:
            print(f"      {os.path.basename(r.get('file', '?'))}: {r.get('detail', '')}")


def cmd_validate(args: argparse.Namespace) -> int:
    print(f"scanning {args.root} ...")
    songs, issues, summary = D.build_dataset(args.root)
    print("\n== summary ==")
    for k in ("root", "n_pairs", "n_songs", "n_issues", "n_errors", "n_warnings",
              "n_bars", "bpm", "n_melody_notes", "n_pedal_notes",
              "mean_melody_chord_tone", "mean_pedal_chord_tone", "mean_melody_in_key"):
        v = summary.get(k)
        if isinstance(v, float):
            print(f"  {k:26s} {v:.4f}")
        else:
            print(f"  {k:26s} {v}")
    print(f"  {'keys':26s} {len(summary['keys'])} distinct: {summary['keys']}")
    print(f"  {'rhythm_patterns':26s} {summary['rhythm_patterns']}")
    print(f"\n== issues ({len(issues)}) ==")
    _print_issues(issues)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "issues": issues,
                       "songs": [s.to_dict() for s in songs]}, f, ensure_ascii=False, indent=1)
        print(f"\nreport written to {args.out}")
    return 1 if summary["n_errors"] else 0


def cmd_extract(args: argparse.Namespace) -> int:
    songs, issues, summary = D.build_dataset(args.root)
    D.save_cache(songs, summary, issues, args.out)
    print(f"songs: {len(songs)}  errors: {summary['n_errors']}  warnings: {summary['n_warnings']}")
    print(f"cache written to {args.out}")
    return 1 if summary["n_errors"] else 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train
    train_ids = None
    if args.train_on_all:
        import json
        songs, _s, _i = D.load_cache(args.cache)
        if os.path.exists(args.split_file):
            with open(args.split_file, "r", encoding="utf-8") as f:
                split = json.load(f)
            train_ids = sorted(split["train"] + split["val"])
            print(f"training on train+val = {len(train_ids)} songs")
        else:
            train_ids = [s.song_id for s in songs]
            print(f"training on all {len(train_ids)} songs")
    cfg = TrainConfig(
        cache=args.cache, out_dir=args.out, seed=args.seed,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        dropout=args.dropout, use_template=not args.no_template,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        lr=args.lr, label_smoothing=args.label_smoothing,
        prev_dropout=args.prev_dropout, template_dropout=args.template_dropout,
        num_threads=args.threads, log_every=args.log_every,
        train_song_ids=train_ids, save_every=args.save_every,
        select_metric=args.select_metric, restore_best=not args.keep_last,
        target_mode=args.target_mode,
    )
    print(f"training seed={cfg.seed} template={cfg.use_template} -> {cfg.out_dir}")
    result = train(cfg)
    print(f"\nbest val loss {result['best_val_loss']:.4f} @epoch {result['best_epoch']} "
          f"({result['seconds']:.0f}s, {result['param_count']} params)")
    print(f"test loss {result['test_loss']:.4f}")
    for k, v in result["test_teacher_forced"].items():
        print(f"  {k:22s} {v:.4f}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import EvalConfig, evaluate, teacher_forced_summary, write_markdown
    ckpts = [p for p in args.checkpoints if p]
    if not ckpts:
        print("no checkpoints given (--checkpoints)", file=sys.stderr)
        return 2
    print(f"evaluating {len(ckpts)} checkpoint(s) on song-held-out test set")
    cfg = EvalConfig(
        cache=args.cache, checkpoints=ckpts, split_seed=args.split_seed,
        modes=tuple(args.modes.split(",")), protocols=tuple(args.protocols.split(",")),
        seeds=tuple(range(args.seeds)), calibrate=not args.no_calibrate,
        threads=args.threads, out_dir=args.out_dir,
        calib_candidates=tuple(float(x) for x in args.calib_candidates.split(",")),
        target_mode=args.target_mode,
    )
    result = evaluate(cfg)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    tf = teacher_forced_summary(ckpts, args.cache, args.split_seed) if args.teacher_forced else {}
    if tf:
        with open(os.path.join(args.out_dir, "teacher_forced.json"), "w", encoding="utf-8") as f:
            json.dump(tf, f, ensure_ascii=False, indent=1)
    md = os.path.join(args.out_dir, "eval.md")
    write_markdown(result, tf, md, title=args.title)
    print(f"\n== corpus reference ==")
    for k in ("in_key", "chord_tone", "motif_consistency", "distinct_pitches", "mean_abs_interval"):
        print(f"  {k:20s} {result['corpus'].get(k):.4f}")
    print("\n== generated ==")
    hdr = f"{'mode/protocol':22s} {'in_key':>7s} {'chord':>7s} {'motif':>7s} {'ival_l1':>8s} {'contour':>8s} {'dup':>6s} {'distinct':>9s}"
    print(hdr)
    for key in [f"{m}/{p}" for p in ("continuation", "free") for m in ("free", "motif", "plan")]:
        g = result["groups"].get(key)
        if not g:
            continue
        print(f"{key:22s} {g.get('in_key', float('nan')):7.3f} {g.get('chord_tone', float('nan')):7.3f} "
              f"{g.get('motif_consistency', float('nan')):7.3f} {g.get('interval_l1', float('nan')):8.3f} "
              f"{g.get('contour_r', float('nan')):8.3f} {g.get('dup_rate', float('nan')):6.3f} "
              f"{g.get('distinct_pitches', float('nan')):9.1f}")
    print(f"\nreport: {md}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from . import versions
    from .generate import GenerateSpec, ascii_piano_roll, describe, generate, render_midi
    from .model import load_checkpoint
    from .sample import MelodySampler
    import torch

    if getattr(args, "list_versions", False):
        for v in versions.list_versions():
            mark = " (default)" if v.name == versions.DEFAULT_VERSION else ""
            print(f"{v.name}{mark}: {v.description}")
            print(f"   mix={v.mix}  decode={v.decode()}")
        return 0

    torch.set_num_threads(args.threads)
    ver = None
    if args.version:
        ver = versions.get(args.version)
    elif not args.checkpoint:
        ver = versions.get(versions.DEFAULT_VERSION)
        print(f"using default version {ver.name}: {ver.description}")
    if ver is not None:
        os.environ["TRANCE_WATERMARK_VERSION"] = ver.name
        if args.checkpoint:
            print("both --version and --checkpoint given; --checkpoint wins", file=sys.stderr)
        else:
            paths = ver.paths()
            args.checkpoint = paths["checkpoint"]
            args.delta_checkpoint = paths["delta_checkpoint"] or None
            if args.mix is None:
                args.mix = ver.mix
            for key, val in ver.decode().items():
                if getattr(args, key, None) is None:
                    setattr(args, key, val)

    if not args.out:
        print("--out is required", file=sys.stderr)
        return 2
    models = []
    for p in args.checkpoint:
        m, _ = load_checkpoint(p)
        models.append(m)
    if not models:
        print("no --checkpoint given (and no --version)", file=sys.stderr)
        return 2
    if args.delta_checkpoint:
        from .evaluate import Ensemble
        from .hybrid import HybridLogitModel
        delta_models = [load_checkpoint(p)[0] for p in args.delta_checkpoint]
        pitch = Ensemble(models) if len(models) > 1 else models[0]
        delta = Ensemble(delta_models) if len(delta_models) > 1 else delta_models[0]
        mix = 0.5 if args.mix is None else float(args.mix)
        print(f"hybrid decoding: mix={mix} (0=pitch model, 1=interval model)")
        sampler = MelodySampler(HybridLogitModel(pitch, delta, mix=mix))
    elif len(models) > 1:
        from .evaluate import Ensemble
        sampler = MelodySampler(Ensemble(models))
    else:
        sampler = MelodySampler(models[0])
    spec = GenerateSpec(
        key=args.key, bpm=args.bpm, progression=args.progression, rhythm=args.rhythm,
        bars=args.bars, anchor=args.anchor, variation=args.variation,
        temperature=args.temperature,
        greedy=args.greedy, seed=args.seed, pedal=not args.no_pedal,
        pedal_offset=args.pedal_offset, chord_track=args.chord_track,
        **{k: getattr(args, k) for k in
           ("mode", "w_chord", "w_template", "w_repeat", "w_leap", "w_range", "w_bass")
           if getattr(args, k) is not None},
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    # repeat with different seeds for a batch of takes
    base, ext = os.path.splitext(args.out)
    for k in range(args.takes):
        spec.seed = args.seed + k
        result = generate(sampler, spec)
        path = args.out if args.takes == 1 else f"{base}_{k:02d}{ext}"
        render_midi(result, path, spec)
        print(f"[{k}] -> {path}")
        print(describe(spec, result))
        if args.show_roll:
            print(ascii_piano_roll(result.condition, result))
        if args.stats_json:
            import json
            with open(args.stats_json, "w", encoding="utf-8") as f:
                json.dump({"spec": {kk: vv for kk, vv in spec.__dict__.items()},
                           "stats": result.stats}, f, ensure_ascii=False, indent=1)
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    songs, _s, _i = D.load_cache(args.cache)
    split = D.split_songs([s.song_id for s in songs], seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=1)
    print(f"split written to {args.out}")
    for k, v in split.items():
        print(f"  {k:6s} {len(v):3d} songs")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trance_lora", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="scan and validate the corpus")
    v.add_argument("--root", default=D.DEFAULT_ROOT)
    v.add_argument("--out", default="")
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("extract", help="build the parsed dataset cache")
    e.add_argument("--root", default=D.DEFAULT_ROOT)
    e.add_argument("--out", default=os.path.join("data", "cache", "dataset.json"))
    e.set_defaults(func=cmd_extract)

    t = sub.add_parser("train", help="train a model")
    t.add_argument("--cache", default=os.path.join("data", "cache", "dataset.json"))
    t.add_argument("--out", default=os.path.join("runs", "model"))
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--d-model", dest="d_model", type=int, default=128)
    t.add_argument("--n-layers", dest="n_layers", type=int, default=2)
    t.add_argument("--n-heads", dest="n_heads", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.25)
    t.add_argument("--epochs", type=int, default=400)
    t.add_argument("--patience", type=int, default=80)
    t.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--label-smoothing", dest="label_smoothing", type=float, default=0.08)
    t.add_argument("--prev-dropout", dest="prev_dropout", type=float, default=0.10)
    t.add_argument("--template-dropout", dest="template_dropout", type=float, default=0.30)
    t.add_argument("--no-template", action="store_true", help="train the baseline model without motif conditioning")
    t.add_argument("--target-mode", dest="target_mode", default="pitch",
                   choices=["pitch", "delta"],
                   help="pitch: predict absolute relative pitch (default, existing models); "
                        "delta: predict the interval from the previous melody note (Step A)")
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--log-every", dest="log_every", type=int, default=25)
    t.add_argument("--save-every", dest="save_every", type=int, default=0,
                   help="also save a checkpoint every N epochs (diagnostics)")
    t.add_argument("--select-metric", dest="select_metric", default="token_acc",
                   choices=["token_acc", "val_loss", "none"],
                   help="early-stopping / best-weight criterion (val_loss stops far too early)")
    t.add_argument("--keep-last", dest="keep_last", action="store_true",
                   help="keep the last-epoch weights instead of the best-scoring ones "
                        "(use with --select-metric none for a fixed epoch budget)")
    t.add_argument("--split-file", dest="split_file", default="data/splits/split_seed42.json")
    t.add_argument("--train-on-all", dest="train_on_all", action="store_true",
                   help="train on train+val songs (final model); test songs stay held out")
    t.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="generation-metric evaluation on held-out songs")
    ev.add_argument("--cache", default=os.path.join("data", "cache", "dataset.json"))
    ev.add_argument("--checkpoints", nargs="+", required=True)
    ev.add_argument("--out-dir", dest="out_dir", default=os.path.join("reports"))
    ev.add_argument("--split-seed", dest="split_seed", type=int, default=42)
    ev.add_argument("--modes", default="free,motif,plan")
    ev.add_argument("--protocols", default="continuation,free")
    ev.add_argument("--seeds", type=int, default=3)
    ev.add_argument("--no-calibrate", dest="no_calibrate", action="store_true")
    ev.add_argument("--threads", type=int, default=4)
    ev.add_argument("--calib-candidates", dest="calib_candidates", default="0,0.75,1.5,2.5,4")
    ev.add_argument("--target-mode", dest="target_mode", default="auto",
                    choices=["auto", "pitch", "delta"])
    ev.add_argument("--teacher-forced", dest="teacher_forced", action="store_true")
    ev.add_argument("--title", default="Trance melody model — evaluation")
    ev.set_defaults(func=cmd_evaluate)

    g = sub.add_parser("generate", help="generate a MIDI melody")
    g.add_argument("--version", default=None,
                   help="registered delivery version (see --list-versions); "
                        "defaults to v2 when no --checkpoint is given")
    g.add_argument("--list-versions", dest="list_versions", action="store_true",
                   help="print the registered versions and exit")
    g.add_argument("--checkpoint", nargs="+", default=None)
    g.add_argument("--delta-checkpoint", dest="delta_checkpoint", nargs="+", default=None,
                   help="interval-model checkpoints; when given with --mix, the two "
                        "models are blended in logit space (mix=0 pitch .. mix=1 interval)")
    g.add_argument("--mix", type=float, default=None,
                   help="hybrid mixing weight in [0,1] (requires --delta-checkpoint)")
    g.add_argument("--out", default=None,
                   help="output .mid path (required unless --list-versions is used)")
    g.add_argument("--key", default="Amin")
    g.add_argument("--bpm", type=int, default=140)
    g.add_argument("--progression", default="i,VI,III,VII")
    g.add_argument("--rhythm", default="12112112")
    g.add_argument("--bars", type=int, default=8)
    g.add_argument("--anchor", type=int, default=None)
    g.add_argument("--mode", default=None, choices=["free", "motif", "plan", "plan_root"])
    g.add_argument("--variation", type=float, default=0.0)
    g.add_argument("--w-chord", dest="w_chord", type=float, default=None,
                   help="chord-tone logit weight; omit to auto-calibrate to the 50-90%% contract")
    g.add_argument("--w-template", dest="w_template", type=float, default=None)
    g.add_argument("--w-repeat", dest="w_repeat", type=float, default=None,
                   help="penalty on repeating the previous melody note (corpus dup rate ≈ 0.22)")
    g.add_argument("--w-leap", dest="w_leap", type=float, default=None,
                   help="penalty on intervals wider than 5 semitones (fixes big leaps; "
                        "0 lets the model's own interval prior shape the melody)")
    g.add_argument("--w-range", dest="w_range", type=float, default=None,
                   help="penalty on drifting far from the song register")
    g.add_argument("--w-bass", dest="w_bass", type=float, default=None,
                   help="pull each bar's register toward the bass/pedal contour "
                        "(0.8 works well with --w-leap 0)")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--greedy", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--takes", type=int, default=1)
    g.add_argument("--no-pedal", dest="no_pedal", action="store_true")
    g.add_argument("--pedal-offset", dest="pedal_offset", type=int, default=15)
    g.add_argument("--chord-track", dest="chord_track", action="store_true")
    g.add_argument("--show-roll", dest="show_roll", action="store_true")
    g.add_argument("--stats-json", dest="stats_json", default=None)
    g.add_argument("--threads", type=int, default=4)
    g.set_defaults(func=cmd_generate)

    sp = sub.add_parser("split", help="write the deterministic song-level split")
    sp.add_argument("--cache", default=os.path.join("data", "cache", "dataset.json"))
    sp.add_argument("--out", default=os.path.join("data", "splits", "split_seed42.json"))
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_split)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
