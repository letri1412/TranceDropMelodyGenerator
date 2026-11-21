# 命令行参考（全部参数）— TranceDropMelodyGenerator

下面是每个子命令 `--help` 的原样输出。

## `validate`

```text
usage: trance_lora validate [-h] [--root ROOT] [--out OUT]

options:
  -h, --help   show this help message and exit
  --root ROOT
  --out OUT
```

## `extract`

```text
usage: trance_lora extract [-h] [--root ROOT] [--out OUT]

options:
  -h, --help   show this help message and exit
  --root ROOT
  --out OUT
```

## `train`

```text
usage: trance_lora train [-h] [--cache CACHE] [--out OUT] [--seed SEED]
                         [--d-model D_MODEL] [--n-layers N_LAYERS]
                         [--n-heads N_HEADS] [--dropout DROPOUT]
                         [--epochs EPOCHS] [--patience PATIENCE]
                         [--batch-size BATCH_SIZE] [--lr LR]
                         [--label-smoothing LABEL_SMOOTHING]
                         [--prev-dropout PREV_DROPOUT]
                         [--template-dropout TEMPLATE_DROPOUT] [--no-template]
                         [--target-mode {pitch,delta}] [--threads THREADS]
                         [--log-every LOG_EVERY] [--save-every SAVE_EVERY]
                         [--select-metric {token_acc,val_loss,none}]
                         [--keep-last] [--split-file SPLIT_FILE]
                         [--train-on-all]

options:
  -h, --help            show this help message and exit
  --cache CACHE
  --out OUT
  --seed SEED
  --d-model D_MODEL
  --n-layers N_LAYERS
  --n-heads N_HEADS
  --dropout DROPOUT
  --epochs EPOCHS
  --patience PATIENCE
  --batch-size BATCH_SIZE
  --lr LR
  --label-smoothing LABEL_SMOOTHING
  --prev-dropout PREV_DROPOUT
  --template-dropout TEMPLATE_DROPOUT
  --no-template         train the baseline model without motif conditioning
  --target-mode {pitch,delta}
                        pitch: predict absolute relative pitch (default,
                        existing models); delta: predict the interval from the
                        previous melody note (Step A)
  --threads THREADS
  --log-every LOG_EVERY
  --save-every SAVE_EVERY
                        also save a checkpoint every N epochs (diagnostics)
  --select-metric {token_acc,val_loss,none}
                        early-stopping / best-weight criterion (val_loss stops
                        far too early)
  --keep-last           keep the last-epoch weights instead of the best-
                        scoring ones (use with --select-metric none for a
                        fixed epoch budget)
  --split-file SPLIT_FILE
  --train-on-all        train on train+val songs (final model); test songs
                        stay held out
```

## `evaluate`

```text
usage: trance_lora evaluate [-h] [--cache CACHE]
                            --checkpoints CHECKPOINTS [CHECKPOINTS ...]
                            [--out-dir OUT_DIR] [--split-seed SPLIT_SEED]
                            [--modes MODES] [--protocols PROTOCOLS]
                            [--seeds SEEDS] [--no-calibrate]
                            [--threads THREADS]
                            [--calib-candidates CALIB_CANDIDATES]
                            [--target-mode {auto,pitch,delta}]
                            [--teacher-forced] [--title TITLE]

options:
  -h, --help            show this help message and exit
  --cache CACHE
  --checkpoints CHECKPOINTS [CHECKPOINTS ...]
  --out-dir OUT_DIR
  --split-seed SPLIT_SEED
  --modes MODES
  --protocols PROTOCOLS
  --seeds SEEDS
  --no-calibrate
  --threads THREADS
  --calib-candidates CALIB_CANDIDATES
  --target-mode {auto,pitch,delta}
  --teacher-forced
  --title TITLE
```

## `generate`

```text
usage: trance_lora generate [-h] [--version VERSION] [--list-versions]
                            [--checkpoint CHECKPOINT [CHECKPOINT ...]]
                            [--delta-checkpoint DELTA_CHECKPOINT [DELTA_CHECKPOINT ...]]
                            [--mix MIX] [--out OUT] [--key KEY] [--bpm BPM]
                            [--progression PROGRESSION] [--rhythm RHYTHM]
                            [--bars BARS] [--anchor ANCHOR]
                            [--mode {free,motif,plan,plan_root}]
                            [--variation VARIATION] [--w-chord W_CHORD]
                            [--w-template W_TEMPLATE] [--w-repeat W_REPEAT]
                            [--w-leap W_LEAP] [--w-range W_RANGE]
                            [--w-bass W_BASS] [--temperature TEMPERATURE]
                            [--greedy] [--seed SEED] [--takes TAKES]
                            [--no-pedal] [--pedal-offset PEDAL_OFFSET]
                            [--chord-track] [--show-roll]
                            [--stats-json STATS_JSON] [--threads THREADS]

options:
  -h, --help            show this help message and exit
  --version VERSION     registered delivery version (see --list-versions);
                        defaults to v2 when no --checkpoint is given
  --list-versions       print the registered versions and exit
  --checkpoint CHECKPOINT [CHECKPOINT ...]
  --delta-checkpoint DELTA_CHECKPOINT [DELTA_CHECKPOINT ...]
                        interval-model checkpoints; when given with --mix, the
                        two models are blended in logit space (mix=0 pitch ..
                        mix=1 interval)
  --mix MIX             hybrid mixing weight in [0,1] (requires --delta-
                        checkpoint)
  --out OUT             output .mid path (required unless --list-versions is
                        used)
  --key KEY
  --bpm BPM
  --progression PROGRESSION
  --rhythm RHYTHM
  --bars BARS
  --anchor ANCHOR
  --mode {free,motif,plan,plan_root}
  --variation VARIATION
  --w-chord W_CHORD     chord-tone logit weight; omit to auto-calibrate to the
                        50-90% contract
  --w-template W_TEMPLATE
  --w-repeat W_REPEAT   penalty on repeating the previous melody note (corpus
                        dup rate ≈ 0.22)
  --w-leap W_LEAP       penalty on intervals wider than 5 semitones (fixes big
                        leaps; 0 lets the model's own interval prior shape the
                        melody)
  --w-range W_RANGE     penalty on drifting far from the song register
  --w-bass W_BASS       pull each bar's register toward the bass/pedal contour
                        (0.8 works well with --w-leap 0)
  --temperature TEMPERATURE
  --greedy
  --seed SEED
  --takes TAKES
  --no-pedal
  --pedal-offset PEDAL_OFFSET
  --chord-track
  --show-roll
  --stats-json STATS_JSON
  --threads THREADS
```

## `split`

```text
usage: trance_lora split [-h] [--cache CACHE] [--out OUT] [--seed SEED]

options:
  -h, --help     show this help message and exit
  --cache CACHE
  --out OUT
  --seed SEED
```

## 常用示例

```bat
trance.cmd generate --list-versions
trance.cmd generate --version v2 --key Amin --progression i,VI,III,VII --rhythm 12112112 --out out\a.mid
trance.cmd generate --version v1 --key Cmin --progression i,VI,III,VII --takes 4 --seed 100 --out out\take.mid
python scripts\lottery.py --family mix --mix 0.5 --takes 8 --key Amin --progression i,VI,III,VII
python scripts\pathA_simple.py --freedom 0.3 --seeds 2
python scripts\build_release.py --version v2 --out out\demo --check
```
