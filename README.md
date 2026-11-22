---
license: other
license_name: Apache-2.0 (code) / OpenRAIL-M (weights)
tags:
- trance
- midi
- music-generation
- symbolic-music
- pytorch
---

# TranceDropMelodyGenerator

## 在线地址

* GitHub: https://github.com/letri1412/TranceDropMelodyGenerator
* ModelScope（魔搭）: https://www.modelscope.cn/models/letri1412/TranceDropMelodyGenerator


> 用小模型生成 Trance drop 主旋律 MIDI：给定 **调性 + BPM + 和弦进行 + 16 分节奏型**，输出 8 小节「主旋律 + 低音踏板」双声部 MIDI。
> 纯 CPU 可训练、可推理；单模型 0.33M 参数，12 epoch ≈ 70 秒。

作者：**letri1412**　|　代码许可：Apache-2.0　|　权重许可：OpenRAIL-M（含禁止竞争性蒸馏条款）

---

## 目录

1. [项目简介](#1-项目简介)
2. [亮点](#2-亮点)
3. [效果指标](#3-效果指标)
4. [安装](#4-安装)
5. [快速开始](#5-快速开始)
6. [命令行详解](#6-命令行详解)
7. [输入怎么填](#7-输入怎么填)
8. [输出格式](#8-输出格式)
9. [版本与可复现](#9-版本与可复现)
10. [抽卡与自由度](#10-抽卡与自由度)
11. [训练自己的数据](#11-训练自己的数据)
12. [评估](#12-评估)
13. [仓库结构](#13-仓库结构)
14. [FAQ](#14-faq)
15. [许可与引用](#15-许可与引用)

---

## 1. 项目简介

这是一个**专门为 Trance drop 段落**设计的主旋律生成器。它不生成音频、不做编曲，只输出 **MIDI 旋律**，
方便你丢进 FL Studio / Ableton / Cubase 继续做音色与编曲。

**它解决的问题**：Trance drop 的 lead 旋律通常建立在「和弦进行 + 固定节奏型 + 简单重复动机」上，
但手工写 8 小节又很费时间。这个工具把「调性/和弦/节奏」作为硬条件输入，直接生成符合规范的主旋律草稿。

**能力边界（重要）**：

| 做 | 不做 |
|---|---|
| 8 小节（或 16 小节）Trance drop 主旋律 | 不渲染音频、不生成音色 |
| 主旋律 + 低音踏板（根音）双声部 | 不做鼓/贝斯/铺底等多轨编曲 |
| 调性、和弦进行、16 分节奏型条件化 | 不保证「每一条都好听」——它是草稿生成器，需要你抽卡与筛选 |
| MIDI 文件输出（DAW 可直接导入） | 不支持任意风格（只面向 140 BPM 小调 Trance drop 训练分布） |

## 2. 亮点

* **小**：单模型 0.33M 参数，CPU 上 8 小节生成 ≈ 2–5 秒；
* **可控**：调内率 100%，和弦内音率自动标定到 50–90% 契约，低音严格跟随和弦根音；
* **双声部**：一条 lead 轨同时包含主旋律与低音踏板，导进 DAW 直接分层；
* **两个版本**：v1（绝对音高模型）与 v2（音高+音程混合，默认），可随时回滚；
* **可抽卡**：`--takes/--seed` 批量、`lottery.py` 指标预筛、`--freedom` 自由度旋钮；
* **可复现**：`build_release.py --version v1/v2` 可逐字节重建示例集。

## 3. 效果指标

test 集 11 首（训练时从未见过）/ 44 个 8 小节窗口，3 模型集成 × 3 采样种子：

| 指标 | 语料参考 | v2（默认） | v1 |
|---|---|---|---|
| 调内率 | 1.000 | **1.000** | 1.000 |
| 和弦内音率（契约 50–90%） | 0.676 | **0.726** | 0.823 |
| 平均音程（半音） | 3.00 | 2.68 | 2.33 |
| 每小节音域 | 5.30 | ~5.0 | 4.35 |
| 跟随低音（小节均值相关） | +0.497 | **+0.59** | +0.58 |
| 大跳率（>7 半音） | 0.090 | 0.017 | 0.002 |
| 动机一致度 | 0.535 | 0.486 | 0.460 |

> v2 比 v1 更「活」（音程/音域更接近语料），v1 更平滑稳定。示例在 `demos/v2/`、`demos/v1/`。

## 在魔搭（ModelScope）加载

模型仓库根目录已包含平台要求的文件：

```
configuration.json   平台元数据（framework=pytorch, pipeline=dummy）
config.json          架构超参（d_model / n_layers / n_heads / 词表）
model.pt             顶层主模型（v2 pitch，示例）
model_interval.pt    顶层音程模型（v2 interval，示例）
MODEL_STRUCTURE.md   结构说明与文件对应关系
example_inference.py 最小推理示例
runs/                完整权重（每个模型 3 个种子，供集成）
```

```bat
pip install -r requirements.txt
python example_inference.py
```

## 4. 安装

要求：**Python 3.10+**（开发与测试用 3.15）、Windows/Linux/macOS 均可，**不需要 GPU**。

```bat
rem 1) 进入项目目录，建虚拟环境
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

rem 2) 自检（可选）：列出发布版本
trance.cmd generate --list-versions
```

Linux / macOS：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=_bootstrap:src .venv/bin/python -m trance_lora generate --list-versions
```

依赖只有三个：`torch`（CPU 版即可）、`numpy`、`mido`。

> 如果默认源装不到适合的 torch，用官方 CPU 轮子：

```bat
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Linux / macOS：`pip install torch --index-url https://download.pytorch.org/whl/cpu`，启动器用
`PYTHONPATH=_bootstrap:src .venv/bin/python -m trance_lora ...`（`trance.cmd` 仅 Windows 用）。

## 5. 快速开始

```bat
rem 生成一条 v2（默认）8 小节旋律：Amin，i-VI-III-VII，节奏型 12112112
trance.cmd generate --version v2 --key Amin --progression i,VI,III,VII --rhythm 12112112 --out out\a.mid

rem 一次抽 8 条，不同种子（文件名自动加 _00.._07）
trance.cmd generate --version v2 --key Cmin --progression i,VI,III,VII --rhythm 12112112 --takes 8 --seed 100 --out out\take.mid

rem 用抽卡工具：一次 8 条，自动计算指标并按「新颖度」排序
python scripts\lottery.py --family mix --mix 0.5 --takes 8 --key Amin --progression i,VI,III,VII
```

生成后用 DAW 直接导入 `out\a.mid`：第 1 轨是 **主旋律（高音）**，同轨低音是**踏板（和弦根音）**。

## 6. 命令行详解

全部参数的原样 `--help` 输出见 `docs/CLI.md`。下面是常用说明。

### 6.1 `generate` — 生成旋律

| 参数 | 默认 | 说明 |
|---|---|---|
| `--version` | `v2`（不传 `--checkpoint` 时） | 版本：`v1` / `v2` |
| `--list-versions` | – | 打印版本表后退出 |
| `--checkpoint ...` | – | 自定义权重路径（可多个，集成） |
| `--delta-checkpoint ...` | – | 音程模型权重（配合 `--mix` 做混合解码） |
| `--mix` | `0.35`（v2） | 混合权重：0=纯音高模型，1=纯音程模型 |
| `--out` | 必填 | 输出 MIDI 路径；配合 `--takes` 时自动加序号 |
| `--key` | `Amin` | 调性，如 `Amin` `C#min` `F#min` |
| `--bpm` | `140` | 速度 |
| `--progression` | `i,VI,III,VII` | 和弦进行，语法见 §7 |
| `--rhythm` | `12112112` | 16 分节奏型，语法见 §7 |
| `--bars` | `8` | 小节数（训练分布是 8 与 16） |
| `--anchor` | 自动 | 旋律音区锚点（一般不用传） |
| `--mode` | `plan_root` | 动机/移调模式：`free` `motif` `plan` `plan_root` |
| `--variation` | `0.0` | 变奏强度（0–1） |
| `--w-chord` | 自动 | 和弦内音偏置；省略=自动标定到 50–90% |
| `--w-template` | `2.0` | 动机模板偏置（越大越贴动机） |
| `--w-repeat` | `1.2` | 同音重复惩罚（越大越不容易连续同音） |
| `--w-leap` | `0.0` | 大跳惩罚（v2 默认关；v1 默认 0.4） |
| `--w-range` | `0.2` | 音区偏移惩罚 |
| `--w-bass` | `0.8` | 低音跟随偏置（每小节旋律音区拉向低音线） |
| `--temperature` | `1.0` | 采样温度（越大越随机） |
| `--greedy` | 关 | 贪心解码（稳定但没有抽卡） |
| `--seed` | `0` | 随机种子 |
| `--takes` | `1` | 一次生成几条（文件名自动加 `_00`…） |
| `--no-pedal` | 关 | 不输出低音踏板层 |
| `--pedal-offset` | `15` | 踏板音相对锚点的偏移（半音） |
| `--chord-track` | 关 | 额外输出一条和弦参考轨 |
| `--show-roll` | 关 | 终端打印钢琴卷帘 |
| `--stats-json` | – | 把统计写到 JSON |
| `--threads` | `4` | CPU 线程数 |

### 6.2 `train` — 训练

```bat
python -m trance_lora train --cache data/cache/dataset.json --out runs\my --epochs 12 ^
    --keep-last --select-metric none --d-model 96 --threads 4
```

关键参数：`--target-mode pitch|delta`（音高/音程目标）、`--no-template`（无动机条件基线）、
`--train-on-all`（train+val 全量训练，测试集仍留出）、`--keep-last`（保留最后一轮，推荐配合固定 epochs）。

### 6.3 `evaluate` — 评估

```bat
python -m trance_lora evaluate --checkpoints runs\final_motif_s0\model.pt runs\final_motif_s1\model.pt ^
    --modes free,motif,plan_root --protocols continuation,free --seeds 3 --teacher-forced --out-dir reports\eval
```

### 6.4 `validate` / `extract` / `split` — 数据处理

```bat
python -m trance_lora validate --root <你的数据集目录> --out data\validate.json
python -m trance_lora extract  --root <你的数据集目录> --out data\cache\dataset.json
python -m trance_lora split    --cache data\cache\dataset.json --out data\splits\split_seed42.json --seed 42
```

### 6.5 工具脚本

| 脚本 | 用途 |
|---|---|
| `scripts/lottery.py` | 一次抽 N 条并按指标排序预筛（`--family pitch/delta/mix`、`--rank novelty/corpus`） |
| `scripts/pathA_simple.py` | 用已有旋律动机发展新旋律；`--freedom 0..1` 控制自由度 |
| `scripts/rhythm_quantize.py` | 把任意旋律量化到指定节奏型（`--pattern 12112112` 或 `--pattern from:<歌名>`） |
| `scripts/adapt_external_midi.py` | 分析外部 MIDI / 把外部动机接进本模型发展 |
| `scripts/build_release.py` | 可复现地重建示例集：`--version v1/v2 --out out\demo --check` |
| `scripts/check_outputs.py` | 对生成结果做规范自检（调内/和弦/节奏/踏板/层间隔） |

## 7. 输入怎么填

### 7.1 调性 `--key`

`Amin` `Cmin` `F#min` `G#min` …（小调；训练集全部是自然小调）。

### 7.2 和弦进行 `--progression`

支持罗马数字（自然小调 7 个三和弦）：`i` `ii°` `III` `iv` `v` `VI` `VII`。

```bat
--progression i,VI,III,VII          rem 4 个和弦：每个占 2 小节
--progression i,i,VI,VI,iv,iv,v,v   rem 8 个和弦：每个占 1 小节（重复写=延长）
```

> 注意：语料 89% 是 **2 小节换一次和弦**。写 4 个不同和弦 = 每 2 小节换；
> 写重复和弦（如 `i,i` 或 `i,i,i,i` 等价）会让同一和弦保持更久，听感更「静」。

### 7.3 节奏型 `--rhythm`

16 分音符网格上的逐格标记，长度 8 或 16：

* `1` = 踏板（低音）占据这一格
* `2` = 旋律音符占据这一格
* `0` = 休止

```bat
--rhythm 12112112                                  rem 经典 trance 型（8 格，自动重复成 16 格）
--rhythm 1121212111212121                          rem 16 格完整型
--rhythm 2121212121212121                          rem 连续 8 分音符型
```

节奏型直接决定**旋律密度**：`2` 越多音符越密。

## 8. 输出格式

* MIDI Type-1，480 PPQ，16 分音符 = 120 ticks；
* 第 1 轨（Conductor）：tempo / 4-4 拍号 / 水印元事件；
* 第 2 轨（Lead）：**主旋律（高音）+ 低音踏板（根音）**，两者音区自动分开 ≥ 6 半音；
* 每个音长度 = 1 个 16 分音符；
* 调内 100%，和弦内音 50–90%（自动标定）。

## 9. 版本与可复现

| 版本 | 模型 | 解码参数 | 示例 |
|---|---|---|---|
| **v2（默认）** | pitch + interval 混合（mix=0.35） | w_leap 0.0 / w_bass 0.8 / w_range 0.2 | `demos/v2/` |
| v1 | 纯 pitch | w_leap 0.4 / w_bass 1.5 / w_range 0.3 | `demos/v1/` |

```bat
trance.cmd generate --list-versions
trance.cmd generate --version v1 --key Amin --progression i,VI,III,VII --out out\v1.mid
python scripts\build_release.py --version v2 --out out\demo --check
```

`build_release.py` 用固定种子与选择规则重建 10 条示例（逐字节可复现），v1/v2 权重都随仓库提供。

## 10. 抽卡与自由度

生成器本身是随机采样，**同一条输入多抽几次会得到不同旋律**，这是特性：

| 旋钮 | 范围 | 作用 |
|---|---|---|
| `--takes N` + `--seed` | – | 一次出 N 条，不同种子 |
| `scripts/lottery.py` | – | 一次抽 N 条并输出每条的合法性/新颖度/动机/低音跟随，自动排序 |
| `--mix` | 0–1 | 0=稳（纯音高模型），1=野（纯音程模型），默认 0.35 |
| `--temperature` | 0.5–1.2 | 越大越随机 |
| `--freedom`（pathA_simple） | 0–1 | 动机发展的自由度：0=严格移调，1=完全自由 |

推荐工作流：

1. 要稳：`--version v2 --mix 0.2`；要新鲜：`--mix 0.6` + `--temperature 1.05`；
2. 批量抽 8–16 条 → `lottery.py` 按指标预筛 → 只试听前 3–5 条；
3. 觉得太机械：降低 `--w-repeat`、提高 `--freedom`（Path A）或换 `--mix`。

## 11. 训练自己的数据

数据集**不随仓库发布**（作者原创作品）。格式与复现步骤见 `data/README_DATA.md`：

```
<root>/<song>__<bpm>__<key><maj|min>__<section>__lead.mid
<root>/<song>__<bpm>__<key><maj|min>__<section>__chord.mid
```

```bat
trance.cmd validate --root <root> --out data\validate.json
trance.cmd extract  --root <root> --out data\cache\dataset.json
trance.cmd split    --cache data\cache\dataset.json --out data\splits\split_seed42.json
python -m trance_lora train --cache data\cache\dataset.json --out runs\my --epochs 12 --keep-last --select-metric none
```

CPU 上单模型 12 epoch ≈ 70 秒（约 100 首 8 小节的数据量）。更多数据量可把 `--d-model` 提到 96→128、epochs 提到 20–30。

## 12. 评估

`evaluate` 输出（test 集）：

| 指标 | 含义 |
|---|---|
| `in_key` | 调内音比例（契约 100%） |
| `chord_tone` | 和弦内音比例（契约 50–90%） |
| `motif_consistency` | 动机在后续小节被移调重现的程度 |
| `interval_*` / `contour_r` | 音程分布、逐格音高相关 |
| `quality_corr_melody_bass` | 旋律与低音踏板的跟随度 |
| `quality_bar_pitch_range` | 每小节音域（不要过大） |
| `quality_jump7_rate` | 大跳率（>7 半音） |
| `distinct_pitches` | 每窗口不同音高数 |

自检（生成规范）：`scripts/check_outputs.py`，检查 8 小节/16 分对齐/单音/调内/和弦/踏板根音/层间隔/节奏型一致。

## 13. 仓库结构

```
src/trance_lora/       核心库：理论、数据、特征、模型、采样、评估、CLI、版本表、水印
scripts/               训练/评估/发布/抽卡/节奏量化/动机发展工具
runs/                  权重（v1: pitch×3；v2: pitch×3 + interval×3）
demos/v1, demos/v2     示例 MIDI（各 10 条）
demos/pathA/           动机发展示例
docs/CLI.md            命令行参考（全部参数原样输出）
data/README_DATA.md    数据集格式（数据集不随发布）
LICENSE                代码许可（Apache-2.0）
LICENSE_WEIGHTS.md     权重许可（OpenRAIL-M + 禁止竞争性蒸馏）
MODEL_CARD.md          模型卡
trance.cmd / trance.ps1  启动器
```

## 14. FAQ

**Q：为什么生成旋律比语料「平滑」？**
A：训练目标 + 解码惩罚倾向保守，大跳率 0.017（语料 0.090）。想要更跳跃：`--mix 0.6 --w-leap 0`。

**Q：和弦内音率为什么不是 100%？**
A：强制 100% 会让旋律变成琶音、失去人味。默认自动标定到 50–90%，可用 `--w-chord 2.0` 手动提高。

**Q：能生成 16 小节吗？**
A：可以 `--bars 16`，但训练集只有 2 首 16 小节，效果弱于 8 小节。

**Q：能改 BPM 吗？**
A：可以 `--bpm 138`，但训练集全是 140，偏离越多越可能「不像 trance」。

**Q：FL Studio 导入后只有一轨怎么办？**
A：主旋律和踏板在**同一轨**（音区分离 ≥6 半音），进 FL 后可用「Split by channel」或按音高切开。
需要单独和弦轨时加 `--chord-track`。

**Q：生成的 MIDI 带水印吗？**
A：带。每个文件的 `text`/`marker`/`copyright` 元事件与每轨 `text` 都写入
`TranceDropMelodyGenerator 版本 | 作者 | canary | 仓库`，不影响播放。

**Q：可以商用吗？**
A：**由本模型生成的音乐作品**你可以自由使用/发布（含商业发布），但**模型权重本身**的商用需按
`LICENSE_WEIGHTS.md` 获得许可；不得用本模型或其输出训练竞争性 Trance lead 生成模型。

## 15. 许可与引用

* **代码**：Apache-2.0（见 `LICENSE`）；
* **权重**：OpenRAIL-M + 禁止竞争性蒸馏附加条款（见 `LICENSE_WEIGHTS.md`），含 canary；
* **训练集**：作者原创，不随仓库发布；
* **作者**：letri1412；
* **canary**：`TRANCE-LORA-V2-CANARY-20261122-7f3a9c`（同时写入每个生成 MIDI 的元数据）。

若用于研究或二次开发，请标注仓库地址与作者；欢迎 issue 反馈生成质量与 bug。
