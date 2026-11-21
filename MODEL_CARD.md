# Model Card — TranceDropMelodyGenerator v2

## 概览
给定**调性 + BPM + 和弦进行 + 16 分节奏型**，生成 8 小节 Trance drop 主旋律（MIDI，含旋律轨 + 低音踏板层）。
两套交付版本：**v1** = 绝对音高模型；**v2** = v1 与音程（interval）模型的 logit 混合（mix=0.35，默认）。

## 模型
| 版本 | 架构 | 参数量 | 解码 |
|---|---|---|---|
| v1 | 因果 Transformer（8 输入通道 + 动机模板），6 层/96 维 | 0.327M × 3 seeds | plan_root，w_leap 0.4 / w_bass 1.5 |
| v2 | 同上 + 音程模型，logit 混合 mix=0.35 | 0.327M + 0.342M，各 3 seeds | plan_root，w_leap 0 / w_bass 0.8 / w_range 0.2 |

## 训练数据
102 对自创 Trance lead/chord MIDI（作者原创，**不随本仓库发布**；85% 8 小节 / 15% 16 小节，全部 140 BPM，11 个自然小调）。
train/val/test 按歌曲划分（seed 42），test 11 首从未参与训练。

## 指标（test 44 窗口，3 模型集成 × 3 种子）
| 指标 | 语料 | v2 |
|---|---|---|
| 调内率 | 1.000 | 1.000 |
| 和弦内音率（契约 50–90%） | 0.676 | 0.726 |
| 平均音程 | 3.00 | 2.68 |
| 每小节音域 | 5.30 | 5.0 |
| 跟随低音（bar-mean 相关） | +0.497 | +0.59 |
| 大跳率（>7 半音） | 0.090 | 0.017 |

## 用途与限制
* 用途：Trance 风格的 lead 旋律草稿、动机发展、MIDI 素材生成；
* 限制：只认 140 BPM 训练分布与小调/三和弦；不做音频渲染；不做多轨编曲；
* 已知偏差：生成比语料更平滑（大跳偏少），可通过 `--w-leap 0` 与 `--freedom` 调节。

## 许可与引用
代码 Apache-2.0；权重见 `LICENSE_WEIGHTS.md`（OpenRAIL-M + 禁止竞争性蒸馏）。
作者：letri1412。

输出水印：每个生成的 MIDI 元数据中包含 `letri1412 + 工具版本 + canary`，便于溯源。

Canary: `TRANCE-LORA-V2-CANARY-20261122-7f3a9c`。
