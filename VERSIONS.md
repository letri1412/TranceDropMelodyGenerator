# 版本记录（可回滚）— TranceDropMelodyGenerator

两个版本共用同一套 checkpoint 与代码，**不重训、不覆盖**；切换只改「用哪个版本生成」。
版本注册表在 `src/trance_lora/versions.py`，命令行用 `--version` 选择。

## v2（当前默认）

```bat
trance.cmd generate --version v2 --key Amin --progression i,VI,III,VII --rhythm 12112112 --out out\v2.mid
```

* 模型：pitch（`runs/final_motif_s*`）+ interval（`runs/stepA_final_motif_s*`）**logit 混合，mix=0.35**
* 解码：`plan_root`，`w_chord` 自动标定（**和弦内音契约 50–90%**），`w_template 2.0`，`w_repeat 1.2`，`w_leap 0.0`，`w_range 0.2`，`w_bass 0.8`
* 样例：`out/demo/`（10 条，由 `scripts/build_release.py --version v2` 生成）
* 特点：平均音程 2.89（语料 3.00）、每小节音域 5.09（语料 5.30）、低音跟随 +0.87，
  保住了 pitch 模型的和弦内音（0.856）与混合带来的低音跟随

## v1（上一版，保留回滚）

```bat
trance.cmd generate --version v1 --key Amin --progression i,VI,III,VII --rhythm 12112112 --out out\v1.mid
```

* 模型：仅 pitch（`runs/final_motif_s*`）
* 解码：`plan_root`，`w_chord` 自动标定，`w_template 2.0`，`w_repeat 1.2`，`w_leap 0.4`，`w_range 0.3`，`w_bass 1.5`
* 样例：`out/demo_v1/`（10 条，冻结快照）
* 特点：更「跳」、抽样方差更大；多抽几次偶尔能出更有人味的旋律

## 回滚 / 复现

```bat
rem 查看版本
trance.cmd generate --list-versions

rem 用 v1 生成（无需记 checkpoint 路径）
trance.cmd generate --version v1 --key Amin --progression i,VI,III,VII --out out\rollback.mid

rem 完整重建某版本的 10 条样例并自检（可复现，字节一致）
python scripts/build_release.py --version v1 --out out\demo_v1_check --check
python scripts/build_release.py --version v2 --out out\demo --check
```

验证记录（2026-11-22）：
- 用注册表重建 v1 → 与冻结的 `out/demo_v1/` **10/10 MIDI SHA256 完全一致**，回滚确定、可复现。
- v1 与 v2 的 10 条样例 **10/10 全部不同**，两个版本确实是两种输出。
- 两版都通过自检（`docs/output_check_v1.json` / `output_check_v2.json`）；v2 在 **50–90%** 新契约下重建：和弦内音 0.726 / 动机 0.486 / 调内 1.0。
- demo 层面旋律形状：v1 平均音程 2.33、每小节音域 4.35、低音跟随 0.58；
  **v2 平均音程 2.68、每小节音域 4.99、低音跟随 0.59、大跳率 0.017**（v1 为 0.002）——v2 更活。
