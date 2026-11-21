# 训练数据格式（不随发布）

本仓库**不包含**训练集。若要复现训练，请准备：

```
<root>/<song>__<bpm>__<key><maj|min>__<section>__lead.mid     # 旋律+踏板合并在一轨
<root>/<song>__<bpm>__<key><maj|min>__<section>__chord.mid    # 柱式和弦
```

* 16 分音符网格，整小节对齐，lead 轨单音、chord 轨 3–5 音；
* 你的数据必须是你有权使用的（本发布不含任何第三方 MIDI）；
* 放置后运行：

```bat
trance.cmd validate --root <root> --out data/validate.json
trance.cmd extract  --root <root> --out data/cache/dataset.json
trance.cmd train --cache data/cache/dataset.json --out runs/mymodel --epochs 12 --keep-last --select-metric none
```
