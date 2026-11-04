# 执行时推理耗时日志（2026-11-04）

恢复中断的“执行过程中显示模型推理时间”任务。量化探索保持暂停。

## 改动

- `robot_runtime` 在 vla_cpp 正式预测完成后打印一行耗时；full/pruned 共用。
- 复用 policy 已有 `native_call_ms`、`preprocess_ms` 和 `postprocess_ms`，同时显示 runtime 的
  `predict_ms` 与 `observation_to_prediction_ms`；预热不计入正式预测编号。
- 原生调用包含 ctypes 参数封装、native 前处理／推理与结果复制，不是纯 GPU kernel 时间。
  观测至预测时间从读取相机开始，到预测完成为止，不包含动作下发与执行等待。
- 遥测保留 `backend_native_call_ms`；缺失／无效值显示 `n/a`，终端断开不导致预测失败。
- vla.cpp 计算代码无需修改，公共 API 文档补充计时口径，并修正已过时的 verified 执行门禁说明。
- runtime 的本轮四个代码／测试文件补丁存于 `runtime-timing.patch`。原有相机与动作契约修复保留。

## 验证

23 项相关离线回归测试通过：终端输出 5、遥测 12、vla_cpp 桥接 6。
命令与代码基线／文件 SHA256 在 `provenance.json`，单元测试原始输出未保存。

两个真实模型均通过 runtime API 用已保存的合成 RGB／状态观测回放：每模型 3 次预热 + 3 次预测，
输出均为有限的 50×7 动作，计时字段有限且非负，整次预测时间覆盖各阶段总和。
没有读取真机相机、使能机械臂或下发动作；没有做闭环效果或新数值精度对照验收。

硬件 M1000、驱动 5.1.1-M1000、performance 电源模式；MUSA SDK 版本未记录。
沿用策略的 F16 权重、F32 激活／累加及 FlashAttention。输入文件 SHA256 和完整配置解析值在 JSON。

本轮均值（ms，仅验证计时输出，不是优化前后性能对比）：

| 模型 | native_call_ms | preprocess_ms | postprocess_ms | predict_ms |
| --- | ---: | ---: | ---: | ---: |
| pruned | 462.37 | 6.58 | 0.07 | 469.08 |
| full | 757.11 | 198.95 | 0.08 | 956.21 |

完整模型本轮 Python 前处理约 199ms，按实际测量保留，未在此日志任务中优化。
回放观测来自内存，日志里的 observation_to_prediction_ms 不代表真实相机读取延迟。
没有性能基线对比，不声称本轮提升速度。

## 复现与原始结果

在 `/home/skysi/code/robot_runtime` 执行：

```bash
source scripts/env.sh --config configs/vla_piper_live.json
.venv/bin/python ../vla.cpp/优化记录/2026-11-04_推理耗时日志/verify_timing.py pruned
.venv/bin/python ../vla.cpp/优化记录/2026-11-04_推理耗时日志/verify_timing.py full
```

`verify_timing.py` 使用模型回放锁，不调用机械臂或相机。
`pruned.json` / `full.json` 保存逐次计时，`pruned.log` / `full.log` 保存原始终端输出。
实际启动仍使用既有 `run_robot.sh --config configs/vla_piper_live.json --profile full|pruned`，
每次正式预测自动显示日志，无需新增参数。
