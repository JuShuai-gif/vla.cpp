# robot_runtime 接入验证（2026-11-04）

本轮完成 Python policy API 和 Piper 单臂后端接入；没有计算内核或低精度优化。
两个模型均使用既有 MUSA F16 权重/F32 激活/F32 累加/FlashAttention 配置。
硬件、版本基线和库校验值见 `provenance.json`；环境与完整验收范围见 [环境与验收报告](环境与验收报告.md)。

## 动作对照

- `pruned-api.json/full-api.json`：原 Python 3.10 环境直接调用新 API。
- `python312/pruned-api.json/python312/full-api.json`：最终 Python 3.12 独立环境，经 robot_runtime 适配层调用 API。
- 四次检查相对原入口保存动作的最大绝对差均为 **0**，输出形状均为 50×7；每模型使用一组固定合成 RGB 图与状态均值，不代表真实任务精度验收。
- 每次 3 次预热，预热恢复噪声 RNG；首个预测对照 seed=0 参考，第二次预测确认使用新噪声。
- 参考文件位于相邻记录 `../2026-11-04_模型加载与图像预处理优化/entry-B2-{pruned,full}.actions.json`。
- 原始图像在两个模型的 `gguf/validation/{top,wrist}-480x640.png`，状态来自各自 `model.stats.json` 的 `observation.state.mean`。

## 回放与软件检查

`pruned-replay.json/full-replay.json` 为 robot-replay 正式入口，每模型 3 次预热 + 3 次预测。
少量时延只作接入冒烟记录，不是优化前后性能对比或稳定性能基准；期间有 CPU 离线测试负载。
`offline-tests.log`：803 项 Python 测试，39 项可选后端/fixture 跳过，其余通过。
资源检查、部署配置和环境 freeze 随本记录保存。日志保留原输出；生成的 NPZ/NPY 为本轮合成观测和预测数组，随记录保存；模型权重仍为外部文件。

## 复现

在 robot_runtime 中执行文档 `docs/VLA_CPP.md` 的 robot-replay 命令。动作对照在 vla.cpp 根目录执行：

```bash
../robot_runtime/.venv/bin/python tests/py/check_policy_integration.py \
  --config ../robot_runtime/configs/policies/vla_pruned.json \
  --robot-config ../robot_runtime/configs/vla_piper.json \
  --reference 优化记录/2026-11-04_模型加载与图像预处理优化/entry-B2-pruned.actions.json \
  --output-dir validation/policy-api-pruned
```

完整模型将 `pruned` 换为 `full`。当前机器配置优先于记录快照；不要把历史设备绑定或执行开关恢复为真机授权。

## 尚未验收

当前没有连接 Piper 或相机，未验证 USB 设备枚举、真实 CAN 收发、图像采集、动作闭环或成功率。
训练单位、零位、任务提示词和当前机械臂标定待核实。两个 source_contract 均保持 verified=false，live 执行被拒绝。
详细 API 契约见 ../../docs/POLICY_API.md；本轮未继续执行低精度探索方案。
