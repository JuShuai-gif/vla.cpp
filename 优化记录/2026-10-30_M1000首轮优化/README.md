# π0.5 / M1000 优化阶段与原始测量

测量日期：2026-10-30；归档日期：2026-11-02。

两张合成 224×224 图像、6 个固定文本 token、相同 state/noise，chunk=50；裁剪/完整模型分别保留 3/10 步去噪。下表是预热后 Python 调用 wall time 中位数，包含视觉与主推理，不含加载。单项耗时中位数相加不一定等于总耗时中位数。

**记录范围：已保存主要实验阶段，但不是每个代码编辑都有独立测速或历史 Git 提交。** QKV stride view 与行索引 GEGLU 是合并测量，不能拆分归因；已撤回实验不保证能用最终代码直接重现。历史报告未完整记录电源模式、温度、频率和每阶段源码 SHA，不能补造。当前默认开关也不能重现所有历史中间版本。

| 方案 / 原始阶段 | 裁剪模型 ms（次数） | 完整模型 ms（次数） |
| --- | ---: | ---: |
| F32 基线 `baseline-f32` | [1844.533（5）](perf-fusion/pi05_cube_merge_cksize50_step3_0929/baseline-f32/report.json) | [4106.170（5）](perf-fusion/cup_cube_joint_061_a800_50k_20261022/baseline-f32/report.json) |
| 早期 F16 权重路径 `baseline-f16` | [1011.761（5）](perf-fusion/pi05_cube_merge_cksize50_step3_0929/baseline-f16/report.json) | [2322.777（5）](perf-fusion/cup_cube_joint_061_a800_50k_20261022/baseline-f16/report.json) |
| F16 权重，F32 累加/输出，关闭融合 `final-f16-nofuse1` | [991.651（5）](perf-fusion/pi05_cube_merge_cksize50_step3_0929/final-f16-nofuse1/report.json) | [2288.611（5）](perf-fusion/cup_cube_joint_061_a800_50k_20261022/final-f16-nofuse1/report.json) |
| 增加 AdaRMS / 门控残差融合 `final-f16-nofuse0` | [987.979（5）](perf-fusion/pi05_cube_merge_cksize50_step3_0929/final-f16-nofuse0/report.json) | [2277.720（5）](perf-fusion/cup_cube_joint_061_a800_50k_20261022/final-f16-nofuse0/report.json) |
| 增加 QKV、Gate/Up 合并与直接 GEGLU `merged-f16` | [957.337（5）](perf-fusion/pi05_cube_merge_cksize50_step3_0929/merged-f16/report.json) | [2194.016（5）](perf-fusion/cup_cube_joint_061_a800_50k_20261022/merged-f16/report.json) |
| 原生 F16 激活，尚无最新布局优化 `native-f16-attn0` | [961.086（5）](perf-lowp/pi05_cube_merge_cksize50_step3_0929/native-f16-attn0/report.json) | [2158.337（5）](perf-lowp/cup_cube_joint_061_a800_50k_20261022/native-f16-attn0/report.json) |
| QKV stride view + 行索引 GEGLU，F32 激活 `layout-f32` | [907.842（5）](perf-lowp/pi05_cube_merge_cksize50_step3_0929/layout-f32/report.json) | [2073.754（5）](perf-lowp/cup_cube_joint_061_a800_50k_20261022/layout-f32/report.json) |
| 相同最新布局，F16 激活 `layout-f16` | [884.295（5）](perf-lowp/pi05_cube_merge_cksize50_step3_0929/layout-f16/report.json) | [2039.342（5）](perf-lowp/cup_cube_joint_061_a800_50k_20261022/layout-f16/report.json) |
| Attention F16 实验（退步，撤回） `attention-f16` | [1100.084（3）](perf-lowp/pi05_cube_merge_cksize50_step3_0929/attention-f16/report.json) | [2792.045（3）](perf-lowp/cup_cube_joint_061_a800_50k_20261022/attention-f16/report.json) |
| 另一低精度 Attention 实验（撤回） `native-f16-attn1` | 无已完成 report；不填补数值 | [2762.747（5）](perf-lowp/cup_cube_joint_061_a800_50k_20261022/native-f16-attn1/report.json) |
| 早期 BF16 权重对照 `baseline-bf16` | [1026.059（5）](perf-fusion/pi05_cube_merge_cksize50_step3_0929/baseline-bf16/report.json) | [2315.575（5）](perf-fusion/cup_cube_joint_061_a800_50k_20261022/baseline-bf16/report.json) |
| F32 权重，小算子融合对照 `final-f32-nofuse0` | [1832.615（5）](perf-fusion/pi05_cube_merge_cksize50_step3_0929/final-f32-nofuse0/report.json) | [4090.332（5）](perf-fusion/cup_cube_joint_061_a800_50k_20261022/final-f32-nofuse0/report.json) |
| F32 权重，合并投影对照 `merged-f32` | [1798.363（5）](perf-fusion/pi05_cube_merge_cksize50_step3_0929/merged-f32/report.json) | [4090.561（5）](perf-fusion/cup_cube_joint_061_a800_50k_20261022/merged-f32/report.json) |

## 对比方式

- `final-f16-nofuse1 → final-f16-nofuse0`：小算子融合；`final-f16-nofuse0 → merged-f16`：合并投影与相关数据流优化。
- `merged-f16 → native-f16-attn0`：激活精度实验；`merged-f16 → layout-f32`：同为 F32 激活的布局优化。
- `layout-f32 → layout-f16`：相同布局下的激活精度对照。不要把全部收益归因于 F16 激活。
- `attention-f16` 每个模型只有 3 次正式测量；其他表中完整报告为 5 次。以原始报告中的 warmup/runs 字段为准。
- BF16 早期对照中裁剪模型的 relative_peak 约为 1.44%，超过 0.5% 合成误差阈值，不能归为通过的优化路径。

## 原始证据与误差

每个 report 保留逐次总耗时、视觉/主推理耗时、中位数、P95、重复输出漂移和已测参考误差。runtime.json 保留当时写出的运行配置；早期报告字段较少，未补写成新的实验记录。manifest.json 记录来源与 SHA-256。

误差以各原始报告和 summary 中注明的参考为准；合成容差不是机器人验收标准。完整入口误差与最终 CTest 历史日志位于 validation-2026-10-30。模型权重、图像、完整 action 数组和构建目录未纳入 Git，仍保留在原机器。

## 验证状态

- 2026-10-30：历史记录为 CTest 14/14 通过，包含真实 MUSA 执行；两个模型有合成数值与完整入口检查。
- 2026-11-02 提交前：构建通过；CTest 13 项通过、1 项跳过。GPU 初始化报 CreatePlatform failed / initialization error，musa_ops 按退出码 77 跳过，不能描述为本轮 GPU 回归通过。日志见 validation-2026-11-02。
- 尚无完整原始 Python 策略动作对照与真实机器人验收。

## 尚未实施的官方文档优化

Graph、FlashAttention、进一步的 KV/RoPE 布局、muDNN copy、muBLASLt 调优均未产生新增优化结果。详见 [官方调研与实施顺序](../MUSA_M1000_PI05_优化调研_2026-10-30.md)；使用与验证命令见 [本地模型工作流](../方案与验证说明.md)。

更早的单次 CPU/MUSA 验证耗时及原始日志见 [首次整模型验证](../2026-10-30_首次整模型验证/README.md)，与本目录重复基准的计时口径不同。
