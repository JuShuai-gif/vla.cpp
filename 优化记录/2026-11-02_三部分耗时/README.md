# 两个 π0.5 模型的三部分耗时

起点提交：`23d7052`。本轮只增加诊断计时，没有实施新的性能优化。统计为当前 Graph/FA 路径的稳态 wall time，原始模型和 GGUF 未改变。

## 结果

条件：M1000、performance 模式、4 CPU 线程；F16 权重 / F32 激活、FlashAttention 与 Graph 开启；双路 224×224 合成图、6 个固定文本 token、固定 state/noise、chunk=50；预热 3 次、测量 5 次，取中位数，排除加载和预热。

| 部分 | 裁剪模型（15 层视觉、3 步去噪） | 完整模型（27 层视觉、10 步去噪） |
| --- | ---: | ---: |
| 视觉编码与图像 token 准备 | **292.01 ms** | **442.50 ms** |
| VLM 多模态前缀与 KV 生成 | **300.33 ms** | **487.41 ms** |
| 动作去噪（全部步数） | **157.00 ms** | **597.56 ms** |
| 动作去噪平均每步（总耗时 / 步数） | 52.33 ms | 59.76 ms |
| 分段诊断整次 predict wall time | 751.46 ms | 1529.04 ms |
| 未分段完整 Graph 整次 predict wall time | 747.06 ms | 1531.09 ms |

分段与完整 Graph 总延迟差为 **+0.59% / -0.13%**，不应将这点差异解释成优化收益；它包含执行顺序/额外同步影响及测量波动。两种模式输出逐元素一致，max_abs=0；包括预热在内的重复输出漂移也是 0。

各部分分别取中位数，另有 token 查表/上传、位置和噪声设置、动作读回与反归一化等少量开销，三项相加不要求严格等于整次 wall time。

## 三部分的边界

1. **视觉部分**：沿用已有 ms_vision，包含 predict 内图像处理、视觉 tower 与投影、输出搬运；裁剪模型还包含注意力统计、CPU 选 token 和重新上传。每图保留 136 / 256 个 token。不是只量 ViT GEMM。
2. **VLM 前缀部分**：对图像 token 和文本 token 共同进行 Gemma 前缀计算，生成后续专家读取的逐层 K/V。前缀长度为 2×136+6=278 / 2×256+6=518，不能理解成只计算 6 个文本 token。
3. **动作去噪部分**：保留原始 3 / 10 步，包括动作输入/输出投影、专家层、残差与 Euler 积分，得到完整 50×7 动作 chunk。每步值是总时间除步数，没有单独声称每一步都一样快。

不包含模型加载、GGUF 转换、PNG 文件读取或外部 Python tokenizer/图像预处理。固定时间步的 Ada 调制在加载时预计算。此处是统一合成基准，不等于真实机器人任务输入分布。

按三部分中位数之和粗略计算，裁剪模型视觉/前缀/动作约为 39%/40%/21%，完整模型约为 29%/32%/39%。

## 计时方法与影响控制

原实现仅有视觉和合并主图两项统计；前缀与第一步专家的节点可能交错，不能把一个 CPU 时间戳直接当作边界。

新增仅诊断时启用的 `VLA_PI05_SPLIT_PHASES=1` / `bench_pi05.py --split-phases`：
- 先在同一完整 GGML 图中展开所有前缀 K/V 依赖，再展开动作图。
- 整张图仍统一分配内存、管理中间张量生命周期；没有复制权重或单独加载模型。
- 使用与现有 backend_fallback 相同的图切片方式，保留父图 use counts，分别执行前缀和动作两个范围。
- 每段调用同步的 graph_compute，以 wall clock 记录完成时间；两个子图仍能捕获与重放。裁剪模型分段日志有 2 个 capture 标记，完整模型有 3 个（另含视觉）。
- 因此这是“分段 Graph 的 wall time 诊断”，不是从原始单张 Graph 无扰动抽取的纯 GPU kernel 时间。以未分段对照检查影响，未将 msys 不可靠的 kernel 汇总百分比用于推算。
- 开关默认关闭，正常推理继续原图执行。仅支持当前普通缓存前缀路径，遇到原位 KV / fq_fast 诊断请求会明确拒绝。

每模型均运行 whole 和 split 两个独立进程；同一 GGUF、输入、精度、步数与电源设置。每组保存 8 次迭代日志、5 次正式采样、输出数组和参考误差。两模型 split 对旧 MUSA F32 参考的峰值相对误差为 0.09042% / 0.19086%，小于原有 0.5% 数值回归阈值。

## 复现

```bash
cd /home/skysi/code/vla.cpp
export VLA_LIBRARY="$PWD/build-musa-official/libvla.so"
unset GGML_CUDA_DISABLE_GRAPHS
unset GGML_MUSA_DISABLE_GRAPHS
.venv-converter/bin/python scripts/bench_pi05.py \
  --model /home/skysi/VLA/models/pi05_cube_merge_cksize50_step3_0929/gguf/model.gguf \
  --out-dir /tmp/pi05-three-phases-new-run \
  --weight-dtype f16 --act-dtype f32 --flash-attn \
  --warmup 3 --runs 5 --split-phases
```

换成完整模型目录即可。标准输出/错误日志中的 `VLA_PI05_PHASES` JSON 给出 vision_ms、prefix_ms、denoise_ms、steps；report.json 保存整体延迟与诊断标记。图形状或开关应在进程开始前确定。

本轮完整测量脚本为 `run_phases.py`，会拒绝覆盖已有同名实验；`summary.json` 保存中位数、全部正式分段样本、完整图对照和动作一致性检查。

## 验证与归档

- CTest 15/15 通过，无 GPU 跳过。
- 两个整模型的 whole/split 数值、重复性与参考误差检查通过。
- `environment.json` 记录起点版本、电源与当前动态库哈希；源码改动与本记录一起提交。
- `manifest.json` 保存归档文件的 SHA-256（不含自身）；小型 actions.npy 保留在原实验目录，哈希列入 external-artifacts.json，不加入 Git。
- 原始日志保留工具输出，包括换行与空白；Mac 的中文记录目录同步副本。

未在本轮重新评估 F16 激活、真实任务输入或机器人成功率；这些数字不能当作任务验收结果。
