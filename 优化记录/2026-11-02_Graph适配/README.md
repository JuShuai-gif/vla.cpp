# 2026-11-02 Graph 适配

起点：`47a75fb`。目标是在两个 π0.5 模型上实际捕获、重放 MUSA Graph，并保留原有精度与图像/去噪配置。上轮记录中的 Graph 故障在本轮继续定位；原始失败证据不改写。

## 修复结果与范围

- 修复刚实例化的 executable 又对同一张图调用 GraphExecUpdate 导致的数值错误。首次实例化后直接 launch；只有已有 executable 重新捕获时才 update。
- 动作输入/输出投影保留 FP32，复用 GGML 已有的 FP32 向量乘内核，每次处理至多 8 个 token 列，避免进入不可捕获的 muBLAS FP32 同步路径。没有把这两层改为 F16。
- 移除此前对 FA/BF16 图的整体禁用；仍保留对其余不兼容 FP32 GEMM 的捕获保护。
- 构建脚本默认 Graph ON、FA ON。推理入口新增可选 `--warmup 2`，在同一个加载实例中完成普通执行、捕获、重放，并报告重复输出差值。
- 当前验证配置为 **F16 权重、FA 开启，F32 或 F16 激活**。完整模型的视觉图与语言/动作主图均重放；裁剪模型的语言/动作主图重放，保留注意力裁剪统计的视觉图仍按普通流执行。
- F32 权重模式、FA 关闭或其他含不兼容 FP32 GEMM 的图可能回退。没有宣称任意模型、任意精度组合都已全部 Graph 化。

## 根因定位与失败尝试

### 1. 捕获/重放数值错误

上轮 `ctest-graph-guard.log` 中 FA 链误差为参考峰值的 32%–43%，BF16 视觉链误差约 7.95 倍参考峰值。

本轮先保留 FP32 GEMM 保护，恢复 FA/BF16 图捕获，仅把：

```cpp
if (instance == nullptr) instantiate();
if (update_required) update();
```

改为：

```cpp
if (instance == nullptr) instantiate();
else if (update_required) update();
```

结果：15/15 测试通过。继续加入同一计算图改变 scale 参数、9 次执行的重新捕获测试，也通过。证据支持问题出在当前 MUSA 路径对“刚创建 executable 的同图更新”的处理；未声称已经定位驱动内部哪一行错误。不同图重新捕获后的正常 update 保留并验证。

`test-no-initial-update.log`、`test-recapture-before.log` 对应上述两个步骤；后一个名称表示加入 FP32 投影路径之前，不表示该测试失败。

### 2. FP32 GEMM 捕获失效

msys 独立探针 API 序列显示：FP32 GEMM 捕获区内有两次 `musaMemcpyAsync`，随后 `musaStreamSynchronize`，捕获返回 901。F16 输入/F32 输出 GEMM 捕获区只有 kernel launch，可正确重放。同步调用是已观测到的阻塞点；搬运的是标量为结合接口行为的推断，未读取闭源 muBLAS 实现。

继续尝试：
- device pointer mode：第一个 FP32 Sgemm 捕获仍失败。其后共享失败流的结果不作独立能力结论。
- muBLASLt FP32：k=32、m=1024、n=50 普通执行正确，捕获仍报 901。第二形状被失败流状态影响，没有完成独立验证。
- 因此选择复用已有 FP32 GPU 内核，处理两处小投影；没有通过降低它们的精度绕过问题。

小投影分派限制为二维、连续、F32 输入/输出、K 为偶数、K 或输出维度不超过 32。更一般矩阵继续原路径；新增 CPU 对照覆盖实际 32→1024 和 1024→32 投影及 7/50/51 列，包含最后不足 8 列的尾部。

## 正式端到端对照

所有模型保留两个图像、chunk=50、原始 3/10 去噪步。performance 模式，4 CPU 线程，预热 3 次、正式 5 次，以下为不含加载的 wall time 中位数。正式测量不带 msys 或 host-submit 诊断。

| 配置 | 裁剪模型 ms | 完整模型 ms |
| --- | ---: | ---: |
| 47a75fb 原实现，Graph 关闭，F32 激活 | 743.179 | 1514.137 |
| 新实现，Graph 关闭，F32 激活 | 743.235 | 1517.548 |
| 新实现，Graph 开启，F32 激活 | **747.986** | **1534.180** |
| 新实现，Graph 关闭，F16 激活 | 739.433 | 1505.854 |
| 新实现，Graph 开启，F16 激活 | **741.599** | **1514.217** |

Graph 对新版关闭 Graph 的 F32 激活路径约慢 0.64% / 1.10%，F16 激活约慢 0.29% / 0.56%。**本轮修复的是可用性和主机提交开销，没有端到端加速收益。**

同一新版开启/关闭 Graph 的动作逐元素一致；每组包括首次执行/捕获/重放在内的固定输入重复漂移为 0。对旧 MUSA F32 参考的峰值相对误差：
- F32 激活：0.09042% / 0.19086%。
- F16 激活：0.04936% / 0.34124%。

FP32 投影换用向量内核改变浮点累加顺序，与 47a75fb 不保证逐位相同；因此分别检查“新实现对旧参考误差”和“新实现 Graph 开关一致性”。

初次捕获也有额外开销：完整模型第二次调用约 2658 ms，稳态约 1534 ms。短命进程或只推理一次不能依靠 Graph 获益。

## 主机提交时间与真实重放证据

独立诊断设置 `VLA_PI05_HOST_SPLIT=1`，记录主图异步提交返回与 GPU 同步完成之间的时间。预热 3 次、正式 5 次：

| 模型 | 主图节点 | Graph 关闭，提交 ms | Graph 开启，提交 ms | 开启后的 GPU 完成 ms |
| --- | ---: | ---: | ---: | ---: |
| 裁剪 | 2003 | 250.0 | 0.9 | 456.0 |
| 完整 | 5622 | 925.2 | 2.5 | 1090.9 |

“提交时间”包含 CPU/驱动等待，不等于纯 CPU busy 时间或 CPU 利用率。GPU 运算与主机提交原本存在重叠，因此不能把这几百毫秒直接从端到端时间中扣除。Graph 已减少提交耗时，而设备执行仍占主要时间。

官方 msys 对完整模型运行 4 次（2 次预热、2 次测量）：
- 2 次 BeginCapture、2 次 EndCapture、2 次 Instantiate。
- **6 次 musaGraphLaunch**：第二次调用捕获并启动两张图，第三/第四次调用各重放两张图。
- 最后一次 Instantiate 后 **没有 musaLaunchKernel API 调用**；后续通过 GraphLaunch 执行，另有数据搬运和同步。
- 没有首次实例化后的 GraphExecUpdate。
- msys 汇总用于确认调用路径，不替代未带分析器的正式延迟。上轮 kernel 时长汇总异常没有用于本轮速度结论。

详见 `summary-host-submit.json`、`summary-graph-trace.json` 与 msys 原始统计。

## 使用

无需重新转换 GGUF，两类 checkpoint/tokenizer 继续共用入口。

```bash
cd /home/skysi/code/vla.cpp
export VLA_LIBRARY="$PWD/build-musa-official/libvla.so"
unset GGML_CUDA_DISABLE_GRAPHS
unset GGML_MUSA_DISABLE_GRAPHS

MODEL_DIR=/home/skysi/VLA/models/cup_cube_joint_061_a800_50k_20261022
.venv-converter/bin/python scripts/run_pi05.py \
  --model "$MODEL_DIR/gguf/model.gguf" \
  --image top=/path/to/top.png --image wrist=/path/to/wrist.png \
  --text 'pick up the cube' --state '0,0,0,0,0,0,0' \
  --weight-dtype f16 --act-dtype f32 --flash-attn --warmup 2 \
  --seed 0 --output actions.json
```

替换 MODEL_DIR 即可运行裁剪模型。示例图片与 state 需替换为实际观测。需要 F16 激活时改为 `--act-dtype f16`。

常驻应用复用同一个模型实例：首次普通执行，第二次捕获并执行，此后重放。输入形状改变时需要重新预热/捕获。单次 CLI 默认 `--warmup 0`，避免强制增加两次推理成本；`--warmup 2` 用于检查真实重放。

关闭 Graph 仍可设置 `GGML_CUDA_DISABLE_GRAPHS=1`；开启应移除变量，设置成 0 也会被视为关闭。

构建默认已变为 Graph/FA ON。此次在原 `build-musa-official` 增量构建；47a75fb 的动态库保存于 `/home/skysi/VLA/graph-baseline-47a75fb`。对照脚本显式设置 LD_LIBRARY_PATH，防止 baseline 的 RUNPATH 错误加载新后端。

## 验证与复现记录

- `test-projections.log`：15/15 CTest，无 GPU 跳过；含 FA/BF16 Graph 重放、9 次参数改变/重新捕获、小 FP32 投影 CPU 对照。
- `patch-validation.json`：fresh pinned llama 补丁和重复执行幂等检查通过。
- `upgrade-patch-validation.json`：从 47a75fb 已打补丁的依赖升级，结果与 fresh/current 完全一致，避免旧的 FA/BF16 禁用残留。
- `run_graph_comparison.py` 保存逐次 timing、误差、环境、命令与库哈希。拒绝覆盖已有同标签实验。
- `check_full_entry.py`：两类 tokenizer、43/42 token、480×640 PNG 预处理、相机顺序、反归一化动作；每次加载内预热 2 次后取重放输出，对照 CPU F32。
- `check_changing_inputs.py`：6/9 token 各连续 3 次，多轮切换图像/state/noise；检查普通执行、捕获、重放以及重建后的输出一致性。
- 原始动作数组保存在各实验目录，并记录哈希；大 msys 原始文件在 external-artifacts.json 中标注位置和哈希。
- 全部记录统一在本目录，Mac 副本随后同步。

完整入口与变化输入检查的最终结果见附表。数值回归阈值仍为 max_abs / reference_peak < 0.005；没有放宽阈值。未执行原始 Python 完整策略对照或真实机器人任务验收。

## 最终完整入口与变化输入回归

| 模型 | 激活 | token | 重放对 CPU F32 峰值相对误差 | 首次/捕获/重放漂移 |
| --- | --- | ---: | ---: | ---: |
| 裁剪 | f32 | 43 | 0.10470% | 0 |
| 裁剪 | f16 | 43 | 0.14146% | 0 |
| 完整 | f32 | 42 | 0.12191% | 0 |
| 完整 | f16 | 42 | 0.14455% | 0 |

两个模型均完成 6→6→6→9→9→9→6→6→6→9→9→9 token 的同实例执行，图像、state、noise 随两种输入一起变化。10 次重复对照的最大差值均为 0；新输入产生不同动作。

Graph 开关的数组精确对照见 summary-graph-equivalence.json，4 组均为 max_abs=0。新实现对旧 CPU/MUSA 参考不要求逐位一致，阈值未变。

记录随本次实现提交；manifest.json 保存归档文件 SHA-256（不含自身），actions.npy 留在目录但不入 Git，哈希列入 external-artifacts.json。源码/文本空白检查排除原始 log 和 patch，保留工具输出原样。
