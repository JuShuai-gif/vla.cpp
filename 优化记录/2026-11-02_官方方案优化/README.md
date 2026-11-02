# 2026-11-02 官方方案优化

基于 [官方调研](../MUSA_M1000_PI05_优化调研_2026-10-30.md)，在现有 M1000 配套 SDK 上实施并逐阶段测量。起点提交：`309b1c74db4e38f7d0fb8e0089fe06cb3f62413f`。本目录随实现一起提交；最终实现版本为包含本记录的 Git 提交。

## 测试条件

- M1000 / aarch64；驱动包 5.1.1-M1000，SDK 包 5.1.1，muBLAS/Lt 1.12.0。
- 用户恢复 render/video 组权限后重新建立 SSH；所有正式 GPU 测量为 performance 模式。初始无权限/旧 SSH 会话跳过 GPU 的日志保留，不计作通过。
- 旧构建 `build-musa-m1000` 保留；新构建 `build-musa-official`。同一 GGUF，无重新量化、无减层、无改变原始去噪步数。
- 两张 224×224 合成图、固定 6 个 token、固定 state/noise、chunk=50；裁剪模型 15 层视觉、136 token/图、3 步，完整模型 27 层视觉、256 token/图、10 步。
- 每组预热 3 次、正式 5 次，报告包含视觉与动作推理的 wall time 中位数，不含加载、预热和分析器开销。每轮输出有限性、对参考误差和重复漂移同时检查。
- 此处 F32/F16 激活指模型逻辑激活类型。F16 权重/F32 激活的 muBLAS 路径仍会将 GEMM 输入转换为 F16，累加/输出为 F32。原生 F16 路径部分 GEMM 使用 COMPUTE_16F，归约、RoPE、残差/积分保留原有高精度路径。
- FA 将 K/V 存为 F16，Q 与注意力归约保持 F32；这不属于 FP4 量化。

## 每一步耗时

下表均为 F16 权重、F32 激活。括号中的目录是实际测量阶段，不能用当前源码的开关伪装重跑历史版本。

| 阶段 | 裁剪模型 ms | 完整模型 ms | 方案与结论 |
| --- | ---: | ---: | --- |
| 旧版同条件基线（old-f32） | 880.731 | 2022.080 | 与本轮一致的电源模式和预热 |
| 新构建关闭 Graph/FA（control-f32） | 880.881 | 2020.769 | 排除单纯编译开关带来的收益 |
| 固定步长批量 GEMM（fixed-control-f32） | 827.175 | 1765.194 | MQA 广播使用 strideA=0，避免构造设备指针数组；原输出完全一致 |
| 接通普通 FA（fixed-fa-f32） | 807.344 | 1740.591 | 修复 host 能力判断；保留裁剪模型视觉统计 |
| Q 直接读 stride，K/V 提前转 F16 缓存（kv-fa-f32） | 745.476 | 1580.044 | 前缀每层只转换一次，后续拼接流量减半；与前一阶段动作一致 |
| 视觉 FA 直接读 Q view（final-fa-f32） | **744.515** | **1529.243** | K/V 转换同时完成布局复制，移除额外连续化；裁剪模型视觉不走 FA |

相对旧版同条件基线，最终 F32 激活延迟降低 **15.47% / 24.37%**，吞吐加速比约 **1.18× / 1.32×**；约 **1.34 / 0.654 个动作 chunk/秒**。不是动作 token/s，也不等于闭环机器人控制频率。

K/V 阶段同时实施“Q view”和“前缀 F16 缓存/拼接”，没有进一步拆开独立测量，不能分别归因。仍有每步 CONCAT，没有声称已经完成原位 KV 写入或 GPU top-k/gather。

### 原生 F16 激活

| 模型 | 旧版 F16 激活 ms | 最终 F16 激活 ms | 降幅 | 对 MUSA F32 参考的 max_abs / peak |
| --- | ---: | ---: | ---: | ---: |
| 裁剪模型 | 872.871 | 740.158 | 15.20% | 0.04942% |
| 完整模型 | 2003.350 | 1504.893 | 24.88% | 0.33921% |

F32 激活最终误差分别为 0.08586% / 0.19414%。所有正式成功组固定输入重复漂移为 0。0.5% 是数值回归阈值，不是机器人任务验收标准。F16 激活相对最终 F32 激活只快约 0.6% / 1.6%，完整模型误差更大，优先推荐 F32 激活。误差偶然变小不能推断低精度更准确。

## Graph：失败、回退和实际结论

1. 直接打开 Graph：两个整模型预热均退出 -6，MUSA 901，捕获被库调用失效。见 `graph-f32/*/run.log`。
2. 独立 muBLAS 探针：FP32 Sgemm、FP32 GemmEx、FP32 strided batch 均触发 901；F16 输入/F32 输出 GemmEx 可捕获/重放。非零输入探针清空输出后再次重放仍得到正确结果，不能仅凭全零输入判定正确。
3. 初步排除 FP32 GEMM 后，单元回归仍发现 BF16 视觉链和 FA 链重放误差超限（约 7.95 / 0.32–0.43 的 normalized error）。见 `ctest-graph-guard.log`。尚未定位这两类复合图误差的最终根因。
4. 初步保护下整模型 Graph 对照为 825.715 / 1784.415 ms，Graph+FA 为 815.784 / 1747.179 ms，未得到收益。普通 Graph 两模型日志无捕获标记；Graph+FA 仅完整模型出现一个 warmup-complete 标记，不能声称整条策略已 Graph 化。
5. 最终保护：含 FLASH_ATTN_EXT 或非 F16 MUL_MAT 的图回到普通流执行；兼容的小图仍可捕获。新增第三次执行覆盖重放，最终普通 CTest 15/15 通过。构建脚本默认仍 Graph OFF，FA 编译 ON。

因此本轮 **没有计入 Graph 加速**，也没有以放宽误差阈值使测试通过。最终测时使用 `GGML_CUDA_DISABLE_GRAPHS=1`；完整入口回归移除该变量以检查保护分支。该变量在此上游后端按是否存在判断，设置为 0 仍会禁用。

## 官方 msys 与 muBLASLt 检查

msys 1.7.2 在 KV 阶段完整模型采集，包含加载、1 次预热、1 次推理：

```bash
VLA_LIBRARY=$PWD/build-musa-official/libvla.so VLA_DEVICE=0 VLA_N_THREADS=4 \
GGML_CUDA_DISABLE_GRAPHS=1 msys profile -t musa \
  -o /home/skysi/VLA/perf-official-20261102/kv-fa-f32 \
  .venv-converter/bin/python scripts/bench_pi05.py \
  --model /home/skysi/VLA/models/cup_cube_joint_061_a800_50k_20261022/gguf/model.gguf \
  --out-dir 优化记录/2026-11-02_官方方案优化/profile-kv-fa-f32 \
  --weight-dtype f16 --act-dtype f32 --flash-attn --warmup 1 --runs 1
```

- 已确认实际执行 `flash_attn_tile<256,...>` 与 `flash_attn_tile<72,...>`，以及 F16 CONCAT；不是只打开配置却仍走未融合 Attention。
- 报告列出 12548 次 kernel launch，包括大量 F32→F16 转换和 720 次 F16 CONCAT。这些仍是后续优化候选。
- **工具计时存在不一致**：kernel 汇总约 27.58 秒，但两次推理 wall time 合计约 3.40 秒，多个无关 kernel 显示相同最大时长。未将这些 kernel 百分比当作可靠的端到端耗时分解，也不以其推算可节省毫秒数。原报告和统计完整保留，后续需要独立 event/分析器核对。
- 加载期 memcpy/malloc 包含在 API 汇总中，不能当成每次推理开销。

muBLASLt 探针用 FP16 A/B、FP32 compute、m=256、n=50、k=257/1024、4 MiB workspace、最多 8 个候选。每形状实际只返回一个成功 heuristic；F16 输出执行返回 8，F32 输出通过且 ones 输入误差 0。没有把“一个 heuristic”写成已完成算法调优，也未对全部模型 GEMM 形状作不支持推断。

源码核查发现现有 muDNN copy 开关仅替换“同类型且已连续”的 memcpy，并不覆盖这里主要的转置/类型转换，因此本轮没有盲目开启或宣称其收益。原位 KV、GPU 裁剪、完整形状 muBLASLt 调优及 epilogue 融合仍未实施；SDK 未升级。

## 使用

默认构建脚本已编译 FA，保留 Graph OFF。无需重新转换 GGUF，两类模型共用入口。

```bash
cd /home/skysi/code/vla.cpp
# 当前已验证的新构建（旧 build-musa-m1000 保留）：
export VLA_LIBRARY="$PWD/build-musa-official/libvla.so"
export GGML_CUDA_DISABLE_GRAPHS=1
MODEL_DIR=/home/skysi/VLA/models/cup_cube_joint_061_a800_50k_20261022
.venv-converter/bin/python scripts/run_pi05.py \
  --model "$MODEL_DIR/gguf/model.gguf" \
  --image top=/path/to/top.png --image wrist=/path/to/wrist.png \
  --text 'pick up the cube' --state '0,0,0,0,0,0,0' \
  --device 0 --weight-dtype f16 --act-dtype f32 --flash-attn \
  --seed 0 --output actions.json
```

替换 MODEL_DIR 即可运行裁剪模型；相机名按 checkpoint 内顺序排序。示例 state 应替换为真实观测；`--act-dtype f16` 是可选更低精度路径。省略 `--flash-attn` 保留未融合注意力。

独立实验构建命令：

```bash
BUILD_DIR="$PWD/build-musa-official" JOBS=4 bash scripts/build_musa.sh \
  -DFETCHCONTENT_SOURCE_DIR_LLAMA="$PWD/build-musa-deps/llama" \
  -DFETCHCONTENT_SOURCE_DIR_SENTENCEPIECE="$PWD/build-musa-m1000/_deps/sentencepiece-src" \
  -DGGML_MUSA_GRAPHS=ON -DGGML_CUDA_FA=ON
```

新部署通常直接用 `scripts/build_musa.sh` 的 Graph OFF 默认值即可。以上 Graph ON 是本轮对照构建，不是性能推荐。

## 验证与证据

- 最终 GPU/CPU CTest：15/15，无 GPU skip。覆盖 F32/F16/BF16、非对齐序列、mask、head_dim=72/256、双视图、融合、三次重复和 >4 GiB 权重分段。
- patch-validation.json：在未修改的 pinned llama `4da6337767f973e2b4d0797e5b323d77d8565e4a` 三个文件上应用补丁两次，幂等且与实际依赖完全一致。
- 完整入口与变化输入回归结果见下面的自动生成附表。真实图像采集/机器人任务不在这些合成输入回归的覆盖范围内。
- 未完成原始 Python 完整策略动作对照及真实机器人任务成功率验证。
- `summary-*.json` 对应各阶段；每组 `report.json` 保存 5 次测量、误差、运行参数；`command.json` 和 `run.log` 保留命令与原输出。
- 早期报告只有 libvla 哈希，KV 及之后增加 backend 哈希；`stage-fixed-source.patch`、`stage-kv-source.patch` 保存相对起点的阶段差异；最终修订见本 Git 提交，不能声称每次中间尝试均有独立源码提交。
- `environment-before.json` / `environment-after.json` 保存权限、电源、SDK 与库校验；`external-artifacts.json` 记录外部 msys 原始文件位置和哈希。
- 初次 sentencepiece 网络获取失败、旧 SSH GPU 跳过、FA host gate 拒绝、Graph 捕获失败及重放精度失败等日志均保留。

## 完整入口回归

输入为归档的两张 480×640 PNG，经真实预处理、状态归一化、SentencePiece/JSON tokenizer 和反归一化动作入口；对照此前 CPU F32 输出。这些图片是验证输入，不代表真实任务分布。

| 模型 | 激活 | token 数 | max_abs | error / CPU peak | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| 裁剪 | f32 | 43 | 0.089607 | 0.10524% | 通过 |
| 裁剪 | f16 | 43 | 0.117920 | 0.13850% | 通过 |
| 完整 | f32 | 42 | 0.108627 | 0.11688% | 通过 |
| 完整 | f16 | 42 | 0.133148 | 0.14327% | 通过 |

## 连续输入与长度变化

两个模型分别在同一个 resident 实例中执行 6→9→6→9 token，交替改变图像、state 和 noise。同一种输入重现时最大差值均为 0，不同输入的动作最大差值为 21.7648 / 11.6433。见 summary-changing-inputs.json。该检查覆盖模型图缓存重建/复用，不代表 Graph 捕获已启用。

manifest.json 记录本目录随 Git 归档文件的 SHA-256（不包含清单自身）。原始 actions.npy 留在同一实验目录并列入 external-artifacts.json，不随 Git 提交；完整入口动作已保存在 JSON。

最终性能组之后仅增加 pre-mp22 的 FA 能力保护及注释/记录修订；M1000 执行路径未改变。补丁检查、15 项 CTest、完整入口及变化输入回归在该修订后通过，库哈希见 environment-after.json。
