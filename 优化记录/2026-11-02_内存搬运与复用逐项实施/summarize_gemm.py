from pathlib import Path
import json
d=Path(__file__).resolve().parent;records=json.loads((d/'gemm-shapes/summary.json').read_text())
lines=['# GEMM 形状研究与实现','',
'采用 muBLAS 列主序命名：C[M,N] = A[K,M]^T × B[K,N]，不采用 FlashRT 文档把 token 数叫 M 的写法。实际图清单见 GEMM形状清单.json；Attention 内部另见 Attention内部GEMM形状.json，包含 transpose、leading dimension 和 batch stride。清单来自上轮最终同配置图；本轮只消除搬运及折叠批次，权重/有效 token 形状未改变。','',
'## 候选与约束','',
'- 动作：四类主要 F16 GEMM 的 N=50，测试补齐 56/64/128；3 步模型每类 54 次，10 步每类 180 次。不能把去噪的不同 step 批处理，因为下一步依赖上一步动作。',
'- 前缀：测试基准提示长度对应的 N=278/518，以及完整入口的 N=315/554，补齐到 16/64 的倍数。GateUp 已合并为 M=32768，QKV 已合并为 M=2560。补齐必须计入额外输入转换、计算和输出还原，不能只看 tile 整齐程度。',
'- 视觉：比较 M=4304 或 K=4304 补齐到 4352/4480；另测共享权重的 batch=2、N=256 展平为 batch=1、N=512。剪枝模型逐相机执行并在末层计算裁剪分数，本轮不跨独立视觉图拼接它的输入。',
'- F32 action in/out 与裁剪末层 Attention、im2col patch embedding 保留原路径。它们列在实际清单里，但本轮 F16 常量权重 padding 探针不模拟动态 im2col 的权重侧数据，不能据此宣称这些形状已优化。','',
'## 测量方法','',
'probe_gemm_shapes.cu 使用 F16 输入乘法、F32 累加/输出，每候选 3 次预热、Graph 捕获，再做 3×20 次重放。计时包含 F32→F16 输入打包、零填充、GEMM 和有效输出还原。权重补零在初始化完成，稳态不计初始化成本；使用合成输入和同一权重反复重放，不能把单算子收益直接加成整模型收益。所有候选有效输出均与同形状基线逐值一致。','',
'首版通用 crop kernel 使用逐元素 64 位除法，严重拖慢 padding 候选；保留在 gemm-shapes-naive-crop。复核后改为 N-only 的 D2D 连续复制，以及 M-padding 的二维 float4 kernel，最终结论只采用 gemm-shapes 的复测结果。','',
'|M×N×K / batch|候选 M×N×K / batch|基线 ms|候选 ms|变化|','|---|---|---:|---:|---:|']
for r in records:
 a=r['measurements'];baseline=a[0]['mean_ms']
 for x in a[1:]:
  dst=[x['gemm_m'],x['gemm_n'],x['gemm_k'],x['gemm_batch']]
  lines.append('|'+ '×'.join(map(str,r['shape'][:3]))+f' / {r["shape"][3]}|'+'×'.join(map(str,dst[:3]))+f' / {dst[3]}|{baseline:.4f}|{x["mean_ms"]:.4f}|{100*(x["mean_ms"]/baseline-1):+.1f}%|')
lines+=['','## 结论与落地','',
'已测试的 padding 全部变慢：N=50→56/64 约慢 6.5%–10.9%；前缀 N 补齐也没有正收益；4304 补齐没有正收益。因此保留原有效形状，不增加模型计算量或长期 padded 权重副本。','',
'共享权重的双相机展平在五类视觉 GEMM 上降低约 5.7%–9.1%。已实现 VLA_MUSA_FLATTEN_GEMM=1：仅在 F16 计算/F32 输出、共享 A、B 批次连续、目标连续时，把 ne11×ne12 作为单次 GEMM 的 N，不增加数据拷贝或改变输出布局；其余情况继续走原路径。完整模型的 27 层视觉可用，剪枝模型的独立相机图不触发。已增加 musa_flatten_gemm CTest；整模型结果见后续逐项计时及最终复测。','',
'FlashRT 的参考价值是根据小 token 数挑选 GEMM/布局并融合消费者。其 skinny 文档使用 RTX sm_120a、FP8 和最多 16 行的专用 Attention，本任务为 M1000、50 行、F16，不能直接套用这些数值、内核或吞吐结论。已阅读 /home/skysi/code/FlashRT/docs/pi05_decoder_skinny.md，本轮没有采用 FP4/FP8 量化。']
(d/'06_GEMM形状研究.md').write_text('\n'.join(lines)+'\n')
print('GEMM study saved',len(records),'shape groups')
