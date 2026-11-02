from pathlib import Path
import json,shlex
d=Path(__file__).resolve().parent;repo=d.parents[1]
perf=json.loads((d/'最终对比.json').read_text())
checks=json.loads((d/'validation-summary.json').read_text())
assert len(checks)==13 and all(x['returncode']==0 for x in checks)
lines=['# 内存搬运与复用逐项实施','',
'覆盖原定五项：视觉异步、视觉直写前缀、输入与常量复用、KV 尾部更新、内部打包/重排；另完成用户追加的 GEMM 形状研究。代码基线 000be2d，计划提交 f8533a4。各报告保存源码补丁、库 SHA256、配置和原始样本。记录只在服务器保存，不同步 Mac。','',
'## 最终结果','',
'每配置 30 次 A→B→B→A 正常计时，F16 权重、F32 激活，chunk50、3/10 去噪步、两路图像、三段 Graph；基线保留此前全部已验收优化。','',
'|模型|基线 ms|推荐 ms|减少 ms|改善|','|---|---:|---:|---:|---:|']
for x in perf:lines.append(f"|{x['model']}|{x['baseline']['mean_ms']:.3f}|{x['final']['mean_ms']:.3f}|{x['saved_ms']:.3f}|{x['saved_percent']:.2f}%|")
lines+=['','三部分诊断和 P95 见 [最终复测](05_最终交错复测.md)。异步模式普通 ms_vision 只代表提交区间；分段表全部另跑 split-phases，不能与正常延迟混比。','',
'## 逐条落实','',
'|项目|实现和证据|','|---|---|',
'|稳态定位|预热后复制方向/字节/API 等待/Graph 数量：[基线](01_稳态搬运基线.md)|',
'|视觉同步|同 stream 异步提交；早退/销毁等待；语言/噪声/回读使用持久 pinned，消除 pageable 隐式等待|',
'|视觉→前缀 D2D|输出绑定主图输入；encode_images 恢复地址；覆盖相机数量切换和预计算 embedding|',
'|输入与常量|分别实现/测试 pageable、pinned、mapped；语言/噪声复用容量且每次更新，位置/mask 按真实长度更新；推荐 pinned|',
'|KV 复用|重测只更新动作尾部，覆盖 3/10 步，最终组合启用|',
'|内部布局|视觉 Q/K/V 转换、打包合并；填充一次初始化；Gemma AV 直写目标布局，释放每 stream 5 MiB 输出 scratch|',
'|Gemma Q 转型|四种长度的 F16 K × F32 Q 均 NOT_SUPPORTED，保留 Q pack：[边界与失败依据](03_布局搬运实现与边界.md)|',
'|GEMM 形状|30 组形状、N/M/K 补齐、双相机展平、实际分派；展平实现后 20 样本 ABBA 不稳定，不纳入推荐：[研究](06_GEMM形状研究.md)|',
'',
'每项总耗时与三部分耗时见 [逐项测量](02_输入与边界逐项结果.md)，包括没有收益的方案。mapped 消除了图像 H2D，但未测到稳定延迟优势。展平微基准改善 5.7%–9.1%，整模型仅改善 0.142%、两轮符号相反，不能混为整模型收益。','',
'## 验收','',
'- 24 项 CTest 全通过，无 GPU 跳过。',
'- 14 组基础生命周期回归通过；最终再验两模型 pinned 主输入及 host-prune 回退。',
'- 两模型变图像、文本、状态、噪声，同语言桶长度变化，输出与上轮参考逐值一致，重复差异为 0。',
'- 两种 GGUF 经过完整入口：各测 F32/F16 激活、480×640 图像预处理、各自 tokenizer/相机顺序；四份输出与上轮验收逐值一致。',
'- 最终稳态 3 次 GraphLaunch，无重新实例化或动态分配，见 [搬运与隐式同步复测](04_搬运复测与隐式同步.md)。',
'- 构建与验收.json 校验最终源码、运行库和验收；压缩记录清单.json 校验原始日志/补丁；外部产物清单.json 记录服务器 trace/npy 的路径和 SHA256。',
'',
'## 使用与范围','',
'推荐配置是 selection.json。[使用命令](使用命令.md) 给出两模型完整入口命令。同一 scripts/run_pi05.py 和 scripts/convert_pi05_local.py 继续支持两类 checkpoint，本轮未量化或改变模型配置。',
'',
'新开关默认关闭，启用用值 1，关闭请 unset。HOST_INPUT 取 pageable/pinned/mapped；PIN_MAIN_INPUT 管理语言/噪声/回读的 pinned slab；完整变量名见 selection.json 和使用命令。VLA_MUSA_FLATTEN_GEMM 为未推荐的实验开关，VLA_MUSA_TRACE_GEMM 仅用于分派诊断。',
'',
'收益限固定形状稳态，不包含加载、首次分配/捕获、形状变化后的重建。direct/mapped 保留原 allocator 空间用于回退，不能按复制字节宣称峰值显存等量下降。本轮是本地数值/性能验收，未测机器人任务成功率，不宣称全局最优。']
(d/'README.md').write_text('\n'.join(lines)+'\n')
fence=chr(96)*3
cmdlines=['# 推荐配置与复现命令','',
'使用现有验证图片、各模型真实 state 均值，结果写 /tmp。替换图片/任务/state 后仍走同一入口。转换说明见 ../方案与验证说明.md。','']
for short,name in [('pruned','pi05_cube_merge_cksize50_step3_0929'),('full','cup_cube_joint_061_a800_50k_20261022')]:
 c=json.loads((d/'full-entry'/name/'f32-command.json').read_text());args=c['argv'];args[args.index('--output')+1]='/tmp/pi05-'+short+'-memory-result.json'
 cmdlines+=['## '+short,'',fence+'bash']
 cmdlines+=['export '+k+'='+shlex.quote(v) for k,v in c['env'].items()]
 cmdlines+=[shlex.join(args),fence,'']
cmdlines+=['## 复现','',fence+'bash','cd /home/skysi/code/vla.cpp','python3 scripts/patch_ggml_musa.py build-musa-deps/llama','cmake --build build-musa-official -j 8','ctest --test-dir build-musa-official --output-on-failure',fence,'',
'run_io_matrix.py / run_layout_matrix.py / run_late_matrix.py / run_fold_abba.py / run_validation.py / run_comparison.py 保留实验顺序，拒绝覆盖已有 stage；复测需使用新记录目录/stage。GPU 实验串行，期间不修改实现或构建。','',
'在本记录目录构建独立形状探针：','',fence+'bash',
'/usr/local/musa/bin/mcc -O3 -std=c++17 -x musa -mtgpu --cuda-gpu-arch=mp_22 -ffp-contract=off -I/usr/local/musa/include probe_gemm_shapes.cu -L/usr/local/musa/lib -lmusart -lmublas -Wl,-rpath,/usr/local/musa/lib -o /tmp/pi05-probe-gemm-shapes',fence]
(d/'使用命令.md').write_text('\n'.join(cmdlines)+'\n')
p=d/'待做清单.md';s=p.read_text().replace('[ ]','[x]');s+='\n全部条目已完成实现或实测后的不采用判定；未采用项不写成收益，见对应记录。\n';p.write_text(s)
p=d.parent/'README.md';s=p.read_text().replace('| [2026-11-02 GEMM 输出融合]', '| [2026-11-02 内存搬运与复用](2026-11-02_内存搬运与复用逐项实施/README.md) | 五项搬运/缓存、GEMM 形状研究、30 样本复测与验收 |\n| [2026-11-02 GEMM 输出融合]',1)
s=s.replace('\n- [内存搬运与复用逐项实施](2026-11-02_内存搬运与复用逐项实施/待做清单.md)：稳态测量和五项独立实验。\n','\n');p.write_text(s.rstrip()+'\n')
p=d.parent/'方案与验证说明.md';s=p.read_text().replace('最新实现与用时见 [2026-11-02 搬运与缓存迭代](2026-11-02_搬运与缓存迭代/README.md)。','最新实现与用时见 [2026-11-02 内存搬运与复用](2026-11-02_内存搬运与复用逐项实施/README.md)。');p.write_text(s)
print('Final documentation and index written')
