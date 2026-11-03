from pathlib import Path
import json,hashlib,subprocess,datetime
import numpy as np
repo=Path.cwd();d=repo/'优化记录/2026-11-03_八项优化与延迟目标'
j=lambda p:json.loads(p.read_text())
assert j(d/'candidate.json')==j(repo/'scripts/pi05_musa_profiles.json')
assert '100% tests passed, 0 tests failed out of 28' in (d/'ctest-small-gemm-all-v2.log').read_text()
for f in ['candidate-validation-summary.json','accept-status.json','tail-summary.json']:
 assert all(x['returncode']==0 for x in j(d/f))
for s in ['pruned','full']:
 a=j(d/'real-input30'/s/'report.json')
 assert a['samples']==30 and a['max_ms']<(500 if s=='pruned'else 800)
 assert a['repeat_max_abs']==0 and a['reference_max_abs']==0
 assert len(j(d/'random-candidate'/s/'summary.json'))==10
 assert all(x['max_abs']==0 for x in j(d/'random-candidate'/s/'summary.json'))
 assert len(j(d/'formats-candidate'/s/'summary.json'))==16
 assert all(x['max_abs']==0 for x in j(d/'formats-candidate'/s/'summary.json'))
 assert len(j(d/'embedding-candidate'/s/'summary.json'))==3
assert all(x['max_abs']==0 for x in j(d/'optimized-launcher-summary.json'))
rows=j(d/'正式验收耗时汇总.json')
text='# 八项优化最终验收\n\n2026-11-03，服务器skysi，仓库/home/skysi/code/vla.cpp，基于2c271b0。两个延迟目标均已通过，源码、运行入口与中文记录一并提交Git。八项均有实施或实测结论，只启用正确且有收益的组合。\n\n## 固定基准交替验收\n\nM1000/MP22、MUSA 5.1.x、performance电源模式、4线程、双相机、chunk50、剪枝3步/完整10步。F16权重及Tensor乘法，F32累加/输出/残差，不是FP4/FP8量化。\n\n固定6-token输入；每段预热3次，A→B→C→C→B→A，每段15次，每配置共30次。wall包含Python绑定到动作读回，不含模型加载、磁盘解码/resize和tokenizer。\n\n|配置|剪枝平均/P95 ms|完整平均/P95 ms|\n|---|---:|---:|\n'
for v,label in [('baseline','同轮基线'),('before-small','未用新小GEMM'),('candidate','最终选择')]:
 a=[next(x for x in rows if x['variant']==v and x['model']==s)for s in ['pruned','full']]
 text+=f"|{label}|{a[0]['mean_ms']:.3f}/{a[0]['p95_ms']:.3f}|{a[1]['mean_ms']:.3f}/{a[1]['p95_ms']:.3f}|\n"
text+='\n最终相对同轮基线下降8.51%/12.47%。原始证据：accept-*/、accept-status.json、正式验收耗时汇总.json。\n\n## 真实输入30次\n\n真实480×640图片，保持LeRobot float32 resize/pad，使用各自保存的tokenizer、state统计和相机顺序，预热3次后各计时30次。以下也是常驻predict wall，全部样本均低于目标，输出逐元素一致。\n\n|模型|提示token|平均 ms|P95 ms|最大 ms|\n|---|---:|---:|---:|---:|\n'
for s,label in [('pruned','剪枝'),('full','完整')]:
 a=j(d/'real-input30'/s/'report.json')
 text+=f"|{label}|{a['text_tokens']}|{a['mean_ms']:.3f}|{a['p95_ms']:.3f}|{a['max_ms']:.3f}|\n"
text+='\n来源：real-input30/。自动推理入口干净环境验证亦通过，见optimized-launcher-summary.json。\n'
text+="""
## 八项逐条结论

|项|实施/实测结论|最终选择|
|---|---|---|
|1 QKV拆分、RoPE/cast/KV写入|拆分本身为零拷贝VIEW，保留已有cast+KV尾写融合。K RoPE直接F16输出/尾写已实施并重测；late-k-rope-v2为465.737/768.535ms，但新增relative_peak为0.000135/0.000325，收益不足0.5%且没有稳定收益验收。|保持逐元素一致路径，不再启用额外RoPE融合|
|2 RoPE表|按每次当前positions生成prefix/suffix sin/cos表，各层共享；F32输出与原生路径逐元素一致，变位置/bucket检查通过。|ROPE_TABLE=1|
|3 U8 LUT+patch|原始HWC字节上传，精确F16 LUT及patch布局在一次GPU打包完成，直接进入embedding GEMM；覆盖全部256值和1/2视图。F32 resize输入保留原精度。|U8_PATCH=1，F32自动回退|
|4 固定RMS gamma折叠|加载时折叠到F16合并QKV/GateUp已实施，实测501.599/858.268ms；现有RMS+gamma已融合，额外舍入没有稳定收益。专家动态AdaRMS未当常量折叠。|关闭|
|5 最后前缀层KV-only|跳过无人使用的Q/O/FFN，只保留动作网络需要的K/V；独立和整模型对照通过。|LAST_KV_ONLY=1|
|6 视觉布局/projector出口|U8 embedding免旧布局复制；剪枝双相机批处理、分别评分/top-k并保留空间顺序；完整projector输出取消强制reshape+cont，直接写目标。既有bias/GELU/residual融合保留，未采用已知错误的Lt batch bias。|PRUNED_BATCH、VISION_NO_OUT_COPY及既有融合|
|7 GEMM/Attention|共享K/V的QK和AV由8批合并。实现MP22原生32×32×16 MMA、128线程协作；小GEMM选择M32/N64，O投影M64/N64，共享内存ld68。私有输入补零到64，但仍输出50个真实token。布局、数学模式、拆分、MMA原型和硬件剖析均有记录。|SMALL_GEMM=auto、FLAT_QK/AV=denoise|
|8 真实整网预取|旁路流在当前Down时预取下一层QKV，事件join保证使用前完成；实测64/256/4096KiB，不只测合成GEMM。最终64KiB为467.280/774.903ms，4096KiB为471.131/782.732ms，没有两模型稳定正收益。|关闭|

第1项没有另外增加处理混合dtype/多目标输出的大QKV节点：拆分为VIEW，已有cast/尾写融合，新增RoPE融合实测没有满足收益/数值取舍。第7项的GEMM+GeGLU合并原型也测试过，最终选择实测更好的独立小GEMM及现有GeGLU。未将微基准收益冒充整网收益，也未声称证明了所有可能实现的全局最优。

## 三部分诊断

下表是同步分段诊断，与普通wall的同步开销/样本数不同，不能把分段和代替上面的正式验收。

|模型/配置|视觉 ms|前缀 ms|动作 ms|分段wall ms|
|---|---:|---:|---:|---:|
"""
for stage,label in [('baseline','初始5样本'),('selected-split','最终8样本')]:
 for a in j(d/stage/'summary.json'):
  p=a['phases']
  text+=f"|{a['model']}/{label}|{p['vision_ms']:.3f}|{p['prefix_ms']:.3f}|{p['denoise_ms']:.3f}|{a['wall_ms']:.3f}|\n"
text+='\n异步完整入口JSON里ms_vision约0.5ms是主机提交，不是GPU视觉时长。\n'
text+="""
## 正确性

28/28 CTest通过且无GPU跳过；两模型各10组随机、32次U8/F32/视图切换、encode_images与precomputed/直接predict三轮重绑均通过，本轮新增选定路径对本轮基线逐元素一致。普通/bucket、生命周期、完整入口及旧参考对照10项均通过。证据：ctest-small-gemm-all-v2.log、random-candidate/、formats-candidate/、embedding-candidate/、candidate-validation-summary.json。

基线relative_peak 0.1%和既有CPU F32参考0.5%的原门限不变。完整入口结果：

|模型|逻辑激活|相对既有CPU F32参考 relative_peak|重复差异|
|---|---|---:|---:|
"""
for a in j(d/'candidate-summary-full-entry.json'):
 text+=f"|{'剪枝' if a['model'].startswith('pi05_') else '完整'}|{a['act_dtype']}|{a['relative_peak']:.9f}|{a['repeat_max_abs']}|\n"
text+="""
CPU单算子新增参考显式采用F16输入、F32累加，避免ARM原生F16 dot累加精度与内核契约混淆；原生GPU四种真实GEMM形状也独立对照。严格1e-5单算子门限和旧CPU整模型参考均未修改。更早full ActionLinear变化沿用上轮已验收范围。

同时修复CUSTOM userdata地址复用、语义常量未进Graph签名的问题，dt/epsilon/base复制进op_params；同图dt3→10→3回归通过。无效开关、错误fixture与撤回方案见实验汇总.md，未将未覆盖的早期测试冒称有效覆盖。

## 搬运与硬件证据

U8基准H2D总量1,259,776B→356,608B；剪枝4→3次，完整保持3次；D2H均一次6400B。显式Graph D2D copy节点：剪枝2个/2,359,296B、完整1个/4,194,304B，均降至0。每次各重放三段Graph。来源io-baseline/summary.json、io-candidate/summary.json。AV结果还原等GPU kernel仍有访存，copy节点为0不等于无GPU搬运。

MCU：前缀大GateUp原生库约90.53%计算吞吐，继续用库；新动作Down/GateUp约89.28%/90.29%，49寄存器/线程、8724B静态共享内存，独立kernel约245.34us/416.21us。不是整网时间。L2 API报0不代表物理没有L2。日志mcu-ffn.log、mcu-small-*.log；大型原始报告在/home/skysi/VLA/eight-target-20261103，校验见验收环境与产物.json。

## 运行和复现

现有两类GGUF直接复用。convert_pi05_local.py支持直接checkpoint/pretrained_model子目录、15/27层视觉及保存的SPM/JSON tokenizer。新--musa-optimized按GGUF vision_token_prune自动选两类性能环境，按元数据处理相机、状态与tokenizer。

~~~bash
cd /home/skysi/code/vla.cpp
VLA_LIBRARY=$PWD/build-musa-official/libvla.so VLA_N_THREADS=4 \
.venv-converter/bin/python scripts/run_pi05.py --musa-optimized \
 --model /home/skysi/VLA/models/pi05_cube_merge_cksize50_step3_0929/gguf/model.gguf \
 --image top=/path/top.png --image wrist=/path/wrist.png \
 --text 'pick up the cube' --state '0,0,0,0,0,0,0' --warmup 3 \
 --output /tmp/pi05-actions.json
~~~

换完整模型GGUF路径即可。--musa-optimized选择验收过的F16权重/F32逻辑激活/FA；普通入口仍可做精度对照。环境见scripts/pi05_musa_profiles.json和本目录selection.json；早期RoPE+KV环境保存在selection-rope-kv.json。

~~~bash
BENCH_CONFIG=selection.json BENCH_RUNS=30 .venv-converter/bin/python \
 优化记录/2026-11-03_八项优化与延迟目标/run_bench.py 新的实验目录名
.venv-converter/bin/python 优化记录/2026-11-03_八项优化与延迟目标/bench_real_input.py \
 pruned /tmp/pi05-real-pruned-new
.venv-converter/bin/python 优化记录/2026-11-03_八项优化与延迟目标/bench_real_input.py \
 full /tmp/pi05-real-full-new
~~~

使用新输出目录以保存旧证据。Conda若使系统powerprofilesctl缺gi，应使用系统Python/系统PATH；该启动失败保留在late-k-rope.log，late-k-rope-v2已正常重测。

逐阶段实测表.md列出每步候选，实验汇总.md保留失败/无效实验，原始命令/环境/source.patch/动作均在本目录。仅保存服务器，未同步Mac。验收限本机M1000和记录中的输入，未做其他GPU或真实机器人成功率实测。
"""
(d/'README.md').write_text(text)
if not (d/'selection-rope-kv.json').exists():
 (d/'selection-rope-kv.json').write_text((d/'selection.json').read_text())
(d/'selection.json').write_text((d/'candidate.json').read_text())
table='# 逐阶段实测\n\n短测按实际目录列出，不替代交替验收；源代码、命令、环境、样本数见各目录。fast-geglu、no-output-copy为无效/受额外转换污染的实验，不能声称其收益。\n\n|阶段|剪枝ms|完整ms|动作relative_peak（剪枝/完整）|\n|---|---:|---:|---:|\n'
for f in sorted(d.glob('*/summary.json'),key=lambda p:p.stat().st_mtime):
 a=j(f)
 if not isinstance(a,list) or len(a)!=2 or not all(isinstance(x,dict)and 'wall_ms'in x and 'baseline_relative_peak'in x for x in a):continue
 table+=f"|{f.parent.name}|{a[0]['wall_ms']:.3f}|{a[1]['wall_ms']:.3f}|{a[0]['baseline_relative_peak']:.6g}/{a[1]['baseline_relative_peak']:.6g}|\n"
(d/'逐阶段实测表.md').write_text(table+'\n失败与撤回原因见实验汇总.md，正式统计见README.md。\n')
(d/'当前进度.md').write_text('# 当前进度\n\n八项逐项实施/实测评估完成；目标通过：固定输入交替30次467.927/770.523ms；真实43/42-token输入30次481.444/779.535ms，最大482.963/791.099ms。28项CTest、20组随机及完整入口通过；本轮选定新增路径输出逐元素一致。结论、阶段用时、未采用原因与复现见README.md。\n\n自动入口：scripts/run_pi05.py --musa-optimized；环境：selection.json。源码及中文目录一并提交Git，提交号见git log。\n')
p=d/'目标与待办.md';old=p.read_text()
if not old.startswith('# 最终完成状态'):
 p.write_text('# 最终完成状态\n\n- [x] 八项实施/实测评估，采用与淘汰逐条见README\n- [x] 剪枝<500ms、完整<800ms，交替30次及真实输入30次均通过\n- [x] 原精度门限不放宽，最终新增优化对本轮基线逐元素一致\n- [x] 28项测试、随机输入、Graph重建/生命周期、完整入口与拷贝剖析\n- [x] 自动推理支持两类GGUF，中文目录保存全部阶段与失败\n\n以下为开工时计划原文，当前状态以上述最终验收为准。\n\n'+old)
p=d/'实验汇总.md';old=p.read_text()
if not old.startswith('# 历史阶段'):
 p.write_text('# 历史阶段明细\n\n以下保留当时的过程状态，包含尚未达标/尚待实施等早期描述；最终结论、选择、30样本及真实输入验收见README.md。未用最终代码冒充历史结果。\n\n'+old+'\n\n## 后续补充\n\n32×32片段需128线程协作。修正后数值正确但GeGLU整体融合不适合前缀；最终小GEMM见small-gemm-*.jsonl。\n\nctest-small-gemm-all.log里的新增CPU对照失败来自ARM原生F16 dot累加精度与F32累加契约不同。显式F16输入转F32的独立CPU matmul参考在不改1e-5门限下通过；原生GPU真实形状与原CPU整模型参考也通过，见ctest-small-gemm-all-v2.log。\n\nlate-k-rope第一次因Conda PATH令系统powerprofilesctl缺gi而未启动GPU，late-k-rope-v2使用系统PATH后成功；不是模型数值失败。\n')
p=repo/'优化记录/README.md';lines=p.read_text().splitlines()
for i,line in enumerate(lines):
 if '[八项优化与延迟目标]'in line:
  lines[i]='| [八项优化与延迟目标](2026-11-03_八项优化与延迟目标/README.md) | 已验收：真实输入30次481.444/779.535ms；交替30次467.927/770.523ms，八项与28项测试 |'
p.write_text('\n'.join(lines)+'\n')
p=repo/'优化记录/方案与验证说明.md';s=p.read_text()
if '八项优化与延迟目标最终验收（最新）'not in s:
 s+='\n\n## 2026-11-03：八项优化与延迟目标最终验收（最新）\n\n[最终记录](2026-11-03_八项优化与延迟目标/README.md)：真实输入30次平均481.444/779.535ms、最大482.963/791.099ms；固定输入交替30次467.927/770.523ms。28项CTest及20组随机通过，选定新增优化对本轮基线输出逐元素一致。\n\nscripts/run_pi05.py新增--musa-optimized，按GGUF自动选择剪枝/完整模型配置，默认build-musa-official，使用F16权重/F32逻辑激活/FA；两类GGUF可复用。没有稳定收益或新增舍入的候选保持关闭，过程全保留。\n'
p.write_text(s)
p=d/'bench_real_input.py';s=p.read_text()
a="out=d/'real-input30'/short;out.mkdir(parents=True,exist_ok=False)"
if a in s:s=s.replace(a,"out=Path(sys.argv[2]) if len(sys.argv)>2 else d/'real-input30'/short\nout.mkdir(parents=True,exist_ok=False)")
p.write_text(s)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
paths=[repo/'build-musa-official/libvla.so',repo/'build-musa-official/libvla_core.so',repo/'build-musa-official/bin/libggml-musa.so.0',repo/'scripts/pi05_musa_profiles.json',repo/'scripts/run_pi05.py']
raw=Path('/home/skysi/VLA/eight-target-20261103')
paths+=list(raw.glob('*.mcu-rep'))+list((raw/'candidate').glob('*.msys-rep'))
art=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}for p in paths if p.is_file()]
manifest={'created_at':datetime.datetime.now().astimezone().isoformat(),'code_base':'2c271b0','git_head_before_final_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'artifacts':art,'accepted_report_metadata':j(d/'accept-candidate-r1/pruned/report.json')['metadata']}
(d/'验收环境与产物.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('FINAL RECORDS WRITTEN',len(art),'artifacts')
print('RECORD BYTES',sum(p.stat().st_size for p in d.rglob('*')if p.is_file()))
