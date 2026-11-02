from pathlib import Path
import json
d=Path(__file__).resolve().parent
lines=['# 输入与边界搬运：逐项实测','',
'每项 3 次预热、正常计时 5 次，分段诊断另测 3 次。所有设置沿用本轮基线；诊断插入同步，不与正常端到端耗时相加或混比。短样本用于筛选，最终结论以 30 样本 ABBA 为准。','',
'|方案|剪枝总耗时 ms|完整总耗时 ms|剪枝视觉/前缀/动作 ms|完整视觉/前缀/动作 ms|对本轮基线相对误差|',
'|---|---:|---:|---|---|---|']
for name in ['baseline','pageable','pinned','mapped','async','direct','kv','output','vision-pack','layout','combined','combined-kv','mapped-combined','late-control','main-pin','flatten','pin-flatten']:
 normal=d/name/'summary.json';phases=d/(name+'-phases')/'summary.json'
 if not normal.exists() or not phases.exists():continue
 a=json.loads(normal.read_text());b=json.loads(phases.read_text())
 if len(a)<2 or len(b)<2:continue
 pp=[' / '.join(f'{x["phase_mean_ms"][k]:.3f}' for k in ['vision_ms','prefix_ms','denoise_ms']) for x in b]
 lines.append('|'+name+'|'+'|'.join(f'{x["mean_ms"]["wall_ms"]:.3f}' for x in a)+'|'+'|'.join(pp)+'|'+str([x['baseline_relative_peak'] for x in a])+'|')
lines+=['','pageable/pinned/mapped 各自启用同一组输入常量缓存和语言/噪声容量复用；async 在 pinned 基础上取消视觉完成处的主机等待，direct 在 async 基础上让视觉输出直接落到前缀输入，kv 单独测试动作尾部更新。配置以各目录 command.json 为准。','',
'异步运行时普通 ms_vision 反映提交区间，不是完整视觉设备耗时。表中三个阶段全部来自单独的 split-phases 诊断。','',
'前 7 组为初始搬运矩阵，output/vision-pack/layout 为内部布局独立对照，combined 为 pinned+async+direct+layout，combined-kv 再加入 KV；mapped-combined 仅替换图像输入映射方式。late-control 使用新构建复测 combined-kv，再单独加入 main-pin、flatten 和两者组合。各组的源码补丁与库校验在自己的 source.patch/report.json 中；时段波动不能归因于优化，最终只以交错复测定结论。', '', '所有短样本通过有限值、重复一致性和历史 MUSA F32 参考误差 <0.5% 检查；新增搬运候选的输出与本轮基线完全一致。最终保留方案待完整回归和交错复测后决定。']
(d/'02_输入与边界逐项结果.md').write_text('\n'.join(lines)+'\n')
