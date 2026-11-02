from pathlib import Path
import json,statistics
d=Path(__file__).resolve().parent
data={stage:json.loads((d/('io-'+stage)/'summary.json').read_text()) for stage in ['baseline','before-main-pin','final','mapped-final']}
lines=['# 搬运复测与隐式同步','',
'以下为 msys 预热后 3 次推理的诊断数据，不能代替无 profiler 的端到端延迟。外部复制只统计没有 Graph Node ID 的事件，与对应 MemcpyAsync API 口径一致。baseline 使用 graph trace，后续使用 node trace；Graph 内事件另列，不与 baseline 缺失的节点明细直接做减法。','',
'## 外部复制与调用','',
'|模型/配置|H2D 次数/字节|外部 D2D 次数/字节|D2H 次数/字节|同步次数|GraphLaunch|主机 Memcpy API 平均 ms|',
'|---|---:|---:|---:|---:|---:|---:|']
for stage in ['baseline','before-main-pin','final','mapped-final']:
 for row in data[stage]:
  samples=row['samples'];x=samples[0];cells=[]
  for kind in ['Memcpy HtoD','Memcpy DtoD','Memcpy DtoH']:
   t=x['copies'].get(kind,{'count':0,'bytes_from_export':0})
   cells.append(f"{t['count']} / {t['bytes_from_export']:,}")
  lines.append('|'+row['model']+'/'+stage+'|'+'|'.join(cells)+f"|{x['api_counts'].get('musaStreamSynchronize',0)}|{x['api_counts']['musaGraphLaunch']}|{statistics.mean(y['host_memcpy_api_ms'] for y in samples):.5f}|")
lines+=['',
'最终方案消除视觉→前缀的外部 D2D；复用位置常量减少 H2D。mapped-final 再消除两张 224×224×3 F32 图像的 1,204,224 字节 H2D，但没有稳定端到端优势，推荐仍为 pinned。',
'',
'before-main-pin 已将显式同步降为 1 次，但语言和输出使用 pageable 指针，MemcpyAsync 在主机仍等待 GPU。首个样本的语言上传等待约 112.66/139.34 ms，输出回读等待约 419.21/797.39 ms。最终使用持久 pinned slab 后，MemcpyAsync 调用不再承担这些等待；GPU 完成仍需最后一次同步。不能把这些包含计算的等待时长当作设备复制耗时或可节省的计算时间。',
'',
'主输入 slab 随最大容量复用，存放语言、噪声、结果；每次更新内容，不缓存旧文本/噪声。首次分配发生在视觉提交之前，最终回读入 pinned 后同步，再复制到 API 返回的普通 vector。缓存已满时不再分配；变长度/变相机生命周期已回归。',
'',
'## Graph 内仍存在的复制','',
'|模型|视觉 Graph D2D 字节|KV 尾部节点|每尾部节点字节|Graph 内 D2D 合计字节|','|---|---:|---:|---:|---:|']
for row in data['final']:
 x=row['samples'][0];t=x['graph_copy_nodes']['Memcpy DtoD'];kv=72 if row['model']=='pruned' else 324
 lines.append(f"|{row['model']}|{t['bytes_from_export']-kv*25600:,}|{kv}|25,600|{t['bytes_from_export']:,}|")
lines+=['',
'KV 尾部为 50×256×2 字节，后续 2/9 步×18 层×K/V 两路。视觉内部布局复制与这些 KV 更新仍保留，没有声称全部 D2D 都消失。',
'',
'当前 node export 对多个 Graph 节点复用相同 ts/dur；例如完整模型多个尾部节点同为 400.44 µs 且起始时间相同。直接相加会得到虚假的约 130 ms“复制总时间”。因此内部只报告节点数/字节，不给独立耗时结论。字节数由导出 Throughput×duration 还原并与张量尺寸核对；工具未直接导出 bytes 字段。',
'',
'所有最终采样窗口均为 3 次 GraphLaunch，无 GraphInstantiate、musaMalloc/Free、HostAlloc/Free。外部 API 和完整节点 trace 留在 io-final/io-mapped-final 与外部产物清单中，便于后续研究布局写入和 KV producer 融合。']
(d/'04_搬运复测与隐式同步.md').write_text('\n'.join(lines)+'\n')
print('copy audit saved')
