from pathlib import Path
import json,collections,gzip
d=Path('优化记录/2026-11-03_四项深化与MCU热点优化')
for s in ['pruned','full']:
 archive=d/'原始报告'/('baseline-'+s+'.json.gz')
 a=json.loads(gzip.open(archive,'rt').read()if archive.exists()else (Path('/home/skysi/VLA/four-deep-20261103')/('baseline-'+s+'.json')).read_text())['traceEvents']
 beg=[x for x in a if 'vla_r2_begin'in x.get('name','')and x.get('ph')=='X']
 end=[x for x in a if 'vla_r2_end'in x.get('name','')and x.get('ph')=='X']
 assert len(beg)==len(end)==1
 lo=beg[0]['ts']+beg[0]['dur'];hi=end[0]['ts']
 ev=[x for x in a if x.get('ph')=='X'and 'Block'in x.get('args',{})and x['ts']>=lo and x['ts']<hi]
 groups=collections.defaultdict(list)
 for x in ev:groups[x['name']].append(x['dur']*.001)
 rows=[{'name':n,'count':len(x),'sum_ms':sum(x),'mean_ms':sum(x)/len(x),'max_ms':max(x)}for n,x in groups.items()]
 rows.sort(key=lambda x:-x['sum_ms'])
 m=json.loads((d/('mcu-ranking-'+s+'.json')).read_text())
 result={'model':s,'timing_valid':False,'timing_limitation':'Repeated timestamps and durations across unrelated kernels; counts only, not timing/ranking evidence','scope':'MSYS Graph OFF marked warmed real-input inference','window_ms':(hi-lo)*.001,'launches':len(ev),'sum_kernel_ms':sum(x['sum_ms']for x in rows),'mcu_launches':m['launches'],'kernels':rows}
 assert len(ev)==m['launches'],result
 (d/('msys-ranking-'+s+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(s,'launches',len(ev),'gpu ms',result['sum_kernel_ms'],'top',rows[:5])
text='# MCU与MSYS基线热点\n\n标记包围预热后的一次真实输入推理；Graph OFF以获取独立kernel活动。MCU时间受采集干预，不代替普通Graph ON wall；MSYS存在不同kernel重复时间戳/时长，仅核对调用数量；其累计时间和排序无效，不用于性能结论。\n\n'
for s in ['pruned','full']:
 a=json.loads((d/('mcu-ranking-'+s+'.json')).read_text())
 text+=f"\n## {s}：{a['launches']}次kernel，MCU累计{a['sum_kernel_ms']:.3f}ms\n\n|kernel|次数|累计ms|单次均值ms|寄存器|共享字节|\n|---|---:|---:|---:|---|---|\n"
 for x in a['kernels'][:20]:
  text+=f"|{x['name']}|{x['count']}|{x['sum_ms']:.3f}|{x['mean_ms']:.5f}|{x['registers']}|{x['shared_bytes']}|\n"
text+='\n优先级：大型prefix GEMM、剪枝prefix Attention、动作小GEMM，以及cpy_scalar/GeGLU/pack/softmax。后续逐项记录实际命中、精度与整网收益。\n'
(d/'MCU基线热点.md').write_text(text)
