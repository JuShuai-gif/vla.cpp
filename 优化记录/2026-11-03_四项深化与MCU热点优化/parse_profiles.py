from pathlib import Path
import re,json,collections,gzip
d=Path('优化记录/2026-11-03_四项深化与MCU热点优化')
for short in ['pruned','full']:
 f=d/('mcu-'+short+'.log')
 source=f.open()if f.exists()else gzip.open(str(f)+'.gz','rt')
 headers=[];current=None
 for line in source:
  m=re.match(r'^  (.*?) \(([^()]*)\)x\(([^()]*)\), Context ',line)
  if m:
   current={'name':m[1],'grid':m[2],'block':m[3]};headers.append(current)
  if current:
   m=re.match(r'\s+Duration\s+(ns|us|ms|s)\s+([\d.,]+)',line)
   if m:current['ms']=float(m[2].replace(',',''))*{'ns':1e-6,'us':.001,'ms':1,'s':1000}[m[1]]
   for label,key in [('Registers Per Thread','registers'),('Static Shared Memory Per Block','shared_bytes'),('Compute(MP) Throughput','compute_pct')]:
    if label in line:
     nums=re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',line.split(label,1)[1])
     if nums:current[key]=float(nums[-1].replace(',',''))
 markers=[(i,x['name'])for i,x in enumerate(headers)if 'vla_r2_'in x['name']]
 print('headers',len(headers),'markers',markers[:8])
 assert sum('vla_r2_begin'in x['name']for x in headers)==1
 begin=next(i for i,x in enumerate(headers)if 'vla_r2_begin'in x['name'])
 end=next(i for i,x in enumerate(headers)if 'vla_r2_end'in x['name'])
 selected=headers[begin+1:end];groups=collections.defaultdict(list)
 for x in selected:
  if 'ms'in x:groups[x['name']].append(x)
 rows=[]
 for name,a in groups.items():
  rows.append({'name':name,'count':len(a),'sum_ms':sum(x['ms']for x in a),'mean_ms':sum(x['ms']for x in a)/len(a),'max_ms':max(x['ms']for x in a),'registers':sorted(set(x.get('registers',0)for x in a)),'shared_bytes':sorted(set(x.get('shared_bytes',0)for x in a))})
 rows.sort(key=lambda x:-x['sum_ms'])
 result={'model':short,'scope':'MCU Graph OFF marked warmed real-input inference','launches':len(selected),'with_duration':sum('ms'in x for x in selected),'sum_kernel_ms':sum(x['sum_ms']for x in rows),'kernels':rows}
 (d/('mcu-ranking-'+short+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 print('TOTAL',result['sum_kernel_ms'],'TOP',rows[:10])
 raw=Path('/home/skysi/VLA/four-deep-20261103')/('baseline-'+short+'.json')
 archived=d/'原始报告'/('baseline-'+short+'.json.gz')
 a=json.loads(gzip.open(archived,'rt').read()if archived.exists()else raw.read_text());print('MSYS type/keys',type(a).__name__,list(a)[:8])
 if isinstance(a,dict)and 'traceEvents'in a:
  ev=a['traceEvents'];print('CATEGORIES',collections.Counter(x.get('cat')for x in ev).most_common(10))
  marks=[x for x in ev if 'vla_r2_'in str(x.get('name',''))];print('MARK SAMPLE',marks[:2])
