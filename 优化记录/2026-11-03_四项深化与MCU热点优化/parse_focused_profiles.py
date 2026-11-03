from pathlib import Path
import re,json,collections,gzip
d=Path(__file__).resolve().parent
def parse(f):
 headers=[];current=None
 source=gzip.open(str(f)+'.gz','rt')if not f.exists()else f.open()
 with source:
  for line in source:
   m=re.match(r'^  (.*?) \(([^()]*)\)x\(([^()]*)\), Context ',line)
   if m:current={'name':m[1],'grid':m[2],'block':m[3]};headers.append(current)
   if current:
    m=re.match(r'\s+Duration\s+(ns|us|ms|s)\s+([\d.,]+)',line)
    if m:current['ms']=float(m[2].replace(',',''))*{'ns':1e-6,'us':.001,'ms':1,'s':1000}[m[1]]
    for label,key in [('Registers Per Thread','registers'),('Static Shared Memory Per Block','shared_bytes'),('Compute(MP) Throughput','compute_pct')]:
     if label in line:
      nums=re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',line.split(label,1)[1])
      if nums:current[key]=float(nums[-1].replace(',',''))
 assert headers and all('ms'in x for x in headers),(f,headers[:2])
 return headers
def aggregate(items):
 groups=collections.defaultdict(list)
 for x in items:groups[(x['name'],x['grid'],x['block'])].append(x)
 rows=[]
 for (name,grid,block),a in groups.items():
  rows.append({'name':name,'grid':grid,'block':block,'count':len(a),
               'sum_ms':sum(x['ms']for x in a),'mean_ms':sum(x['ms']for x in a)/len(a),
               'registers':sorted(set(x.get('registers',0)for x in a)),
               'shared_bytes':sorted(set(x.get('shared_bytes',0)for x in a)),
               'mean_compute_pct':sum(x.get('compute_pct',0)for x in a)/len(a)})
 return sorted(rows,key=lambda x:-x['sum_ms'])
rows=[]
for mode in [0,1,4,8]:
 for shape in ['gu','o','down']:
  a=parse(d/f'focus-pipe-{mode}-{shape}-r3.log');assert len(a)==1
  rows.append(dict(mode=mode,shape=shape,**a[0]))
(d/'focused-pipeline-mcu.json').write_text(json.dumps(rows,indent=2)+'\n')
for variant in ['all','all-parallel']:
 a=parse(d/f'focus-attention-{variant}.log')
 (d/('focused-attention-'+variant+'.json')).write_text(json.dumps(aggregate(a),indent=2)+'\n')
for short in ['pruned','full']:
 a=parse(d/f'hot-combined-r2-{short}.log')
 markers=[(i,x['name'])for i,x in enumerate(a)if 'vla_r2_'in x['name']]
 assert len(markers)==2 and 'begin'in markers[0][1]and 'end'in markers[1][1],markers
 measured=a[markers[0][0]+1:markers[1][0]];rows=aggregate(measured)
 result={'model':short,'config':'combined-pipe4-env.json','scope':'MCU Graph OFF warmed marked real input, filtered kernels only; excludes other kernels',
         'launches':len(measured),'sum_filtered_ms':sum(x['ms']for x in measured),'kernels':rows}
 (d/('mcu-hot-'+short+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(short,result['launches'],result['sum_filtered_ms'],[(x['name'].split('(')[0],x['grid'],round(x['sum_ms'],3))for x in rows[:7]])
