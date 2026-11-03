from pathlib import Path
import json,sys,collections,re
d=Path(__file__).resolve().parent;stage=sys.argv[1];result=[]
for model in ['pruned','full']:
 out=d/('io-'+stage)/model
 api=json.loads((out/'trace_musa_api_trace.json').read_text())
 marks=[x for x in api if x['Name']=='musaGetDeviceCount'][-6:]
 assert len(marks)==6
 events=json.loads(Path('/home/skysi/VLA/flashrt-rest-20261103',stage,model+'.json').read_text())['traceEvents']
 rows=[]
 for i in range(0,6,2):
  lo=marks[i]['Start(ns)']+marks[i]['Duration(ns)'];hi=marks[i+1]['Start(ns)']
  a=[x for x in api if lo<=x['Start(ns)']<hi]
  assert not any(x['Name'] in ('musaMalloc','musaFree','musaHostAlloc','musaFreeHost') for x in a)
  assert not any('GraphInstantiate' in x['Name'] for x in a)
  assert sum(x['Name']=='musaGraphLaunch' for x in a)==3
  copies=[x for x in events if x.get('ph')=='X' and x.get('name','').startswith('Memcpy ') and lo<=x['ts']*1000 and (x['ts']+x['dur'])*1000<=hi]
  by={};inside={}
  for x in copies:
   kind=x['name'];internal='Graph Node ID' in x.get('args',{})
   rec=(inside if internal else by).setdefault(kind,{'count':0,'bytes_from_export':0} if internal else {'count':0,'device_us':0,'bytes_from_export':0})
   rec['count']+=1
   if not internal:rec['device_us']+=x['dur']
   rate=x['args']['Throughput'];m=re.fullmatch(r'([0-9.e+-]+)(GiB|MiB|KiB|B)/s',rate);assert m,rate
   rec['bytes_from_export']+=round(float(m[1])*{'GiB':2**30,'MiB':2**20,'KiB':2**10,'B':1}[m[2]]*x['dur']/1e6)
  row={'iteration':i//2,'window_ms':(hi-lo)/1e6,'api_counts':dict(collections.Counter(x['Name'] for x in a)),'copies':by,'graph_copy_nodes':inside,'graph_copy_timing_note':'Node export reuses timestamps/durations across nodes; do not sum as independent GPU copy time.',
       'host_memcpy_api_ms':sum(x['Duration(ns)'] for x in a if 'Memcpy' in x['Name'])/1e6,
       'host_sync_ms':sum(x['Duration(ns)'] for x in a if 'Synchronize' in x['Name'])/1e6}
  rows.append(row)
 result.append({'model':model,'samples':rows})
(d/('io-'+stage)/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
