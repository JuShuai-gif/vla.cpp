from pathlib import Path
import subprocess,json,sys
d=Path(__file__).resolve().parent;out=d/'gemm-shapes';out.mkdir(exist_ok=False)
shapes=[]
for m,k in [(2560,1024),(1024,2048),(8192,1024),(1024,4096)]:shapes.append((m,50,k,1,['56','64','128']))
for n in [278,518,315,554]:
 pads=[str((n+15)//16*16),str((n+63)//64*64)]
 for m,k in [(2560,2048),(2048,2048),(32768,2048),(2048,16384)]:shapes.append((m,n,k,1,sorted(set(pads),key=int)))
for b in [1,2]:
 for m,k in [(3456,1152),(1152,1152),(4304,1152),(1152,4304),(2048,1152)]:
  variants=[]
  if m==4304:variants=[f'{mp}:256:{k}:{b}' for mp in [4352,4480]]
  if k==4304:variants=[f'{m}:256:{kp}:{b}' for kp in [4352,4480]]
  shapes.append((m,256,k,b,variants))
rows=[]
for m,n,k,b,variants in shapes:
 name=f'{m}x{n}x{k}x{b}'
 cmd=['/tmp/pi05-probe-gemm-shapes',str(m),str(n),str(k),str(b),*variants]
 print('GEMM',name,flush=True)
 with (out/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=120)
 records=[json.loads(x) for x in (out/(name+'.log')).read_text().splitlines() if x.startswith('{')]
 rows.append({'shape':[m,n,k,b],'command':cmd,'returncode':p.returncode,'measurements':records})
 (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 if p.returncode:raise SystemExit(p.returncode)
 assert records and all(x['relative_peak']<.001 for x in records),records
print('all GEMM probes passed',flush=True)
