from pathlib import Path
import subprocess,os,json,sys,time,re
d=Path(__file__).resolve().parent;repo=d.parents[1]
status=d/'iteration2-status.json'
while not status.exists()or len(json.loads(status.read_text()))<8:time.sleep(1)
if os.environ.get('WAIT_FOR_HOT')=='1':
 while not(d/'hot-mcu-status.json').exists()or len(json.loads((d/'hot-mcu-status.json').read_text()))<4:time.sleep(1)
rows=[]
for mode in [0,1,4,8]:
 src=d/('probe-pipeline-mcu-'+str(mode)+'.cu');exe='/tmp/pi05-pipeline-mcu-'+str(mode)
 cmd=['/usr/local/musa/bin/mcc','-O3','-x','musa','-mtgpu','-ffp-contract=off','--cuda-gpu-arch=mp_22',str(src),'-o',exe,'-lmublas','-lmusart']
 with(d/('build-focus-'+str(mode)+'.log')).open('w')as f:r=subprocess.run(cmd,cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 assert r.returncode==0
 for shape,m,k,tile in [('gu',8192,1024,1),('o',1024,2048,2),('down',1024,4096,1)]:
  name='focus-pipe-'+str(mode)+'-'+shape+'-r3'
  cmd=['/usr/bin/mcu','--dump-mode','normal','--kernel-name','regex:small_gemm','-o','/home/skysi/VLA/four-deep-20261103/'+name,exe,str(m),'50',str(k),str(tile)]
  (d/(name+'-command.json')).write_text(json.dumps({'argv':cmd},indent=2)+'\n')
  env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/musa/lib:/usr/local/musa/lib64:'+os.environ.get('LD_LIBRARY_PATH',''))
  with(d/(name+'.log')).open('w')as f:r=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
  raw=(d/(name+'.log')).read_text()
  valid='No kernels were profiled'not in raw and '==ERROR=='not in raw and 'Duration'in raw and 'small_gemm'in raw
  row={'mode':mode,'shape':shape,'returncode':r.returncode,'valid_kernel_report':valid}
  rows.append(row);(d/'focused-mcu-status.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(row,flush=True)
  assert valid,row
