from pathlib import Path
import os,sys,json,subprocess
d=Path(__file__).resolve().parent;rows=[]
env=dict(os.environ,PATH='/usr/bin:/bin:'+os.environ.get('PATH',''),BENCH_CONFIG='candidate.json',BENCH_RUNS='8')
for stage,options in [
 ('late-k-rope-v2',{'VLA_PI05_K_ROPE_DIRECT':'1'}),
 ('late-prefetch64',{'VLA_PI05_PREFETCH_KIB':'64'}),
 ('late-prefetch4096',{'VLA_PI05_PREFETCH_KIB':'4096'}),
 ('selected-split',{})]:
 e=dict(env)
 if stage=='selected-split':e['BENCH_SPLIT']='1'
 cmd=[sys.executable,str(d/'run_bench.py'),stage]+[k+'='+v for k,v in options.items()]
 with (d/(stage+'.log')).open('w')as f:p=subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 rows.append({'stage':stage,'returncode':p.returncode});print(stage,p.returncode,flush=True)
 (d/'tail-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
cmd=['/usr/local/musa/bin/mcc','-O3','-x','musa','-mtgpu','-ffp-contract=off','--cuda-gpu-arch=mp_22','-I/usr/local/musa/include','-L/usr/local/musa/lib','-Wl,-rpath,/usr/local/musa/lib',str(d/'probe_small_once.cu'),'-lmublas','-lmusart','-o','/tmp/pi05-small-once']
with (d/'build-small-once.log').open('w')as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
for short,m,k in [('down',1024,4096),('gateup',8192,1024)]:
 cmd=['/usr/bin/mcu','--dump-mode','normal','-o','/home/skysi/VLA/eight-target-20261103/mcu-small-'+short,
      '/tmp/pi05-small-once',str(m),'50',str(k),'1']
 (d/('mcu-small-'+short+'-command.json')).write_text(json.dumps({'argv':cmd},indent=2)+'\n')
 with (d/('mcu-small-'+short+'.log')).open('w')as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 rows.append({'stage':'mcu-small-'+short,'returncode':p.returncode})
 (d/'tail-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print('TAIL COMPLETE',flush=True)
