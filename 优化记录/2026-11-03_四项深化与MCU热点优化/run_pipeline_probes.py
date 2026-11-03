DB=' #pragma unroll\n for(int i=0;i<MT;i++)load_matrix_sync(a[i],w+(size_t)(row+i*32)*k,k);\n #pragma unroll\n for(int j=0;j<NT;j++)load_matrix_sync(b[j],x+(size_t)(col+j*32)*k,k);\n for(int p=16;p<k;p+=16) {\n  fragment<matrix_a,32,32,16,__half,row_major> next_a[MT];\n  fragment<matrix_b,32,32,16,__half,col_major> next_b[NT];\n  #pragma unroll\n  for(int i=0;i<MT;i++)load_matrix_sync(next_a[i],w+(size_t)(row+i*32)*k+p,k);\n  #pragma unroll\n  for(int j=0;j<NT;j++)load_matrix_sync(next_b[j],x+(size_t)(col+j*32)*k+p,k);\n  #pragma unroll\n  for(int i=0;i<MT;i++){\n   #pragma unroll\n   for(int j=0;j<NT;j++)mma_sync(acc[i][j],a[i],b[j],acc[i][j]);\n  }\n  #pragma unroll\n  for(int i=0;i<MT;i++)a[i]=next_a[i];\n  #pragma unroll\n  for(int j=0;j<NT;j++)b[j]=next_b[j];\n }\n #pragma unroll\n for(int i=0;i<MT;i++){\n  #pragma unroll\n  for(int j=0;j<NT;j++)mma_sync(acc[i][j],a[i],b[j],acc[i][j]);\n }\n'
from pathlib import Path
import subprocess,json
d=Path(__file__).resolve().parent
source=(d.parent/'2026-11-03_八项优化与延迟目标/probe_small_gemm.cu').read_text()
start=source.index(' for(int p=0;p<k;p+=16)')
end=source.index(' __shared__ float out[',start)
body=source[start:end]
rows=[]
for mode in [0,1,2,4,8]:
 loop=DB if mode==1 else (' #pragma unroll '+str(mode)+'\n'+body if mode else body)
 src=source[:start]+loop+source[end:]
 f=d/('probe-pipeline-'+str(mode)+'.cu');f.write_text(src)
 cmd=['/usr/local/musa/bin/mcc','-O3','-x','musa','-mtgpu','-ffp-contract=off','--cuda-gpu-arch=mp_22','-I/usr/local/musa/include','-L/usr/local/musa/lib','-Wl,-rpath,/usr/local/musa/lib',str(f),'-lmublas','-lmusart','-o','/tmp/pi05-pipeline-'+str(mode)]
 with (d/('build-pipeline-'+str(mode)+'.log')).open('w')as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 rows.append({'mode':mode,'compile':p.returncode})
 if p.returncode:continue
 for name,m,k in [('qkv',2560,1024),('o',1024,2048),('gu',8192,1024),('down',1024,4096)]:
  cmd=['/tmp/pi05-pipeline-'+str(mode),str(m),'50',str(k)]
  with (d/('pipeline-'+str(mode)+'-'+name+'.jsonl')).open('w')as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=120)
  rows.append({'mode':mode,'shape':name,'returncode':p.returncode})
  print(rows[-1],flush=True)
  (d/'pipeline-probe-status.json').write_text(json.dumps(rows,indent=2)+'\n')
(d/'pipeline-probe-status.json').write_text(json.dumps(rows,indent=2)+'\n')
