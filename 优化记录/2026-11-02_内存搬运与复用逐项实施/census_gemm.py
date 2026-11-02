from pathlib import Path
import json,math,hashlib
d=Path(__file__).resolve().parent
old=d.parent/'2026-11-02_GEMM输出融合'
rows=[]
for model in ['pruned','full']:
 p=old/'final-phases'/model/'shapes.json'
 for x in json.loads(p.read_text()):
  if x['op']!='MUL_MAT':continue
  out,a,b=x['tensors'][:3];m=out['ne'][0];n=b['ne'][1];k=a['ne'][0];batch=math.prod(out['ne'][2:])
  rows.append({'model':model,'phase':x['phase'],'m':m,'n':n,'k':k,'batch':batch,'count':x['count'],'a_type':a['type'],'b_type':b['type'],'out_type':out['type'],
   'a_ne':a['ne'],'b_ne':b['ne'],'out_ne':out['ne'],'a_nb':a['nb'],'b_nb':b['nb'],'out_nb':out['nb'],
   'flops_per_inference':2*m*n*k*batch*x['count'],'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for x in rows:
 if x['a_type']=='f16':
  m,n,k=x['m'],x['n'],x['k']
  x['baseline_blas_layout']={'m':m,'n':n,'k':k,'batch':x['batch'],'trans_a':'T','trans_b':'N','lda':k,'ldb':k,'ldc':m,'stride_a':0 if x['a_ne'][2]==1 else k*m,'stride_b':k*n,'stride_c':m*n,'b_f32_to_f16':x['b_type']=='f32'}
  if x['batch']>1 and x['a_ne'][2]==1:
   x['flattened_blas_layout']={'m':m,'n':n*x['batch'],'k':k,'batch':1,'trans_a':'T','trans_b':'N','lda':k,'ldb':k,'ldc':m}
 else:x['baseline_blas_layout']=None
(d/'GEMM形状清单.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
for x in rows:print(x['model'],x['phase'],x['m'],x['n'],x['k'],x['batch'],x['count'],x['a_type'])
