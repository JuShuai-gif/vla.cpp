from pathlib import Path
import subprocess,os,json,time,collections
d=Path(__file__).resolve().parent;repo=d.parents[1]
while not(d/'selected-msys-status.json').exists()or len(json.loads((d/'selected-msys-status.json').read_text()))<2:time.sleep(1)
for short in ['pruned','full']:
 a=json.loads((d/('selected-msys-'+short+'-counts.json')).read_text())
 hot=json.loads((d/('mcu-hot-'+short+'.json')).read_text())
 families=['small_gemm_kernel','packed_norm_kernel','packed_geglu_kernel','vision_scores_kernel']
 msys={f:sum(c for name,c in a['kernel_counts'].items()if f in name)for f in families}
 mcu={f:sum(x['count']for x in hot['kernels']if f in x['name'])for f in families}
 assert msys==mcu,(short,msys,mcu)
 a['filtered_mcu_count_crosscheck']={'msys':msys,'mcu':mcu,'passed':True}
 (d/('selected-msys-'+short+'-counts.json')).write_text(json.dumps(a,indent=2)+'\n')
with(d/'build-ctest-properties.log').open('w')as f:
 r=subprocess.run(['cmake','--build','build-musa-official','-j','8'],cwd=repo,stdout=f,stderr=subprocess.STDOUT,timeout=900)
assert r.returncode==0
with(d/'ctest-properties.log').open('w')as f:
 r=subprocess.run(['ctest','--test-dir','build-musa-official','-R','musa_(packed_norm|prune_score)','--output-on-failure'],cwd=repo,
                  env=dict(os.environ,VLA_MUSA_SMALL_PIPE='shape'),stdout=f,stderr=subprocess.STDOUT,timeout=240)
assert r.returncode==0
(d/'finish-checks.json').write_text(json.dumps({'passed':True,'msys_mcu_family_counts_match':True,'ctest_skip_properties_verified_on_gpu':True},indent=2)+'\n')
print('FINISH CHECKS COMPLETE',flush=True)
