from pathlib import Path
import re,json,collections
folder=Path(__file__).resolve().parent
for precision in ("baseline","int8-expert-smooth"):
 path=folder/f"v3-mcu-pruned-{precision}.log"
 if not path.exists():continue
 headers=[];current=None
 for line in path.read_text().splitlines():
  m=re.match(r'^  (.*?) \(([^()]*)\)x\(([^()]*)\), Context ',line)
  if m:
   current=dict(name=m[1],grid=m[2],block=m[3]);headers.append(current)
  if current:
   m=re.match(r'\s+Duration\s+(ns|us|ms|s)\s+([\d.,]+)',line)
   if m:current["ms"]=float(m[2].replace(",",""))*{"ns":1e-6,"us":.001,"ms":1,"s":1000}[m[1]]
   for label,key in [("Registers Per Thread","registers"),("Static Shared Memory Per Block","shared_bytes"),("Compute(MP) Throughput","compute_pct")]:
    if label in line:
     nums=re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',line.split(label,1)[1])
     if nums:current[key]=float(nums[-1].replace(",",""))
 begins=[i for i,x in enumerate(headers)if "vla_r2_begin" in x["name"]]
 ends=[i for i,x in enumerate(headers)if "vla_r2_end" in x["name"]]
 if len(begins)!=1 or len(ends)!=1:
  print(precision,"incomplete headers",len(headers),begins,ends);continue
 selected=headers[begins[0]+1:ends[0]];groups=collections.defaultdict(list)
 for x in selected:
  if "ms" in x:groups[x["name"]].append(x)
 rows=[]
 for name,items in groups.items():
  rows.append(dict(name=name,count=len(items),sum_ms=sum(x["ms"] for x in items),mean_ms=sum(x["ms"]for x in items)/len(items),
                   registers=sorted({x.get("registers",0)for x in items}),shared_bytes=sorted({x.get("shared_bytes",0)for x in items})))
 rows.sort(key=lambda x:-x["sum_ms"])
 report=dict(precision=precision,scope="V3 Graph OFF; selected kernels of one warmed observation; profiler-perturbed; not acceptance timing",
             kernels=rows,raw=selected)
 (folder/f"v3-mcu-pruned-{precision}-summary.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(dict(precision=precision,kernels=rows)),flush=True)
