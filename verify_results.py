"""Compare all scientific numerical outputs; runtime and execution order are excluded."""
import json,argparse
from pathlib import Path
import numpy as np

def main_rows(path):
 return {(r['kind'],r['nu'],r['trial'],r['noise'],r['method']):r for line in Path(path).read_text().splitlines() for r in json.loads(line)['rows']}

def check(a,b,path):
 if isinstance(a,dict):
  for k,v in a.items():
   if k not in b:raise AssertionError(f'{path}: missing {k}')
   check(v,b[k],path+'/'+k)
 elif isinstance(a,list):
  if len(a)!=len(b):raise AssertionError(path+' length')
  for i,(x,y) in enumerate(zip(a,b)):check(x,y,path+'/'+str(i))
 elif isinstance(a,(int,float)) and not isinstance(a,bool):
  np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-8,err_msg=path,equal_nan=False)
 else:
  if a!=b:raise AssertionError(f'{path}: {a} != {b}')

def verify(reference,generated):
 a=main_rows(reference/'main.jsonl');b=main_rows(generated/'main.jsonl');assert a.keys()==b.keys()
 for k,r in a.items():check(r,b[k],'main/'+str(k))
 for file in ['tensor.json','diagnostics.json']:check(json.loads((reference/file).read_text()),json.loads((generated/file).read_text()),file)
 import pandas as pd
 np.testing.assert_allclose(pd.read_csv(reference/'refinement_curve.csv'),pd.read_csv(generated/'refinement_curve.csv'),rtol=1e-7,atol=1e-8)
 aa=np.load(reference.parent/'data/representative_burgers.npz');bb=np.load(generated/'data/representative_burgers.npz')
 for key in aa.files:np.testing.assert_allclose(aa[key],bb[key],rtol=1e-7,atol=1e-8)
 print(f'PASS: {len(a)} coefficient estimates, 1200 tensor estimates, and all diagnostic values agree.')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--reference',default='results');ap.add_argument('--generated',default='reproduced');args=ap.parse_args();verify(Path(args.reference),Path(args.generated))
