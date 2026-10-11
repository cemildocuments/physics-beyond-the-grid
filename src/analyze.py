"""Summarize all trials without selecting seeds or discarding failed decisions."""
from pathlib import Path
import json,csv,argparse
import numpy as np
import pandas as pd

def load_main(path):
 return pd.DataFrame([r for line in Path(path).read_text().splitlines() for r in json.loads(line)['rows']])

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='results');args=ap.parse_args();root=Path(args.root)
 d=load_main(root/'main.jsonl');d['diffusion_absolute_error']=abs(d.diffusion-d.nu)
 stats=d.groupby(['noise','method'])[['diffusion_absolute_error','coefficient_error','transfer_error','residual']].quantile([.25,.5,.75]);stats.to_csv(root/'summary.csv')
 d.drop(columns=['coeff']).to_csv(root/'main_metrics.csv',index=False)
 decisions=d[d.decision.notna()].groupby(['nu','noise','method','decision']).size().unstack(fill_value=0);decisions.to_csv(root/'decisions.csv')
 te=pd.DataFrame(json.loads((root/'tensor.json').read_text()));te.groupby(['case','noise','method'])[['error','ratio']].quantile([.25,.5,.75]).to_csv(root/'tensor_summary.csv')
 result={'coefficient_estimates':len(d),'datasets':len(d[['kind','nu','trial','noise']].drop_duplicates()),'unique_trial_blocks':int(d.trial.nunique()),'tensor_estimates':len(te)}
 if (root/'replication.jsonl').exists():
  e=load_main(root/'replication.jsonl');keys=['kind','nu','trial','noise','method'];a=d.set_index(keys).sort_index();b=e.set_index(keys).sort_index();result['replication_max_coefficient_difference']=float(np.max(np.abs(np.vstack(a.coeff)-np.vstack(b.coeff))))
 (root/'analysis_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
