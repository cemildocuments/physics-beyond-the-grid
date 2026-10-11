"""Export the representative raw field and its per-grid fitted coefficients."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from core import initial,solve,weak,fit

if __name__=='__main__':
 output=Path(os.environ.get('PHYSICS_RESULTS_DIR','results'));output.mkdir(parents=True,exist_ok=True)
 t=np.linspace(0,.8,161);rows=[]
 for setting in [0,1]:
  for n in [32,48,64,96,128]:
   records=[]
   for rep in range(8):
    u,_=solve(initial(n,10000+rep,1),2*np.pi/n,1,.002,setting,t);X,y=weak(u,t)
    records.append(dict(X=X,y=y,h=2*np.pi/n,scheme=setting))
   c,_=fit(records,'pooled');rows.append(dict(setting=setting,N=n,h=2*np.pi/n,diffusion=c[6]))
 pd.DataFrame(rows).to_csv(output/'refinement_curve.csv',index=False)
 u,_=solve(initial(128,10000,1),2*np.pi/128,1,.002,0,t)
 (output/'data').mkdir(exist_ok=True)
 np.savez_compressed(output/'data/representative_burgers.npz',u=u,t=t,x=np.arange(128)*2*np.pi/128,nu=.002,a=1.,seed=10000)
