"""Generate independent trajectories and evaluate all prespecified estimators."""
import os
for v in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
from core import *
import time,json,argparse,sys,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

GRIDS=[32,48,64,96,128]
NOISES=[0.,.001,.01]

def classify(lo,hi,delta=.0005):
    if lo>delta:return 'physical'
    if lo>=-delta and hi<=delta:return 'negligible'
    return 'unresolved'

def task(job):
 kind,nu,trial,reps=job;t0=time.perf_counter();t=np.linspace(0,.8,161);clean=[];out=[];work=0
 for scheme in [0,1]:
  for n in GRIDS:
   for rep in range(reps):
    seed=10000+trial*reps+rep;h=2*np.pi/n
    u,steps=solve(initial(n,seed,kind),h,kind,nu,scheme,t);work+=4*n*steps
    clean.append(dict(u=u,h=h,scheme=scheme,rep=rep,n=n))
 refs=[]
 for rep in range(2):
  v=spectral_reference(256,1000000+trial*2+rep,kind,nu,t)
  X,y=weak(v,t);refs.append(dict(X=X,y=y))
 XR=np.vstack([r['X'] for r in refs]);yr=np.concatenate([r['y'] for r in refs])
 for noise in NOISES:
  rs=[]
  for r in clean:
   sigma=noise*np.std(r['u']);rng=np.random.default_rng(np.random.SeedSequence([kind,trial,r['rep'],r['scheme'],r['n'],381]))
   u=r['u']+rng.normal(0,sigma,r['u'].shape);X,y=weak(u,t,sigma)
   rs.append(dict(X=X,y=y,h=r['h'],scheme=r['scheme'],rep=r['rep']))
  coeffs={};intervals={}
  for method in ['finest','pooled','joint2','joint3','drop_coarse']:
   rr=[r for r in rs if r['h']<max(r['h'] for r in rs)] if method=='drop_coarse' else rs
   powers=(1,2,3) if method=='joint3' else (1,2)
   fitmethod='joint' if method in ['joint2','joint3','drop_coarse'] else method
   c,hw,d=fit_interval(rr,fitmethod,powers)
   lo=c[6]-hw[6];hi=c[6]+hw[6]
   coeffs[method]=c;intervals[method]=(lo,hi)
   out.append(dict(kind=kind,nu=nu,trial=trial,noise=noise,method=method,coeff=c.tolist(),diffusion=c[6],lo=lo,hi=hi,decision=classify(lo,hi),coefficient_error=float(np.linalg.norm(c-truth(kind,nu))/np.linalg.norm(truth(kind,nu))),transfer_error=float(np.linalg.norm(XR@c-yr)/np.linalg.norm(yr)),**d))
  for power in [2,3]:
   c=separate_extrapolation(rs,tuple(range(1,power+1)))
   out.append(dict(kind=kind,nu=nu,trial=trial,noise=noise,method='richardson'+str(power),coeff=c.tolist(),diffusion=c[6],coefficient_error=float(np.linalg.norm(c-truth(kind,nu))/np.linalg.norm(truth(kind,nu))),transfer_error=float(np.linalg.norm(XR@c-yr)/np.linalg.norm(yr))))
  lo=min(intervals[m][0] for m in ['joint2','joint3','drop_coarse']);hi=max(intervals[m][1] for m in ['joint2','joint3','drop_coarse'])
  c=coeffs['joint2']
  out.append(dict(kind=kind,nu=nu,trial=trial,noise=noise,method='envelope',coeff=c.tolist(),diffusion=c[6],lo=lo,hi=hi,decision=classify(lo,hi),coefficient_error=float(np.linalg.norm(c-truth(kind,nu))/np.linalg.norm(truth(kind,nu))),transfer_error=float(np.linalg.norm(XR@c-yr)/np.linalg.norm(yr))))
  # Best available high-order discretization is an explicit accuracy control.
  hr=[]
  for rep in range(reps):
   n=128;u,steps=solve(initial(n,10000+trial*reps+rep,kind),2*np.pi/n,kind,nu,3,t)
   sigma=noise*np.std(u);rng=np.random.default_rng(np.random.SeedSequence([kind,trial,rep,3,n,381]));u+=rng.normal(0,sigma,u.shape)
   X,y=weak(u,t,sigma);hr.append(dict(X=X,y=y,h=2*np.pi/n,scheme=0,rep=rep))
  c,hw,d=fit_interval(hr,'pooled')
  out.append(dict(kind=kind,nu=nu,trial=trial,noise=noise,method='fourth_order',coeff=c.tolist(),diffusion=c[6],lo=c[6]-hw[6],hi=c[6]+hw[6],decision=classify(c[6]-hw[6],c[6]+hw[6]),coefficient_error=float(np.linalg.norm(c-truth(kind,nu))/np.linalg.norm(truth(kind,nu))),transfer_error=float(np.linalg.norm(XR@c-yr)/np.linalg.norm(yr)),**d))
 return {'job':job,'seconds':time.perf_counter()-t0,'work':work,'rows':out}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--trials',type=int,default=40);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--reps',type=int,default=8);ap.add_argument('--output',default='results/main.jsonl');args=ap.parse_args()
 jobs=[(k,nu,s,args.reps) for k in range(3) for nu in [0.,.002,.02] for s in range(args.trials)]
 dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True);t0=time.perf_counter()
 with dest.open('w') as f,ProcessPoolExecutor(max_workers=args.workers) as ex:
  futures=[ex.submit(task,j) for j in jobs]
  for i,fu in enumerate(as_completed(futures)):
   result=fu.result();f.write(json.dumps(result)+'\n');f.flush()
   if (i+1)%10==0:print(i+1,'/',len(jobs),'elapsed',round(time.perf_counter()-t0,1),flush=True)
 meta={'args':vars(args),'seconds':time.perf_counter()-t0,'python':sys.version,'platform':platform.system(),'numpy':np.__version__,'jobs':len(jobs),'rows':len(jobs)*27,'schema':1}
 dest.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2))
 print(meta,flush=True)
