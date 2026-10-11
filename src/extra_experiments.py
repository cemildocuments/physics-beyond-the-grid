import os
for v in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[v]='1'
from core import *
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Two-dimensional terms, with 2*u_xy parameterized by the off-diagonal diffusivity.
DERS=[(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(4,0),(3,1),(2,2),(1,3),(0,4)]
MODES=np.array([(a,b) for a in range(-4,5) for b in range(0,5) if (b>0 or a>0)])

def modal_data(n,scheme,seed,D,noise):
    t=np.linspace(0,1.2,161);h=2*np.pi/n;kx,ky=MODES.T
    rng=np.random.default_rng(seed);A=(rng.normal(size=len(kx))+1j*rng.normal(size=len(kx)))/(1+kx*kx+ky*ky)**1.5
    A*=.12/np.sqrt(2*np.sum(abs(A)**2))
    cx=1.;cy=.25;a=1. if scheme==0 else 2.
    sx=np.sin(kx*h)/h;sy=np.sin(ky*h)/h
    lx=4*np.sin(kx*h/2)**2/h**2;ly=4*np.sin(ky*h/2)**2/h**2
    lam=-1j*(cx*sx+cy*sy)-(D[0,0]+a*cx*h/2)*lx-(D[1,1]+a*cy*h/2)*ly-2*D[0,1]*sx*sy
    U=A[None,:]*np.exp(t[:,None]*lam[None,:])
    sigma=noise*.12
    nr=np.random.default_rng(np.random.SeedSequence([seed,n,scheme,113]))
    # DFT of independent real-space N(0,sigma^2) noise: selected non-conjugate modes.
    U+=sigma/n/np.sqrt(2)*(nr.normal(size=U.shape)+1j*nr.normal(size=U.shape))
    rows=[];ys=[];width=.78
    for center in np.linspace(width/2,t[-1]-width/2,4):
        z=(t-center)/(width/2);inside=abs(z)<1
        phi=np.where(inside,(1-z*z)**4,0);dp=np.where(inside,-8*z*(1-z*z)**3/(width/2),0)
        integ=simpson(U*phi[:,None],x=t,axis=0);y=-simpson(U*dp[:,None],x=t,axis=0)
        X=np.column_stack([integ*(1j*kx)**dx*(1j*ky)**dy*(2 if (dx,dy)==(1,1) else 1) for dx,dy in DERS])
        rows.extend([X.real,X.imag]);ys.extend([y.real,y.imag])
    return np.vstack(rows),np.concatenate(ys)

def tensor_task(job):
    case,trial=job;D=np.eye(2)*.002 if case==0 else np.array([[.003,.001],[.001,.003]])
    rows=[]
    for noise in [0,.001,.01]:
        rs=[]
        for s in [0,1]:
            for n in [32,48,64,96,128]:
                for rep in range(4):
                    X,y=modal_data(n,s,50000+trial*4+rep,D,noise);rs.append(dict(X=X,y=y,h=2*np.pi/n,scheme=s,rep=rep))
        for method in ['finest','pooled','joint2','joint3','richardson2']:
            if method=='richardson2':c=separate_extrapolation(rs)
            else:c,_=fit(rs, 'joint' if method.startswith('joint') else method, powers=(1,2,3) if method=='joint3' else (1,2))
            if method=='finest':c,_=fit([r for r in rs if r['scheme']==0],'finest')
            estimate=np.array([[c[2],c[3]],[c[3],c[4]]]);ev=np.linalg.eigvalsh(estimate)
            rows.append(dict(case=case,trial=trial,noise=noise,method=method,D=estimate.tolist(),error=float(np.linalg.norm(estimate-D)/np.linalg.norm(D)),ratio=float(ev[1]/ev[0]) if ev[0]>0 else None))
    return rows

def diagnostic_runs():
    t=np.linspace(0,.8,161);rows=[]
    for reps in [2,8,32]:
        for grids in [[128],[96,128],[64,96,128],[32,48,64,96,128]]:
            rs=[]
            for s in [0,1]:
                for n in grids:
                    for rep in range(reps):
                        u,_=solve(initial(n,9000+rep,0),2*np.pi/n,0,.002,s,t);X,y=weak(u,t)
                        rs.append(dict(X=X,y=y,h=2*np.pi/n,scheme=s,rep=rep))
            c,d=fit(rs,'joint');rows.append(dict(reps=reps,grids=grids,**d))
    # A shared resolution-independent bias is deliberately undetectable by refinement.
    rs=[]
    for s in [0,1]:
        for n in [32,48,64,96,128]:
            for rep in range(8):
                u,_=solve(initial(n,9000+rep,0),2*np.pi/n,0,.006,s,t);X,y=weak(u,t)
                rs.append(dict(X=X,y=y,h=2*np.pi/n,scheme=s,rep=rep))
    c,d=fit(rs,'joint',powers=(1,2,3));negative={'declared_physical_diffusion':.002,'shared_unmodelled_diffusion':.004,'estimated_diffusion':float(c[6]),**d}
    # Reference and weak-quadrature convergence, evaluated on withheld initial conditions.
    conv=[]
    for kind in range(3):
        for nu in [0.,.002,.02]:
            u=spectral_reference(256,777777,kind,nu,t);v=spectral_reference(512,777777,kind,nu,t)
            X,y=weak(u,t);tt=np.linspace(0,.8,321);w=spectral_reference(256,777777,kind,nu,tt);XX,yy=weak(w,tt)
            conv.append(dict(kind=kind,nu=nu,spatial_relative_error=float(np.linalg.norm(u-v[:,::2])/np.linalg.norm(v[:,::2])),weak_residual=float(np.linalg.norm(X@truth(kind,nu)-y)/np.linalg.norm(y)),weak_residual_doubled=float(np.linalg.norm(XX@truth(kind,nu)-yy)/np.linalg.norm(yy))))
    return rows,negative,conv

if __name__=='__main__':
    t0=time.perf_counter()
    output=Path(os.environ.get('PHYSICS_RESULTS_DIR','results'));output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get('PHYSICS_WORKERS','8'))) as ex: rows=[r for block in ex.map(tensor_task,[(c,s) for c in range(2) for s in range(40)]) for r in block]
    (output/'tensor.json').write_text(json.dumps(rows))
    d,n,c=diagnostic_runs();(output/'diagnostics.json').write_text(json.dumps({'rank':d,'shared_bias':n,'convergence':c},indent=2))
    print('completed',len(rows),'tensor estimates; seconds',time.perf_counter()-t0,flush=True)
