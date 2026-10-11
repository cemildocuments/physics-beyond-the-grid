"""Periodic simulators and weak-form continuum identification, float64 throughout."""
import numpy as np
from scipy.integrate import simpson
from scipy.linalg import lstsq, svd
from numba import njit

TERMS=[(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(1,3),(2,3),(1,4)]
LABELS=['u','u^2','u^3','u_x','(u^2)_x','(u^3)_x','u_xx','(u^2)_xx','(u^3)_xx','u_xxx','(u^2)_xxx','u_xxxx']

@njit(cache=True)
def rhs(u,h,kind,nu,scheme):
    n=u.size; out=np.empty(n)
    # Global Lax-Friedrichs numerical flux. Scheme 0/1: dissipation 1/2.
    # Scheme 2: nondissipative centred, scheme 3: fourth-order centred.
    a=1.0 if scheme==0 else (2.0 if scheme==1 else 0.0)
    for i in range(n):
        im=(i-1)%n; ip=(i+1)%n
        lap=(u[ip]-2*u[i]+u[im])/(h*h)
        fip=.5*u[ip]*u[ip] if kind==1 else u[ip]
        fim=.5*u[im]*u[im] if kind==1 else u[im]
        adv=(fip-fim)/(2*h)
        if scheme==3:
            ipp=(i+2)%n; imm=(i-2)%n
            fipp=.5*u[ipp]*u[ipp] if kind==1 else u[ipp]
            fimm=.5*u[imm]*u[imm] if kind==1 else u[imm]
            adv=(-fipp+8*fip-8*fim+fimm)/(12*h)
            lap=(-u[ipp]+16*u[ip]-30*u[i]+16*u[im]-u[imm])/(12*h*h)
        out[i]=-adv+(nu+.5*a*h)*lap
        if kind==2: out[i]+=u[i]*(1-u[i])
    return out

@njit(cache=True)
def solve(u0,h,kind,nu,scheme,times):
    u=u0.copy(); out=np.empty((times.size,u.size)); out[0]=u
    a=1.0 if scheme==0 else (2.0 if scheme==1 else 0.0)
    dtmax=min(.18*h, .20*h*h/max(nu+.5*a*h,1.e-10))
    nsteps=0
    for j in range(1,times.size):
        steps=int(np.ceil((times[j]-times[j-1])/dtmax)); dt=(times[j]-times[j-1])/steps
        for s in range(steps):
            k1=rhs(u,h,kind,nu,scheme);k2=rhs(u+.5*dt*k1,h,kind,nu,scheme)
            k3=rhs(u+.5*dt*k2,h,kind,nu,scheme);k4=rhs(u+dt*k3,h,kind,nu,scheme)
            u=u+dt*(k1+2*k2+2*k3+k4)/6
        out[j]=u; nsteps+=steps
    return out,nsteps

def initial(n,seed,kind,band=4):
    rng=np.random.default_rng(seed);x=np.arange(n)*2*np.pi/n
    coeff=rng.normal(size=(2,band))/(np.arange(1,band+1)**1.5)
    v=sum(coeff[0,k-1]*np.cos(k*x)+coeff[1,k-1]*np.sin(k*x) for k in range(1,band+1))
    # Use a grid-independent L1 amplitude bound, not sample-dependent normalisation.
    v=v/np.sum(np.abs(coeff))
    return (.3+.18*v) if kind==2 else (.3+.55*v)

def weak(u,t,noise_sigma=0.0,kmax=6,windows=4):
    # Periodic Fourier test functions and compact polynomial time tests.
    n=u.shape[1];ks=np.arange(0,kmax+1)
    powers=[u,u*u-noise_sigma**2,u*u*u-3*noise_sigma**2*u]
    fs=[np.fft.rfft(v,axis=1)[:,:kmax+1]/n for v in powers]
    rows=[]; ys=[]
    width=(t[-1]-t[0])*.65
    for center in np.linspace(t[0]+width/2,t[-1]-width/2,windows):
        z=(t-center)/(width/2); inside=np.abs(z)<1
        phi=np.where(inside,(1-z*z)**4,0)
        dphi=np.where(inside,-8*z*(1-z*z)**3/(width/2),0)
        ints=[simpson(f*phi[:,None],x=t,axis=0) for f in fs]
        y=-simpson(fs[0]*dphi[:,None],x=t,axis=0)
        X=np.column_stack([ints[p-1]*(1j*ks)**d for p,d in TERMS])
        rows.extend([X.real,X.imag[1:]]);ys.extend([y.real,y.imag[1:]])
    return np.vstack(rows),np.concatenate(ys)

def truth(kind,nu):
    c=np.zeros(len(TERMS));c[6]=nu
    if kind==1:c[4]=-.5
    else:c[3]=-1
    if kind==2:c[0]=1;c[1]=-1
    return c

def build_design(records,powers=(1,2),scheme_specific=True):
    X=np.vstack([r['X'] for r in records]);y=np.concatenate([r['y'] for r in records]);p=X.shape[1]
    groups=sorted(set(r['scheme'] for r in records)) if scheme_specific else [0]
    Z=np.zeros((len(y),len(groups)*len(powers)*p));offset=0
    hmax=max(r['h'] for r in records)
    for r in records:
        n=len(r['y']); g=groups.index(r['scheme']) if scheme_specific else 0
        for j,power in enumerate(powers):Z[offset:offset+n,(g*len(powers)+j)*p:(g*len(powers)+j+1)*p]=r['X']*(r['h']/hmax)**power
        offset+=n
    return X,y,Z

def project(X,y,Z):
    scale=np.linalg.norm(X,axis=0);scale=np.maximum(scale,1.e-14)
    if Z.shape[1]:
        zn=np.maximum(np.linalg.norm(Z,axis=0),1.e-14)
        U,s,_=svd(Z/zn,full_matrices=False)
        Q=U[:,s>max(s[0]*1.e-10,1.e-12)]
        XP=X-Q@(Q.T@X);yp=y-Q@(Q.T@y)
    else:XP=X;yp=y
    s=np.linalg.svd(XP/scale,compute_uv=False)
    return XP,yp,scale,s

def fit(records,method='joint',powers=(1,2),threshold=0.):
    X,y,Z=build_design(records,powers)
    if method=='pooled':Z=np.zeros((len(y),0))
    if method=='finest':
        hmin=min(r['h'] for r in records)
        X,y,_=build_design([r for r in records if r['h']==hmin],())
        Z=np.zeros((len(y),0))
    A,b,scale,s=project(X,y,Z)
    active=np.ones(X.shape[1],dtype=bool);c=np.zeros(X.shape[1])
    for _ in range(12):
        c[:]=0
        if not np.any(active):break
        c[active]=lstsq(A[:,active]/scale[active],b,cond=1.e-10)[0]/scale[active]
        nxt=np.abs(c)*scale>=threshold*np.linalg.norm(y)
        if np.array_equal(nxt,active):break
        active=nxt
    resid=b-A@c
    return c,{'sigma_min':float(s[-1]),'residual':float(np.linalg.norm(resid)/np.linalg.norm(y)),'rank':int(np.sum(s>1.e-9))}

def separate_extrapolation(records,powers=(1,2)):
    # Unweighted coefficient Richardson extrapolation, including all dictionary terms.
    cs=[];V=[];groups=sorted(set(r['scheme'] for r in records))
    for group in groups:
        for h in sorted(set(r['h'] for r in records if r['scheme']==group)):
            rs=[r for r in records if r['scheme']==group and r['h']==h]
            c,_=fit(rs,'pooled');cs.append(c)
            v=[1.]+[float(group==g)*h**p for g in groups for p in powers];V.append(v)
    return np.linalg.lstsq(V,np.array(cs),rcond=1.e-10)[0][0]

def fit_interval(records,method='joint',powers=(1,2)):
    from scipy.stats import t as student_t
    if method=='finest':
        h=min(r['h'] for r in records); records=[r for r in records if r['h']==h and r['scheme']==0]
    X,y,Z=build_design(records,powers if method=='joint' else ())
    A,b,scale,s=project(X,y,Z)
    An=A/scale; inv=np.linalg.pinv(An,rcond=1.e-10)/scale[:,None]
    c=inv@b
    residual=b-A@c
    groups=sorted(set(r['rep'] for r in records));scores=np.zeros((len(groups),X.shape[1]));offset=0
    for r in records:
        nn=len(r['y']);scores[groups.index(r['rep'])]+=inv[:,offset:offset+nn]@residual[offset:offset+nn];offset+=nn
    se=np.sqrt(np.sum(scores*scores,axis=0)*len(groups)/(len(groups)-1))
    hw=student_t.ppf(.975,len(groups)-1)*se
    return c,hw,{'sigma_min':float(s[-1]),'residual':float(np.linalg.norm(residual)/np.linalg.norm(y)),'rank':int(np.sum(s>1.e-9))}

def spectral_reference(n,seed,kind,nu,times,band=6):
    from scipy.integrate import solve_ivp
    u0=initial(n,seed,kind,band);k=np.fft.fftfreq(n)*n
    if kind==0:
        return np.fft.ifft(np.fft.fft(u0)[None,:]*np.exp((-1j*k-nu*k*k)[None,:]*times[:,None]),axis=1).real
    mask=np.abs(k)<n/3
    def fun(t,u):
        uh=np.fft.fft(u)
        flux=.5*u*u if kind==1 else u
        fh=np.fft.fft(flux)*mask
        out=np.fft.ifft(-1j*k*fh-nu*k*k*uh).real
        if kind==2:out+=np.fft.ifft(np.fft.fft(u*(1-u))*mask).real
        return out
    sol=solve_ivp(fun,(times[0],times[-1]),u0,method='DOP853',t_eval=times,rtol=2.e-11,atol=2.e-13)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y.T
