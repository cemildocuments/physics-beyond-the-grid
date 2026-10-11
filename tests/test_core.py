import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from core import *

class NumericalChecks(unittest.TestCase):
 def test_linear_solver_against_discrete_symbol(self):
  n=64;h=2*np.pi/n;t=np.linspace(0,.8,161);nu=.002;u0=initial(n,88,0);u,_=solve(u0,h,0,nu,0,t)
  k=np.fft.fftfreq(n)*n;lam=-1j*np.sin(k*h)/h-(nu+h/2)*4*np.sin(k*h/2)**2/h**2
  exact=np.fft.ifft(np.fft.fft(u0)[None,:]*np.exp(t[:,None]*lam[None,:]),axis=1).real
  self.assertLess(np.linalg.norm(u-exact)/np.linalg.norm(exact),2.e-9)
 def test_weak_form_analytic_transport(self):
  t=np.linspace(0,.8,321);u=spectral_reference(128,12,0,.002,t,4);X,y=weak(u,t)
  self.assertLess(np.linalg.norm(X@truth(0,.002)-y)/np.linalg.norm(y),1.e-6)
 def test_rank_obstruction(self):
  rng=np.random.default_rng(1)
  for grids in [[1.],[1.,.5]]:
   rs=[dict(X=rng.normal(size=(30,3)),y=rng.normal(size=30),h=h,scheme=s) for s in [0,1] for h in grids]
   X,y,Z=build_design(rs);A,b,scale,s=project(X,y,Z)
   self.assertLess(np.linalg.norm(A)/np.linalg.norm(X),1.e-12)
 def test_exact_continuum_recovery(self):
  rng=np.random.default_rng(2);c=np.array([1.,0.,.002]);a=rng.normal(size=(2,2,3));rs=[]
  for s in [0,1]:
   for h in [1.,.75,.5,.25]:
    X=rng.normal(size=(40,3));y=X@(c+h*a[s,0]+h*h*a[s,1]);rs.append(dict(X=X,y=y,h=h,scheme=s))
  estimate,_=fit(rs,'joint');np.testing.assert_allclose(estimate,c,atol=2.e-12)
 def test_weighted_extrapolation_equivalence(self):
  rng=np.random.default_rng(3);rs=[]
  for s in [0,1]:
   for h in [1.,.75,.5,.25]:
    X=rng.normal(size=(40,3))*rng.uniform(.2,2,3);y=rng.normal(size=40);rs.append(dict(X=X,y=y,h=h,scheme=s))
  X,y,Z=build_design(rs);A=np.column_stack([X,Z]);full=np.linalg.lstsq(A,y,rcond=1.e-10)[0][:3]
  estimate,_=fit(rs,'joint');np.testing.assert_allclose(estimate,full,atol=2.e-12)
  normal=np.zeros((A.shape[1],A.shape[1]));rhs_=np.zeros(A.shape[1])
  for r in rs:
   h=r['h'];s=r['scheme'];B=np.zeros((3,A.shape[1]));B[:,:3]=np.eye(3)
   for j in range(2):B[:,3+(s*2+j)*3:3+(s*2+j+1)*3]=np.eye(3)*h**(j+1)
   G=r['X'].T@r['X'];chat=np.linalg.lstsq(r['X'],r['y'],rcond=None)[0]
   normal+=B.T@G@B;rhs_+=B.T@G@chat
  weighted=np.linalg.solve(normal,rhs_)[:3];np.testing.assert_allclose(estimate,weighted,atol=1.e-10)

if __name__=='__main__':unittest.main()
