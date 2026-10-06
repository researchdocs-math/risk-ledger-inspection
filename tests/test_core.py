import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import betaln
from risk_inspection.core import upper_cs,simulate,fluid_plan

class RiskTests(unittest.TestCase):
 def test_mixture_endpoints(self):
  for n in [1,2,10,100,1000]:
   vals=[]
   for k in sorted(set([0,1,n//2,max(0,n-1),n])):
    u=upper_cs(n,k,.005);vals.append(u)
    self.assertGreaterEqual(u,k/n)
    if k<n:
     logm=betaln(k+.5,n-k+.5)-betaln(.5,.5)-k*np.log(u)-(n-k)*np.log1p(-u)
     self.assertAlmostEqual(logm,np.log(200.),places=7)
   self.assertTrue(np.all(np.diff(vals)>=-1e-12))
 def test_fifo_equal_delay(self):
  rng=np.random.default_rng(314);g=rng.integers(0,2,(200,4));p=np.array([.02,.2]);y=(rng.random(g.shape)<p[g]).astype(np.int64)
  cfg=dict(alpha=.05,capacity=.5,queue_capacity=20,delta_cal=.025,plan_interval=25,exploration=.02)
  common=(g,y,p[g],np.full(g.shape,3,dtype=np.int64),np.array([30,30]),np.array([1,6]),np.array([.5,.5]),cfg)
  a,ta=simulate(*common,'ledger',8);b,tb=simulate(*common,'completed',8)
  np.testing.assert_allclose(ta,tb,rtol=0,atol=0)
  self.assertGreaterEqual(a['min_balance'],-1e-8)
  self.assertLessEqual(a['initiated'],len(g)*2)
 def test_informative_delay_prefix(self):
  rng=np.random.default_rng(6);g=rng.integers(0,2,(1000,4));p=np.array([.03,.35]);y=(rng.random(g.shape)<p[g]).astype(np.int64)
  cfg=dict(alpha=.02,capacity=.5,queue_capacity=20,delta_cal=.025,plan_interval=50,exploration=.02)
  r,tr=simulate(g,y,p[g],(2+200*y).astype(np.int64),np.array([20,20]),np.array([1,7]),np.array([.5,.5]),cfg,'ledger',6)
  self.assertGreaterEqual(r['min_balance'],-1e-8);self.assertEqual(r['potential_violation'],0)
  self.assertTrue(np.all(np.diff(tr[:,8:],axis=0)<=1e-10))
 def test_zero_feedback_no_credit(self):
  g=np.zeros((10,4),dtype=np.int64);y=g.copy();cfg=dict(alpha=.02,capacity=.25,queue_capacity=8,delta_cal=.025,plan_interval=5,exploration=.02)
  r,_=simulate(g,y,np.ones(g.shape)*.5,np.ones(g.shape,dtype=np.int64)*100,np.array([0]),np.array([0]),np.array([1.]),cfg,'ledger',12)
  self.assertEqual(r['released'],0);self.assertLessEqual(r['initiated'],10);self.assertLessEqual(r['unfinished'],18)
 def test_fluid_monotonic_capacity(self):
  w=np.array([.6,.4]);p=np.array([.02,.3]);v=[fluid_plan(w,p,c,.02)[2] for c in [.1,.25,.5,1.]]
  self.assertTrue(np.all(np.diff(v)>=-1e-10));self.assertAlmostEqual(v[-1],np.dot(w,1-p))
if __name__=='__main__':unittest.main()
