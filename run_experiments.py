"""Run reproducible parallel CPU experiments. See configs and README."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,time,argparse,warnings
from pathlib import Path
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor
from risk_inspection.core import simulate,fluid_plan
warnings.filterwarnings('ignore',message='Unrecognized options detected')
ROOT=Path(__file__).resolve().parent
P=np.array([.005,.025,.08,.25,.55]);W=np.array([.45,.25,.15,.1,.05])
def run_job(job):
 cfg,seed=job;rng=np.random.default_rng(seed);T=cfg['slots'];batch=cfg['batch'];K=len(P)
 groups=rng.choice(K,size=(T,batch),p=W);truth=P[groups].copy()
 if cfg.get('drift',False):truth[T//2:]=np.minimum(.95,truth[T//2:]*3+.02)
 labels=(rng.random((T,batch))<truth).astype(np.int64)
 d=cfg['delay'];delays=np.full((T,batch),d,dtype=np.int64)
 if cfg.get('informative',False):delays+=labels*cfg.get('defect_extra_delay',200)
 # Same labels and potential delays for every method.
 cal_n=np.full(K,cfg['calibration_per_group'],dtype=np.int64);cal_k=rng.binomial(cal_n,P)
 rows=[]
 for method in cfg['methods']:
  res,tr=simulate(groups,labels,truth,delays,cal_n,cal_k,W,cfg,method,seed)
  res.update({k:v for k,v in cfg.items() if not isinstance(v,list)});rows.append(res)
  if cfg.get('save_trace',False) and seed==cfg['seeds'][0]:
   np.savez_compressed(ROOT/'results'/f"trace_{cfg['name']}_{method}.npz",trace=tr,cal_n=cal_n,cal_k=cal_k)
 _,_,bound=fluid_plan(W,P,cfg['capacity'],cfg['alpha'])
 for r in rows:r['fluid_good_yield']=bound
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/main.json');ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
 (ROOT/'results').mkdir(exist_ok=True)
 cases=json.loads((ROOT/args.config).read_text());jobs=[(c,s) for c in cases for s in c['seeds']]
 t=time.monotonic();rows=[]
 with ProcessPoolExecutor(args.workers) as ex:
  for j,res in enumerate(ex.map(run_job,jobs)):
   rows.extend(res)
   if (j+1)%10==0:print(f'{j+1}/{len(jobs)} replications, {time.monotonic()-t:.1f}s',flush=True)
 out=ROOT/'results'/f'{Path(args.config).stem}.csv';pd.DataFrame(rows).to_csv(out,index=False)
 print(out, 'seconds',time.monotonic()-t,flush=True)
 print(pd.DataFrame(rows).groupby('method')[['good_yield','defect_rate','true_risk_rate','cs_violation','potential_violation']].mean().to_string())
if __name__=='__main__':main()
