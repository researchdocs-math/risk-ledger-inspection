"""Finite held-out manufacturing replays with simulated inspection resources."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,warnings,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd,numpy as np
from risk_inspection.core import simulate
warnings.filterwarnings('ignore',message='Unrecognized options detected')
ROOT=Path(__file__).resolve().parent

def job(args):
 name,capacity,delay,seed=args
 df=pd.read_csv(ROOT/'data'/'scored'/f'{name}.csv');tr=df[df.split=='train'];cal=df[df.split=='calibration'];test=df[df.split=='test'].copy()
 # SECOM keeps the chronological test order. KSDD2 uses a fixed ID order.
 # Repeated seeds change policy randomization only; no new physical specimens are created.
 K=3;batch=4;assert len(test)%batch==0
 groups=test.group.values.reshape(-1,batch).astype(np.int64);labels=test.label.values.reshape(-1,batch).astype(np.int64)
 truth=np.zeros(groups.shape)
 for g in range(K):truth[groups==g]=labels[groups==g].mean() if np.any(groups==g) else 0.
 # truth is evaluation-only; it is never supplied to the policy's risk estimator.
 cal_n=np.array([(cal.group==g).sum() for g in range(K)]);cal_k=np.array([cal.loc[cal.group==g,'label'].sum() for g in range(K)])
 w=np.array([(tr.group==g).mean() for g in range(K)])
 cfg={'alpha':.02,'capacity':capacity,'queue_capacity':20,'delta_cal':.025,'plan_interval':25,'exploration':.02}
 delays=np.full(groups.shape,delay,dtype=np.int64)+labels*delay*4
 rows=[]
 for method in ['ledger','static','plugin','completed','pending_credit','allocation','inspect_all','threshold','pending_only','prefix_only']:
  r,trace=simulate(groups,labels,truth,delays,cal_n,cal_k,w,cfg,method,seed)
  r.update(dataset=name,capacity=capacity,delay=delay);rows.append(r)
  if seed==0 and capacity==.25 and delay==2:np.savez_compressed(ROOT/'results'/f'trace_{name}_{method}.npz',trace=trace)
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
 (ROOT/'results').mkdir(exist_ok=True)
 t=time.monotonic();jobs=[(n,c,d,s) for n in ['secom','ksdd2'] for c in [.25,.5] for d in [2,20] for s in range(100)]
 with ProcessPoolExecutor(args.workers) as ex:rows=[r for rr in ex.map(job,jobs) for r in rr]
 df=pd.DataFrame(rows);df.to_csv(ROOT/'results'/'replays.csv',index=False)
 print(df[(df.capacity==.25)&(df.delay==2)].groupby(['dataset','method'])[['good_yield','defect_rate','inspection_rate','unfinished']].mean().to_string());print('seconds',time.monotonic()-t)
