"""Compute confidence intervals, paired gains, and violation frequencies."""
from pathlib import Path
import json,numpy as np,pandas as pd
from scipy.stats import t,beta
R=Path(__file__).resolve().parent;x=pd.read_csv(R/'results/main.csv');y=pd.read_csv(R/'results/replays.csv')
def ci(v):
 a=np.asarray(v);mean=a.mean();h=t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))
 return {'mean':float(mean),'low':float(mean-h),'high':float(mean+h),'n':len(a)}
summary={}
for name in x.name.unique():
 a=x[x.name==name];summary[name]={}
 for method,z in a.groupby('method'):
  summary[name][method]={k:ci(z[k]) for k in ['good_yield','defect_rate','true_risk_rate','initiation_rate','mean_queue_wait','mean_good_inspection_latency','unfinished']}
  for k in ['cs_violation','potential_violation','envelope_violation']:
   n=len(z);s=int(z[k].sum());summary[name][method][k]={'failures':s,'n':n,'upper95':float(beta.ppf(.95,s+1,n-s)) if s<n else 1.}
main=x[x.name=='main'].pivot(index='seed',columns='method',values='good_yield');summary['paired_main']={m:ci(main.ledger-main[m]) for m in main if m!='ledger'}
summary['stationary_pooled']={}
for method,z in x[x.name!='drift'].groupby('method'):
 summary['stationary_pooled'][method]={k:int(z[k].sum()) for k in ['cs_violation','potential_violation','envelope_violation']};summary['stationary_pooled'][method]['n']=len(z)
summary['replays']={}
for (name,cap,delay),a in y.groupby(['dataset','capacity','delay']):
 key=f'{name}_{cap}_{delay}';summary['replays'][key]={}
 for m,z in a.groupby('method'):summary['replays'][key][m]={k:ci(z[k]) for k in ['good_yield','defect_rate','initiation_rate','mean_queue_wait','unfinished']}
 a=a.pivot(index='seed',columns='method',values='good_yield');summary['replays'][key]['paired_static']=ci(a.ledger-a.static)
(R/'results/summary.json').write_text(json.dumps(summary,indent=2))
print('Main effect',summary['paired_main']['static'])
print('Pooled stationary',summary['stationary_pooled']['ledger'])
print('Bounds',x[x.name=='main'].fluid_good_yield.iloc[0])
for key in summary['replays']:
 print(key,'gain',summary['replays'][key]['paired_static'],'ledger',summary['replays'][key]['ledger']['good_yield']['mean'],summary['replays'][key]['ledger']['defect_rate']['mean'])
