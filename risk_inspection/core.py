"""Risk-budgeted inspection with delayed, potentially informative feedback."""
import math
import numpy as np
from numba import njit
from scipy.optimize import linprog

METHODS = ['ledger','static','plugin','completed','pending_credit','allocation','inspect_all','oracle','threshold','pending_only','prefix_only']

@njit(cache=True)
def upper_cs(n, k, delta):
    """Upper endpoint of Jeffreys beta-binomial mixture confidence sequence."""
    if n == 0 or k == n:
        return 1.0
    a = math.lgamma(k+.5)+math.lgamma(n-k+.5)-math.lgamma(n+1.)-2*math.lgamma(.5)
    lo = k/n
    hi = 1.-1e-15
    level = math.log(1./delta)
    for j in range(45):
        p = (lo+hi)/2
        val = a - k*math.log(max(p,1e-300))-(n-k)*math.log1p(-p)
        if val > level: hi=p
        else: lo=p
    return hi


def fluid_plan(w, p, capacity, alpha, exploration=0.):
    """Fractions bypassed and inspected; maximize conservative good throughput."""
    k=len(w)
    obj=-np.concatenate([w*(1-p),w*(1-p)])
    A=np.vstack([np.concatenate([w*(p-alpha),-alpha*w*(1-p)]),
                 np.concatenate([np.zeros(k),w]),np.concatenate([np.eye(k),np.eye(k)],axis=1)])
    rhs=np.concatenate([[0.,capacity],np.ones(k)])
    ans=linprog(obj,A_ub=A,b_ub=rhs,bounds=[(0,1)]*k+[(exploration,1)]*k,method='highs',options={'threads':1})
    if not ans.success: raise RuntimeError(ans.message)
    return ans.x[:k],ans.x[k:],-ans.fun

@njit(cache=True)
def block(t0,t1,groups,labels,truth,truth_by_group,delays,routing,cal_n,cal_k,state,counts,upper,
          q,heads,nxt,admit_next,admit_head,admit_tail,revealed,bypass_prob,inspect_prob,alpha,delta,service,qcap,method,trace):
    """One block of FIFO inspection admissions and delayed completions.

    state: R,D,charge,true_mass,G,reject_good,reject_bad,admitted,completed,
    queue_head,queue_tail,queue_len,wait_sum,max_queue,max_pending,min_balance,
    potential_violation,envelope_violation,cs_violation,latency_sum,quality_fail.
    counts: admitted by group, completed by group, completed defects by group.
    """
    batch=groups.shape[1];K=len(cal_n)
    for t in range(t0,t1):
        idx=heads[t]
        while idx != -1:
            g=groups.ravel()[idx];y=labels.ravel()[idx]
            counts[1,g]+=1;counts[2,g]+=y;state[8]+=1
            revealed[idx]=1
            # Use the largest fully revealed prefix of each admission-order stream.
            while admit_head[g] != -1 and revealed[admit_head[g]]==1:
                jj=admit_head[g];counts[3,g]+=1;counts[4,g]+=labels.ravel()[jj]
                admit_head[g]=admit_next[jj]
                if admit_head[g]==-1:admit_tail[g]=-1
            if y==0:
                state[0]+=1;state[4]+=1
                state[19]+=t-idx//batch
            else:state[20]+=1
            idx=nxt[idx]
        for g in range(K):
            if method in (0,4,9,10):
                n=cal_n[g]+counts[0,g]
                kk=cal_k[g]+counts[2,g]+counts[0,g]-counts[1,g]
                if method!=10:upper[g]=min(upper[g],upper_cs(n,kk,delta/K))
                if method!=9:upper[g]=min(upper[g],upper_cs(cal_n[g]+counts[3,g],cal_k[g]+counts[4,g],delta/K))
            elif method==3:
                n=cal_n[g]+counts[1,g];kk=cal_k[g]+counts[2,g]
                upper[g]=min(upper[g],upper_cs(n,kk,delta/K))
            elif method==2 or method==5:
                upper[g]=(cal_k[g]+counts[2,g]+.5)/(cal_n[g]+counts[1,g]+1.)
        # Existing FIFO queue gets the limited initiation capacity.
        available=service
        while available>0 and state[11]>0:
            ix=int(q[int(state[9])]);state[9]=(int(state[9])+1)%qcap;state[11]-=1
            due=t+delays.ravel()[ix]
            nxt[ix]=heads[due];heads[due]=ix;state[12]+=t-ix//batch
            available-=1
        for j in range(batch):
            ix=t*batch+j;g=groups[t,j];y=labels[t,j];u=routing[t,j]
            # These latent values are only read by the evaluator and event generator.
            inspect=False;release=False
            if method==6: inspect=True
            elif method==8:
                if upper[g]<=alpha:release=True
                else:inspect=True
            elif u < inspect_prob[g]:inspect=True
            elif u < inspect_prob[g]+bypass_prob[g]:
                credit=alpha*state[0]-state[2]
                if method==4:credit += alpha*(state[7]-state[8])
                if method==5 or credit+alpha-upper[g] >= -1e-12:release=True
                else:inspect=True
            if release:
                state[0]+=1;state[1]+=y;state[2]+=upper[g];state[3]+=truth[t,j]
            elif inspect:
                if available>0:
                    if admit_tail[g] != -1:admit_next[admit_tail[g]]=ix
                    else:admit_head[g]=ix
                    admit_tail[g]=ix
                    due=t+delays[t,j];nxt[ix]=heads[due];heads[due]=ix;available-=1
                    counts[0,g]+=1;state[7]+=1
                elif state[11]<qcap:
                    if admit_tail[g] != -1:admit_next[admit_tail[g]]=ix
                    else:admit_head[g]=ix
                    admit_tail[g]=ix
                    q[int(state[10])]=ix;state[10]=(int(state[10])+1)%qcap;state[11]+=1
                    counts[0,g]+=1;state[7]+=1
                else:
                    state[5]+=1-y;state[6]+=y
            else:
                state[5]+=1-y;state[6]+=y
            bal=alpha*state[0]-state[2]
            state[15]=min(state[15],bal)
            if state[3]>alpha*state[0]+1e-9:state[16]=1
            # Fixed grid of exponential supermartingales: union over seven lambda values.
            boundary=1e100
            for lam in (.025,.05,.1,.2,.4,.8,1.6):
                z=(math.expm1(lam)*alpha*state[0]+math.log(7/.025))/lam
                boundary=min(boundary,z)
            if state[1]>boundary+1e-9:state[17]=1
        state[13]=max(state[13],state[11]);state[14]=max(state[14],state[7]-state[8])
        for g in range(K):
            # Evaluation-only stratum risks, including groups absent from this batch.
            if truth_by_group[t,g]>upper[g]+1e-10:state[18]=1
        if t%50==0 or t==len(groups)-1:
            rr=t//50
            trace[rr,0]=t+1;trace[rr,1]=state[0];trace[rr,2]=state[1]
            trace[rr,3]=state[2];trace[rr,4]=state[3];trace[rr,5]=state[11]
            trace[rr,6]=state[7]-state[8];trace[rr,7]=alpha*state[0]-state[2]
            for g in range(K):trace[rr,8+g]=upper[g]


def simulate(groups,labels,truth,delays,cal_n,cal_k,w,config,method,seed):
    T,batch=groups.shape;K=len(w);m=METHODS.index(method)
    alpha=config['alpha'];capacity=config['capacity'];service=max(1,round(capacity*batch));qcap=config.get('queue_capacity',100)
    delta=config.get('delta_cal',.025);rng=np.random.default_rng(seed+987654)
    routing=rng.random(groups.shape)
    p0=np.array([truth[groups==g][0] if np.any(groups==g) else 0. for g in range(K)])
    truth_by_group=np.tile(p0,(T,1))
    if config.get('drift',False):
        truth_by_group[T//2:]=np.minimum(.95,3*p0+.02)
    state=np.zeros(21);counts=np.zeros((5,K),dtype=np.int64)
    upper=np.array([upper_cs(int(n),int(k),delta/K) for n,k in zip(cal_n,cal_k)])
    if m in [2,5]: upper=(cal_k+.5)/(cal_n+1.)
    if m==7: upper=np.array([truth[groups==g][0] if np.any(groups==g) else 1. for g in range(K)])
    q=np.zeros(qcap,dtype=np.int64);heads=np.full(T+int(delays.max())+qcap+batch+10,-1,dtype=np.int64);nxt=np.full(T*batch,-1,dtype=np.int64)
    admit_next=np.full(T*batch,-1,dtype=np.int64);admit_head=np.full(K,-1,dtype=np.int64);admit_tail=np.full(K,-1,dtype=np.int64);revealed=np.zeros(T*batch,dtype=np.int64)
    trace=np.zeros(((T-1)//50+1,8+K))
    plan_interval=config.get('plan_interval',250)
    for t in range(0,T,plan_interval):
        if m==5 and t>0: pass # static empirical allocation, fixed at deployment
        else:b,i,_=fluid_plan(w,upper,service/batch,alpha,config.get('exploration',.02) if m not in [6,8] else 0.)
        block(t,min(t+plan_interval,T),groups,labels,truth,truth_by_group,delays,routing,cal_n,cal_k,state,counts,upper,q,heads,nxt,admit_next,admit_head,admit_tail,revealed,b,i,alpha,delta,service,qcap,m,trace)
    N=T*batch;R,D,C,A,G,rej_good,rej_bad,admitted,completed=state[:9]
    out={'method':method,'seed':seed,'n':N,'released':R,'escaped':D,'good_released':R-D,
         'good_yield':(R-D)/N,'release_rate':R/N,'defect_rate':D/max(1,R),'charge_rate':C/max(1,R),
         'true_risk_rate':A/max(1,R),'quarantined_good':rej_good,'quarantined_bad':rej_bad,
         'inspected':admitted,'completed':completed,'unfinished':admitted-completed,
         'inspection_rate':admitted/N,'initiated':admitted-state[11],'initiation_rate':(admitted-state[11])/N,'waiting_at_end':state[11],'mean_queue_wait':state[12]/max(1,admitted-state[11]),
         'mean_good_inspection_latency':state[19]/max(1,G),'max_queue':state[13],'max_pending':state[14],
         'min_balance':state[15],'potential_violation':state[16],'envelope_violation':state[17],
         'cs_violation':state[18],'detected_defects':state[20]}
    assert int(R+rej_good+rej_bad+(admitted-completed)+state[20])==N, (method,state,N)
    if m in [0,1,3,7,9,10]:assert state[15]>-1e-6
    return out,trace
