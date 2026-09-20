from __future__ import annotations
from dataclasses import dataclass
import itertools, math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mutual_info_score
from ..calibration import PlattCalibrator, youden_threshold

@dataclass(frozen=True)
class Rule:
    conditions: tuple
    rate: float
    score: float
    prior_log: float
    def fires(self,X):
        m=np.ones(len(X),dtype=bool)
        for col,op,val in self.conditions:
            z=np.asarray(X[col],dtype=float)
            m &= (z>=val) if op=='>=' else (z<=val)
        return m
    @property
    def key(self): return tuple(self.conditions)

class EBROMH:
    def __init__(self,cfg,random_state=42): self.cfg=cfg; self.random_state=random_state
    def _rule_stats(self,X,y,conds):
        m=np.ones(len(X),dtype=bool)
        for col,op,val in conds:
            z=np.asarray(X[col],dtype=float); m &= (z>=val) if op=='>=' else (z<=val)
        supp=m.mean(); prev=y.mean(); pos=(1+y[m].sum())/(2+m.sum()) if m.sum()>0 else 0.5
        mi=mutual_info_score(y,m.astype(int)); h=mutual_info_score(y,y)
        ev=0.0 if h<=0 else mi/h
        complexity=len(conds)
        s=(pos-prev)*math.sqrt(max(supp,1e-12))+ev-(0.02 if complexity==2 else 0)
        prior=-self.cfg['lambda_complexity']*complexity+self.cfg['lambda_evidence']*ev
        return supp,pos,s,prior
    def _generate(self,X,y):
        qs=self.cfg['quantiles']; one=[]
        for col in X.columns:
            vals=np.asarray(X[col],dtype=float); cuts=np.unique(np.nanquantile(vals,qs))
            for cut in cuts:
                for op in ('>=','<='):
                    cond=((col,op,float(cut)),)
                    supp,pos,s,prior=self._rule_stats(X,y,cond)
                    if supp<self.cfg['min_support']: continue
                    if supp>self.cfg['max_one_rule_support']: continue
                    if pos<=y.mean(): continue
                    one.append(Rule(cond,pos,s,prior))
        one=sorted(one,key=lambda r:r.score,reverse=True)
        pool=one[:self.cfg['initial_pair_pool']]
        pairs=[]
        for a,b in itertools.combinations(pool,2):
            if a.conditions[0][0]==b.conditions[0][0]: continue
            cond=tuple(sorted(a.conditions+b.conditions))
            supp,pos,s,prior=self._rule_stats(X,y,cond)
            if supp>=self.cfg['min_support'] and pos>y.mean(): pairs.append(Rule(cond,pos,s,prior))
        uniq={r.key:r for r in one+pairs}
        return sorted(uniq.values(),key=lambda r:r.score,reverse=True)
    def _rule_prob(self,rule,X,prev):
        m=rule.fires(X); return np.where(m,rule.rate,prev)
    def _ensemble_score(self,rules,w,X,prev):
        fire=np.column_stack([r.fires(X) for r in rules]).astype(float)
        mass=fire@w
        anyfire=fire.sum(axis=1)>0
        return np.where(anyfire,mass,prev)
    def fit(self,X_fit,y_fit,X_val,y_val,X_cal,y_cal):
        rng=np.random.default_rng(self.random_state); self.feature_names_=list(X_fit.columns); self.prev_=float(np.mean(y_fit))
        cand=self._generate(X_fit,np.asarray(y_fit,dtype=int))
        if len(cand)<self.cfg['particles']: raise RuntimeError(f'Only {len(cand)} candidate rules generated')
        lp=np.array([r.prior_log for r in cand],dtype=float); prob=np.exp(lp-lp.max()); prob/=prob.sum()
        idx=rng.choice(len(cand),size=self.cfg['particles'],replace=False,p=prob)
        rules=[cand[i] for i in idx]; w=np.array([prob[i] for i in idx],dtype=float); w/=w.sum()
        order=rng.permutation(len(X_val)); batch=self.cfg['validation_batch_size']; trace=[]; best=-np.inf; stale=0
        for t in range(self.cfg['max_iterations']):
            ids=order[(t*batch)%len(order):((t*batch)%len(order))+batch]
            if len(ids)<batch: ids=np.r_[ids,order[:batch-len(ids)]]
            xb=X_val.iloc[ids]; yb=np.asarray(y_val)[ids]
            losses=[]
            for r in rules:
                pr=np.clip(self._rule_prob(r,xb,self.prev_),1e-6,1-1e-6)
                losses.append(-np.mean(yb*np.log(pr)+(1-yb)*np.log(1-pr)))
            logw=np.log(np.clip(w,1e-300,None))-self.cfg['beta']*np.asarray(losses)
            logw-=logw.max(); w=np.exp(logw); w/=w.sum(); ess=1.0/np.sum(w*w)
            resampled=False; props=0; accepts=0
            if ess < self.cfg['ess_threshold_fraction']*self.cfg['particles']:
                ridx=rng.choice(len(rules),size=len(rules),replace=True,p=w); rules=[rules[i] for i in ridx]; w=np.ones(len(rules))/len(rules); resampled=True
                # conservative reconstruction: propose replacement from candidate pool
                for i,r in enumerate(rules):
                    props+=1; rp=cand[rng.integers(0,len(cand))]
                    old=r.prior_log - self.cfg['beta']*losses[ridx[i]]
                    pr=np.clip(self._rule_prob(rp,xb,self.prev_),1e-6,1-1e-6)
                    newloss=-np.mean(yb*np.log(pr)+(1-yb)*np.log(1-pr)); new=rp.prior_log-self.cfg['beta']*newloss
                    if np.log(rng.random()) < min(0.0,new-old): rules[i]=rp; accepts+=1
            score=self._ensemble_score(rules,w,X_val,self.prev_)
            auc=roc_auc_score(y_val,score)
            trace.append({'iteration':t+1,'auc':float(auc),'ess_before_resampling':float(ess),'resampled':int(resampled),'mh_proposals':props,'mh_acceptances':accepts})
            if auc>best+self.cfg['auc_epsilon']: best=auc; stale=0
            else: stale+=1
            if stale>=self.cfg['no_improve_rounds']: break
        self.rules_=rules; self.weights_=w; self.trace_=trace
        raw_cal=self._ensemble_score(rules,w,X_cal,self.prev_); self.calibrator_=PlattCalibrator().fit(raw_cal,y_cal)
        pval=self.calibrator_.predict(self._ensemble_score(rules,w,X_val,self.prev_)); self.threshold_=youden_threshold(y_val,pval)
        return self
    def predict_proba(self,X):
        raw=self._ensemble_score(self.rules_,self.weights_,X,self.prev_); p=self.calibrator_.predict(raw); return np.column_stack([1-p,p])
