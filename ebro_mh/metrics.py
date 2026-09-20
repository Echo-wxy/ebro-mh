from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, log_loss
import statsmodels.api as sm


def ece_equal_count(y,p,n_bins=10):
    d=pd.DataFrame({'y':np.asarray(y,dtype=float),'p':np.asarray(p,dtype=float)}).dropna().sort_values('p').reset_index(drop=True)
    if len(d)==0:return np.nan
    q=min(n_bins,len(d))
    d['bin']=pd.qcut(np.arange(len(d)),q=q,labels=False,duplicates='drop')
    e=0.0
    for _,g in d.groupby('bin',observed=True):
        e += len(g)/len(d)*abs(g.y.mean()-g.p.mean())
    return float(e)


def calibration_intercept_slope(y,p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    z=np.log(p/(1-p)); X=sm.add_constant(z)
    try:
        fit=sm.GLM(y,X,family=sm.families.Binomial()).fit()
        return float(fit.params[0]),float(fit.params[1])
    except Exception:
        return np.nan,np.nan


def classification_metrics(y,p,pred,n_bins=10):
    ci,cs=calibration_intercept_slope(y,p)
    return {
        'auc':float(roc_auc_score(y,p)),
        'f1':float(f1_score(y,pred,zero_division=0)),
        'brier':float(brier_score_loss(y,p)),
        'ece':ece_equal_count(y,p,n_bins),
        'log_loss':float(log_loss(y,np.clip(p,1e-8,1-1e-8),labels=[0,1])),
        'calibration_intercept':ci,
        'calibration_slope':cs,
    }


def calibration_bins(y,p,n_bins=10):
    d=pd.DataFrame({'y':np.asarray(y,dtype=float),'p':np.asarray(p,dtype=float)}).sort_values('p').reset_index(drop=True)
    q=min(n_bins,len(d)); d['bin']=pd.qcut(np.arange(len(d)),q=q,labels=False,duplicates='drop')
    return d.groupby('bin',observed=True).agg(n=('y','size'),mean_pred=('p','mean'),observed=('y','mean')).reset_index()
