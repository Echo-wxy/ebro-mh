from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

class PlattCalibrator:
    def __init__(self): self.model=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000)
    def fit(self,score,y):
        s=np.asarray(score,dtype=float).reshape(-1,1); self.model.fit(s,np.asarray(y,dtype=int)); return self
    def predict(self,score):
        s=np.asarray(score,dtype=float).reshape(-1,1); return self.model.predict_proba(s)[:,1]

def youden_threshold(y,p):
    fpr,tpr,thr=roc_curve(y,p); j=tpr-fpr
    finite=np.isfinite(thr)
    if finite.any():
        idx=np.where(finite)[0][np.argmax(j[finite])]
    else: idx=int(np.argmax(j))
    return float(thr[idx])
