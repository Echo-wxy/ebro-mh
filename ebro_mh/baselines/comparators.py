from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from ..calibration import PlattCalibrator, youden_threshold
from ..methods.rule_ensemble import EBROMH

class FittedLockedModel:
    def __init__(self,base,calibrator,threshold,imputer,features,extra=None):
        self.base=base; self.calibrator=calibrator; self.threshold=threshold; self.imputer=imputer; self.features=features; self.extra=extra or {}
    def transform(self,X):
        X=X[self.features].copy(); return pd.DataFrame(self.imputer.transform(X),columns=self.features,index=X.index)
    def predict_proba(self,X):
        Xt=self.transform(X); raw=self.extra.get('raw_predict',lambda m,z:m.predict_proba(z)[:,1])(self.base,Xt)
        return self.calibrator.predict(raw)
    def predict(self,X): return (self.predict_proba(X)>=self.threshold).astype(int)


def _make_base(name,cfg,seed):
    if name=='logistic_regression':
        c=cfg['logistic_regression']; return make_pipeline(StandardScaler(),LogisticRegression(C=c['C'],penalty='l2',solver='lbfgs',max_iter=c['max_iter'],random_state=seed))
    if name=='random_forest':
        c=cfg['random_forest']; return RandomForestClassifier(n_estimators=c['n_estimators'],max_features=c['max_features'],min_samples_leaf=c['min_samples_leaf'],n_jobs=c['n_jobs'],random_state=seed)
    if name=='xgboost':
        c=cfg['xgboost']; return XGBClassifier(n_estimators=c['n_estimators'],max_depth=c['max_depth'],learning_rate=c['learning_rate'],subsample=c['subsample'],colsample_bytree=c['colsample_bytree'],reg_lambda=c['reg_lambda'],n_jobs=c['n_jobs'],eval_metric='logloss',random_state=seed)
    if name=='rulefit':
        from imodels import RuleFitClassifier
        c=cfg['rulefit']; return RuleFitClassifier(n_estimators=c['n_estimators'],tree_size=c['tree_size'],max_rules=c['max_rules'],memory_par=c['memory_par'],include_linear=c['include_linear'],cv=c['cv'],random_state=seed)
    if name=='bayesian_rule_lists':
        from imodels import BayesianRuleListClassifier
        c=cfg['bayesian_rule_lists']; return BayesianRuleListClassifier(listlengthprior=c['listlengthprior'],listwidthprior=c['listwidthprior'],maxcardinality=c['maxcardinality'],minsupport=c['minsupport'],n_chains=c['n_chains'],max_iter=c['max_iter'],random_state=seed,verbose=False)
    if name=='ebm':
        from interpret.glassbox import ExplainableBoostingClassifier
        c=cfg['ebm']; return ExplainableBoostingClassifier(max_bins=c['max_bins'],interactions=c['interactions'],outer_bags=c['outer_bags'],learning_rate=c['learning_rate'],max_rounds=c['max_rounds'],min_samples_leaf=c['min_samples_leaf'],n_jobs=c['n_jobs'],random_state=seed)
    raise KeyError(name)


def fit_locked(name,X_fit,y_fit,X_val,y_val,X_cal,y_cal,features,cfg,seed):
    from sklearn.impute import SimpleImputer
    imp=SimpleImputer(strategy='median').fit(X_fit[features])
    f=lambda X: pd.DataFrame(imp.transform(X[features]),columns=features,index=X.index)
    xf,xv,xc=f(X_fit),f(X_val),f(X_cal)
    if name=='ebro_mh':
        model=EBROMH(cfg['ebro_mh'],random_state=seed).fit(xf,y_fit,xv,y_val,xc,y_cal)
        # EBRO already performs Platt calibration and threshold selection internally
        class IdentityCal:
            def predict(self,z): return np.asarray(z)
        return FittedLockedModel(model,IdentityCal(),model.threshold_,imp,features,extra={'raw_predict':lambda m,z:m.predict_proba(z)[:,1]})
    base=_make_base(name,cfg,seed)
    # BayesianRuleListClassifier requires a *binary item matrix* (0/1), not
    # ordinal bin labels such as 0,1,2,3,4. Fit quantile bins on the fitting
    # partition only and expand each bin to a one-hot binary indicator.
    if name=='bayesian_rule_lists':
        bins=KBinsDiscretizer(
            n_bins=5, encode='onehot-dense', strategy='quantile',
            subsample=None, dtype=np.float64
        ).fit(xf)
        xfb=bins.transform(xf); xvb=bins.transform(xv); xcb=bins.transform(xc)
        brl_feature_names=list(bins.get_feature_names_out(features))
        # Defensive check: BRL's implementation explicitly requires every input
        # entry to be 0 or 1.
        if not np.all((xfb == 0) | (xfb == 1)):
            raise RuntimeError('BRL discretization did not produce a binary item matrix.')
        base.fit(xfb,np.asarray(y_fit,dtype=int),feature_names=brl_feature_names)
        raw_cal=base.predict_proba(xcb)[:,1]; cal=PlattCalibrator().fit(raw_cal,y_cal)
        pval=cal.predict(base.predict_proba(xvb)[:,1]); thr=youden_threshold(y_val,pval)
        return FittedLockedModel(
            base,cal,thr,imp,features,
            extra={
                'raw_predict':lambda m,z:m.predict_proba(bins.transform(z))[:,1],
                'bins':bins,
                'brl_feature_names':brl_feature_names
            }
        )
    base.fit(xf,np.asarray(y_fit,dtype=int))
    raw_cal=base.predict_proba(xc)[:,1]; cal=PlattCalibrator().fit(raw_cal,y_cal)
    pval=cal.predict(base.predict_proba(xv)[:,1]); thr=youden_threshold(y_val,pval)
    return FittedLockedModel(base,cal,thr,imp,features)
