"""Familywise PAC risk calibration for the entire select-or-abstain policy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta


def validate(scores,costs,unsafe):
    scores,costs=np.asarray(scores,float),np.asarray(costs,float)
    unsafe=np.asarray(unsafe)
    if (scores.ndim!=2 or min(scores.shape)<1 or costs.shape!=scores.shape
        or unsafe.shape!=scores.shape or not np.isfinite(scores).all()
        or not np.isfinite(costs).all() or np.any((scores<0)|(scores>1))
        or not np.isin(unsafe,[0,1]).all()):raise ValueError('Invalid episode arrays')
    return scores,costs,unsafe.astype(bool)


def policy_losses(scores,costs,unsafe,threshold):
    scores,costs,unsafe=validate(scores,costs,unsafe)
    if np.isnan(threshold):raise ValueError('NaN threshold')
    eligible=scores>=threshold
    covered=eligible.any(axis=1)
    choice=np.argmin(np.where(eligible,costs,np.inf),axis=1)
    loss=unsafe[np.arange(len(scores)),choice]&covered
    selected_cost=np.where(covered,costs[np.arange(len(scores)),choice],0)
    return loss,covered,selected_cost


def upper_binomial(k,n,delta):
    if not (isinstance(k,(int,np.integer)) and isinstance(n,(int,np.integer))
            and 0<=k<=n and n>0 and 0<delta<1):raise ValueError('Invalid binomial inputs')
    return 1.0 if k==n else float(beta.ppf(1-delta,k+1,n-k))


def calibrate(scores,costs,unsafe,thresholds,alpha=.1,delta=.05):
    scores,costs,unsafe=validate(scores,costs,unsafe)
    thresholds=np.asarray(thresholds,float)
    if (thresholds.ndim!=1 or len(thresholds)==0 or not np.isfinite(thresholds).all()
        or np.any((thresholds<0)|(thresholds>1)) or not 0<alpha<1 or not 0<delta<1):
        raise ValueError('Frozen finite threshold family and valid risk levels required')
    records=[]
    for t in thresholds:
        loss,covered,cost=policy_losses(scores,costs,unsafe,t)
        bound=upper_binomial(int(loss.sum()),len(loss),delta/len(thresholds))
        records.append({'threshold':float(t),'violations':int(loss.sum()),
                        'upper_risk':bound,'coverage':float(covered.mean()),
                        'certified':bound<=alpha})
    valid=[r for r in records if r['certified']]
    selected=max(valid,key=lambda r:(r['coverage'],-r['threshold'])) if valid else None
    return {'alpha':alpha,'delta':delta,'episodes':len(scores),
            'selected':selected,'candidates':records,
            'fallback':'abstain with zero modeled violation loss',
            'guarantee':'P_calibration(R(selected policy)<=alpha)>=1-delta under IID episodes'}


def synthetic(seed,n,m):
    g=np.random.default_rng(seed)
    score=g.uniform(.5,1,(n,m)); latent=g.uniform(size=(n,1))
    # Conditional errors may correlate within an episode; episodes remain independent.
    unsafe=(.65*latent+.35*g.uniform(size=(n,m))) < (1-score)
    cost=g.uniform(1,10,(n,m)) - 1.5*unsafe
    return score,cost,unsafe


def benchmark():
    rows=[]
    for m in (10,100,1000):
        cal=synthetic(100+m,600,m); test=synthetic(5000+m,1500,m)
        fit=calibrate(*cal,np.linspace(.5,.995,20))
        chosen=fit['selected'];t=float('inf') if chosen is None else chosen['threshold']
        loss,coverage,_=policy_losses(*test,t)
        naive,naive_cov,_=policy_losses(*test,.9)
        rows.append({'candidate_count':m,'calibration':fit,
                     'selected_test_risk':float(loss.mean()),'test_coverage':float(coverage.mean()),
                     'naive_test_risk':float(naive.mean()),'naive_coverage':float(naive_cov.mean())})
    return {'scope':'unconditional episode violation risk; NOT risk conditional on acceptance',
            'separate_calibration_per_candidate_count':True,'results':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results.json')
    args=p.parse_args();Path(args.output).write_text(json.dumps(benchmark(),indent=2))
