import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
H=pd.read_csv('data/donnees_demandes.csv'); C=pd.read_csv('data/candidats_evaluation.csv')
best=pd.read_csv('upload_model_best/predictions.csv').set_index('id_candidat').decision_octroi
REM=['Bas-Saint-Laurent','Cote-Nord','Gaspesie-Iles-de-la-Madeleine']
def feats(d):
    return pd.DataFrame({'cote':d.cote_r_equivalent,'h':d.heures_travail_semaine,
      'li':np.log(d.revenu_familial_estime),'rem':d.region_administrative.isin(REM).astype(float)})
Fh,Fc=feats(H),feats(C)
mu,sd=Fh.mean(),Fh.std()
Zh=(Fh-mu)/sd; Zc=(Fc-mu)/sd
Zh['rem']=Fh.rem; Zc['rem']=Fc.rem
m=LogisticRegression(penalty=None,max_iter=1000).fit(Zh,H.decision_octroi)
b=pd.Series(m.coef_[0],index=Zh.columns);print(b.round(3).to_dict()); a_c=b.h/b.cote; print('committee a (hours/cote)=',a_c, 'income/cote',b.li/b.cote,'rem/cote',b.rem/b.cote)
# reference model from agent_dgp: s=z(cote)+a z(h)+d rem ; probit sigma .176 ; 1597 positives
def exp_err(pred,a=.198,d=-.049,sig=.176,npos=1597):
    s=(Zc.cote+a*Zc.h+d*Zc.rem).values
    lo,hi=-10,10
    for _ in range(60):
        q=(lo+hi)/2
        if norm.cdf((s-q)/sig).sum()>npos: lo=q
        else: hi=q
    p=norm.cdf((s-q)/sig)
    return (np.where(pred==1,1-p,p)).sum()
def top(score,K):
    r=np.zeros(len(score),int); r[np.argsort(-score,kind='stable')[:K]]=1; return r
bp=best.reindex(C.id_candidat).values
print('best file expected errs under ref', exp_err(bp))
for a in (0.15,0.184,0.198,0.21,0.25):
  for d in (0,-0.05,-0.1):
    for K in (1585,1590,1597,1600,1605):
        s=(Zc.cote+a*Zc.h+d*Zc.rem).values
        pr=top(s,K)
        print(f'a={a} d={d} K={K} diff_vs_best={int((pr!=bp).sum())} E_err={exp_err(pr):.1f}')
