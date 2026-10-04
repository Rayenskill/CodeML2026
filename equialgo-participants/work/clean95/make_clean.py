"""Clean merit-score model. Labels: data/donnees_demandes.csv only. No platform scores used.
Score = z(cote) + a*z(hours); a = hours/cote ratio of the committee logit (bootstrap mean).
Income and region are excluded by policy (merit = academic performance + effort)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
H=pd.read_csv('data/donnees_demandes.csv'); C=pd.read_csv('data/candidats_evaluation.csv')
REM=['Bas-Saint-Laurent','Cote-Nord','Gaspesie-Iles-de-la-Madeleine']
def X(d): return np.c_[d.cote_r_equivalent,d.heures_travail_semaine,np.log(d.revenu_familial_estime),d.region_administrative.isin(REM)]
Xh,Xc=X(H),X(C); mu,sd=Xh[:,:3].mean(0),Xh[:,:3].std(0,ddof=1)
def Z(x): z=x.copy(); z[:,:3]=(x[:,:3]-mu)/sd; return z
Zh,Zc=Z(Xh),Z(Xc); y=H.decision_octroi.values
fit=lambda Z,y: LogisticRegression(C=1e6,max_iter=2000).fit(Z,y).coef_[0]
b=fit(Zh,y); a_pt=b[1]/b[0]
rng=np.random.default_rng(0); A=[]
for _ in range(300):
    i=rng.integers(0,len(y),len(y)); bb=fit(Zh[i],y[i]); A.append(bb[1]/bb[0])
a_boot=np.mean(A); print(f'a point={a_pt:.4f} bootstrap mean={a_boot:.4f} sd={np.std(A):.4f}')
def write(name,a,K):
    s=Zc[:,0]+a*Zc[:,1]; r=np.zeros(len(s),int); r[np.argsort(-s,kind='stable')[:K]]=1
    assert 0.36<=r.mean()<=0.44 and len(r)==4000
    pd.DataFrame({'id_candidat':C.id_candidat,'decision_octroi':r}).to_csv(f'upload_clean95/{name}.csv',index=False)
    best=pd.read_csv('upload_model_best/predictions.csv').set_index('id_candidat').decision_octroi.reindex(C.id_candidat).values
    print(name,'grants',r.sum(),'diff vs best',int((r!=best).sum()))
write('clean_a_boot_k1600',a_boot,1600)
write('clean_a_boot_k1590',a_boot,1590)
write('clean_a_boot_k1610',a_boot,1610)
