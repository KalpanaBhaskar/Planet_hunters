import time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
C=Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters\v2\cache"); D=Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters\dataset_and_problem_statement")
hp=np.load(C/"hp_train.npy",mmap_mode="r"); sig=np.load(C/"sigma_train.npy"); t=np.load(C/"t_train.npy"); y=np.load(C/"y.npy",allow_pickle=True)
rng=np.random.default_rng(0); idx=np.r_[rng.choice(np.where(y=='planet')[0],250,False),rng.choice(np.where(y=='quiet_star')[0],250,False)]
lab=(y[idx]=='planet').astype(int)
def bls(t,y,s,periods,nbins=200,durs=(0.05,0.08,0.12,0.18,0.25),ntrans_factor=True,pmin=None):
    y=np.asarray(y,dtype=np.float64); y-=np.median(y); N=len(y); total=y.sum(); cand=[]
    for P in periods:
        if pmin is not None and P<pmin: continue
        ph=(t%P)/P; b=np.minimum((ph*nbins).astype(np.int32),nbins-1)
        cnt=np.bincount(b,minlength=nbins).astype(np.float64); sm=np.bincount(b,weights=y,minlength=nbins)
        ec=np.r_[cnt,cnt]; es=np.r_[sm,sm]; cc=np.cumsum(np.r_[0,ec]); cs=np.cumsum(np.r_[0,es]); nt=(t.max()-t.min())/P
        for dur in durs:
            w=max(1,int(round(dur/P*nbins)))
            if w>=nbins//2: continue
            nin=cc[w:w+nbins]-cc[:nbins]; sin=cs[w:w+nbins]-cs[:nbins]; nin=np.maximum(nin,1)
            muin=sin/nin; muout=(total-sin)/np.maximum(N-nin,1); dep=muout-muin
            snr=dep/(s*np.sqrt(1/nin+1/np.maximum(N-nin,1))+1e-12)
            sc=snr*(np.sqrt(max(nt,1.0)) if ntrans_factor else 1.0); j=int(np.argmax(sc))
            if sc[j]>0: cand.append((sc[j],P,dur,dep[j],snr[j],nin[j],nt,j))
    if not cand: return (0,0,0,0,0,0,0,0)
    cand.sort(reverse=True); return cand[0]
periods=np.exp(np.linspace(np.log(0.5),np.log(27),400))
variants=[('all_sqrtN',periods,True,None),('all_noN',periods,False,None),('gt1_sqrtN',periods,True,1.0),('gt1_noN',periods,False,1.0)]
for name,per,sf,pmin in variants:
    t0=time.time(); out=np.array([bls(t,hp[i],sig[i],per,ntrans_factor=sf,pmin=pmin) for i in idx]); print(name,'sec',round(time.time()-t0,1))
    print(' planet',np.mean(out[lab==1],0).round(3),'quiet',np.mean(out[lab==0],0).round(3))
    for j,k in enumerate(['score','P','dur','depth','snr','nin','ntrans','phase']): print('  ',k,round(roc_auc_score(lab,out[:,j]),3))
