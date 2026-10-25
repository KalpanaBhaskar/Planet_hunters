import time, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
C=Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters\v2\cache")
hp=np.load(C/"hp_train.npy",mmap_mode="r"); cen=np.load(C/"cen_train.npy",mmap_mode="r"); sig=np.load(C/"sigma_train.npy"); t=np.load(C/"t_train.npy"); y=np.load(C/"y.npy",allow_pickle=True)
rng=np.random.default_rng(0); idx=np.r_[rng.choice(np.where(y=='planet')[0],250,False),rng.choice(np.where(y=='quiet_star')[0],250,False)]
lab=(y[idx]=='planet').astype(int)
periods=np.exp(np.linspace(np.log(0.5),np.log(27),300))
def bfeat(t,y,s,cx,cy,periods,nbins=200,durs=(0.05,0.08,0.12,0.18,0.25)):
    y=np.asarray(y,dtype=np.float64); y-=np.median(y); N=len(y); total=y.sum(); cand=[]
    for P in periods:
        ph=(t%P)/P; b=np.minimum((ph*nbins).astype(np.int32),nbins-1)
        cnt=np.bincount(b,minlength=nbins).astype(np.float64); sm=np.bincount(b,weights=y,minlength=nbins)
        ec=np.r_[cnt,cnt]; es=np.r_[sm,sm]; cc=np.cumsum(np.r_[0,ec]); cs=np.cumsum(np.r_[0,es]); nt=(t.max()-t.min())/P
        for dur in durs:
            w=max(1,int(round(dur/P*nbins)))
            if w>=nbins//2: continue
            nin=cc[w:w+nbins]-cc[:nbins]; sin=cs[w:w+nbins]-cs[:nbins]; nin2=np.maximum(nin,1)
            muin=sin/nin2; muout=(total-sin)/np.maximum(N-nin,1); dep=muout-muin
            snr=dep/(s*np.sqrt(1/nin2+1/np.maximum(N-nin,1))+1e-12)
            j=int(np.argmax(snr)); cand.append((snr[j],P,dur,dep[j],nin[j],nt,j))
    cand.sort(reverse=True); out={}
    for k in range(min(3,len(cand))):
        sc,P,dur,dep,nin,nt,j=cand[k]
        out[f'b{k}_score']=sc; out[f'b{k}_period']=P; out[f'b{k}_dur']=dur; out[f'b{k}_depth']=dep; out[f'b{k}_nin']=nin; out[f'b{k}_ntrans']=nt
    sc,P,dur,dep,nin,nt,phasebin=cand[0]; ph=(t%P)/P
    # exact in-transit mask centered at best phasebin
    center=(phasebin+0.5)/nbins; dphase=((ph-center+0.5)%1.0)-0.5; it=np.abs(dphase)<=dur/P/2
    out['b_in_frac']=it.mean(); out['b_depth_exact']=np.median(y[~it])-np.median(y[it]) if it.sum()>2 else 0
    # per-cycle depths
    cyc=np.floor((t-t.min())/P).astype(int); depths=[]
    for c in np.unique(cyc):
        m=it&(cyc==c); mo=(~it)&(cyc==c)
        if m.sum()>=1 and mo.sum()>=2: depths.append(np.median(y[mo])-np.median(y[m]))
    depths=np.array(depths); out['b_ndepths']=len(depths); out['b_depth_cv']=np.std(depths)/(np.mean(np.abs(depths))+1e-9) if len(depths)>1 else 9
    out['b_depth_pos']=np.mean(depths>0.5*np.max(depths)) if len(depths) else 0
    # odd-even (using phase parity)
    odd=((np.floor((t-t.min())/P).astype(int)%2)==0)
    out['b_odd']=(np.median(y[~it&odd])-np.median(y[it&odd])) if (it&odd).sum()>2 else 0
    out['b_even']=(np.median(y[~it&~odd])-np.median(y[it&~odd])) if (it&~odd).sum()>2 else 0
    out['b_oddeven']=abs(out['b_odd']-out['b_even'])/(abs(out['b_odd'])+abs(out['b_even'])+1e-9)
    # secondary at phase 0.5
    d2=((ph-(center+0.5))%1.0)-0.0; d2=np.minimum(d2,1-d2); it2=d2<=dur/P/2
    out['b_secondary']=(np.median(y[~it2])-np.median(y[it2])) if it2.sum()>2 else 0
    # centroid in-transit shift
    for name,c in [('cx',cx),('cy',cy)]:
        out[f'b_{name}_shift']=(np.median(c[~it])-np.median(c[it])) if (it.sum()>2 and (~it).sum()>2) else 0
    return out
t0=time.time(); rows=[bfeat(t,hp[i],sig[i],cen[i,0],cen[i,1],periods) for i in idx]; print('sec',time.time()-t0)
keys=list(rows[0]); A=np.array([[r[k] for k in keys] for r in rows])
for k in keys:
    j=keys.index(k); print(k, round(roc_auc_score(lab,A[:,j]),3), 'planet',round(np.mean(A[lab==1,j]),4),'quiet',round(np.mean(A[lab==0,j]),4))
