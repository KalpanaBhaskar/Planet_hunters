import os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.environ['TEMP']); import ph_lib as L
warnings.filterwarnings("ignore")
ROOT=Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters"); C=ROOT/"v2"/"cache"; FC=ROOT/"v2"/"feature_cache"; FC.mkdir(exist_ok=True)
PERIODS=np.exp(np.linspace(np.log(0.5),np.log(27.0),300))
DURS=(0.05,0.08,0.12,0.18,0.25)
NB=200

def bls_feats(t,y,s,cx,cy,periods=PERIODS,nbins=NB,durs=DURS):
    y=np.asarray(y,dtype=np.float64); y=y-np.median(y); N=len(y); total=y.sum(); cand=[]
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
            j=int(np.argmax(snr)); cand.append((snr[j],P,dur,dep[j],nin[j],nt,j,w))
    cand.sort(reverse=True); out={}
    for k in range(min(3,len(cand))):
        sc,P,dur,dep,nin,nt,j,w=cand[k]
        out[f'b{k}_score']=sc; out[f'b{k}_period']=P; out[f'b{k}_dur']=dur; out[f'b{k}_depth']=dep; out[f'b{k}_nin']=nin; out[f'b{k}_ntrans']=nt
        out[f'b{k}_duty']=dur/P; out[f'b{k}_width']=w/nbins
    if not cand:
        return out
    sc,P,dur,dep,nin,nt,phasebin,w=cand[0]; ph=(t%P)/P
    center=(phasebin+0.5)/nbins; dphase=((ph-center+0.5)%1.0)-0.5; it=np.abs(dphase)<=dur/P/2
    out['b_in_frac']=float(it.mean()); out['b_depth_exact']=float(np.median(y[~it])-np.median(y[it])) if it.sum()>2 else 0.0
    cyc=np.floor((t-t.min())/P).astype(int); depths=[]
    for c in np.unique(cyc):
        m=it&(cyc==c); mo=(~it)&(cyc==c)
        if m.sum()>=1 and mo.sum()>=2: depths.append(np.median(y[mo])-np.median(y[m]))
    depths=np.array(depths); out['b_ndepths']=len(depths)
    out['b_depth_cv']=float(np.std(depths)/(np.mean(np.abs(depths))+1e-9)) if len(depths)>1 else 9.0
    out['b_depth_pos']=float(np.mean(depths>0.5*np.max(depths))) if len(depths) else 0.0
    odd=((np.floor((t-t.min())/P).astype(int)%2)==0)
    out['b_odd']=float(np.median(y[~it&odd])-np.median(y[it&odd])) if (it&odd).sum()>2 else 0.0
    out['b_even']=float(np.median(y[~it&~odd])-np.median(y[it&~odd])) if (it&~odd).sum()>2 else 0.0
    out['b_oddeven']=abs(out['b_odd']-out['b_even'])/(abs(out['b_odd'])+abs(out['b_even'])+1e-9)
    d2=((ph-(center+0.5))%1.0); d2=np.minimum(d2,1-d2); it2=d2<=dur/P/2
    out['b_secondary']=float(np.median(y[~it2])-np.median(y[it2])) if it2.sum()>2 else 0.0
    for name,c in [('cx',cx),('cy',cy)]:
        out[f'b_{name}_shift']=float(np.median(c[~it])-np.median(c[it])) if (it.sum()>2 and (~it).sum()>2) else 0.0
    out['b_cen_mag']=float(np.hypot(out['b_cx_shift'],out['b_cy_shift']))
    out['b_score_ratio21']=cand[1][0]/(cand[0][0]+1e-9) if len(cand)>1 else 0.0
    out['b_depth_ratio21']=cand[1][3]/(cand[0][3]+1e-9) if len(cand)>1 else 0.0
    return out

def fill_gaps(x,valid):
    full=np.full((x.shape[0],len(valid)),np.nan,np.float32); full[:,valid]=x; idx=np.arange(full.shape[1])
    for i in range(full.shape[0]):
        row=full[i]; g=~np.isnan(row); full[i]=np.interp(idx,idx[g],row[g]).astype(np.float32)
    return full

def build_norm(rel,hp,c,sigma,stars,t,q,valid):
    n=len(rel)
    S={}; S.update(L.basic_stats(rel,"low")); S.update(L.basic_stats(hp,"high")); S.update(L.half_stats(rel,"low")); S.update(L.half_stats(hp,"high"))
    a,ap=L.acf_feats(rel,"low"); S.update(a); b,bp=L.acf_feats(hp,"high"); S.update(b)
    rel_full=fill_gaps(rel,valid); hp_full=fill_gaps(hp,valid)
    f,tops=L.fft_feats(rel_full,hp_full); S.update(f)
    P_low,Pow_low=tops['low']; P_high,Pow_high=tops['high']
    grid=np.exp(np.linspace(np.log(0.3),np.log(15.0),90))
    radius=stars.radius.values.astype(float); contam=stars.contamination.values.astype(float)
    rec=[]
    for i in range(n):
        o=L.per_star(rel[i],hp[i],c[i],sigma[i],t,q,P_low[i],P_high[i],ap[i],grid,radius[i],contam[i])
        o.update(bls_feats(t,hp[i],sigma[i],c[i,0],c[i,1]))
        rec.append(o)
    X=pd.concat([pd.DataFrame(S),pd.DataFrame.from_records(rec),L.metadata_frame(stars)],axis=1)
    X=X.replace([np.inf,-np.inf],np.nan)
    # cap pathological ratios
    for col in ["fold_shift_over_depth","t_shift_ratio","neg_shift_over_depth"]:
        if col in X: X[col]=X[col].clip(-50,50)
    return X

def chunk(args):
    split,lo,hi=args
    rel=np.load(C/f"rel_{split}.npy",mmap_mode="r"); hp=np.load(C/f"hp_{split}.npy",mmap_mode="r"); c=np.load(C/f"cen_{split}.npy",mmap_mode="r")
    sig=np.load(C/f"sigma_{split}.npy"); t=np.load(C/f"t_{split}.npy"); q=np.load(C/f"q_{split}.npy")
    stars=pd.read_csv(ROOT/"dataset_and_problem_statement"/f"{split}_stars.csv").iloc[lo:hi].reset_index(drop=True)
    valid=~np.isnan(np.load(ROOT/"dataset_and_problem_statement"/f"{split}_flux.npy",mmap_mode="r")[:200]).all(0)
    X=build_norm(np.asarray(rel[lo:hi]),np.asarray(hp[lo:hi]),np.asarray(c[lo:hi]),sig[lo:hi],stars,t,q,valid)
    return X

if __name__=="__main__":
    split=sys.argv[1] if len(sys.argv)>1 else "train"; limit=int(sys.argv[2]) if len(sys.argv)>2 else None; nj=int(sys.argv[3]) if len(sys.argv)>3 else 6
    n=18000 if split=="train" else 7000; n=min(n,limit) if limit else n
    spans=[(i,min(i+250,n)) for i in range(0,n,250)]
    todo=[]
    for lo,hi in spans:
        p=FC/f"{split}_{lo:05d}_{hi:05d}.pkl"
        if not p.exists(): todo.append((split,lo,hi))
    print(split,"chunks",len(spans),"todo",len(todo),flush=True)
    if todo:
        res=Parallel(n_jobs=nj,verbose=5)(delayed(chunk)(a) for a in todo)
        for a,X in zip(todo,res): X.to_pickle(FC/f"{a[0]}_{a[1]:05d}_{a[2]:05d}.pkl")
    X=pd.concat([pd.read_pickle(FC/f"{split}_{lo:05d}_{hi:05d}.pkl") for lo,hi in spans],ignore_index=True)
    X.to_pickle(C/f"X_{split}_v2.pkl"); print(split,X.shape,flush=True)
