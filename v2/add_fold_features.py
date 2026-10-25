import numpy as np, pandas as pd, time
from pathlib import Path
ROOT=Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters"); C=ROOT/"v2"/"cache"
cols=[f"fph_{j:02d}" for j in range(32)]+[f"fpl_{j:02d}" for j in range(16)]+["fph_argmin","fph_min","fph_max","fph_ptp","fph_std","fph_ing","fph_egr","fph_flat","fph_sec","fph_sym","fpl_argmin","fpl_min","fpl_ptp","fpl_std","fpl_sym","fpl_med"]
def add(split):
    X=pd.read_pickle(C/f"X_{split}_v2.pkl")
    hp=np.load(C/f"hp_{split}.npy",mmap_mode="r"); rel=np.load(C/f"rel_{split}.npy",mmap_mode="r")
    sig=np.maximum(np.load(C/f"sigma_{split}.npy"),1e-6); t=np.load(C/f"t_{split}.npy")
    P=np.clip(np.nan_to_num(X["b0_period"].values,nan=27.0),0.5,27.0)
    n=len(X); rows=np.zeros((n,len(cols)),np.float32); t0=time.time()
    for i in range(n):
        ph=(t%P[i])/P[i]
        # high-pass profile
        nb=32; b=np.clip((ph*nb).astype(np.int16),0,nb-1)
        cnt=np.bincount(b,minlength=nb).astype(np.float32); s=np.bincount(b,weights=np.asarray(hp[i],dtype=np.float64),minlength=nb)
        prof=(s/np.maximum(cnt,1)).astype(np.float32)/sig[i]; prof-=np.median(prof)
        k=int(np.argmin(prof)); prof=np.roll(prof,-k); rows[i,0:32]=prof
        w=max(1,int(round(X["b0_width"].values[i]*nb)))
        rows[i,48]=k; rows[i,49]=prof.min(); rows[i,50]=prof.max(); rows[i,51]=prof.max()-prof.min(); rows[i,52]=prof.std()
        rows[i,53]=prof[:w].mean(); rows[i,54]=prof[-w:].mean(); rows[i,55]=prof[nb//4:3*nb//4].mean()
        sec=prof[nb//2-max(1,w//2):nb//2+max(1,w//2)+1]; rows[i,56]=sec.mean() if len(sec) else 0; rows[i,57]=prof[:w].mean()-prof[-w:].mean()
        # low-pass profile
        nb=16; b=np.clip((ph*nb).astype(np.int16),0,nb-1)
        cnt=np.bincount(b,minlength=nb).astype(np.float32); s=np.bincount(b,weights=np.asarray(rel[i],dtype=np.float64),minlength=nb)
        prof=(s/np.maximum(cnt,1)).astype(np.float32)/sig[i]; prof-=np.median(prof)
        k=int(np.argmin(prof)); prof=np.roll(prof,-k); rows[i,32:48]=prof
        rows[i,58]=k; rows[i,59]=prof.min(); rows[i,60]=prof.max()-prof.min(); rows[i,61]=prof.std(); rows[i,62]=prof[:2].mean()-prof[-2:].mean(); rows[i,63]=np.median(prof)
        if i%5000==0: print(split,i,round(time.time()-t0,1),flush=True)
    out=pd.concat([X,pd.DataFrame(rows,columns=cols,index=X.index)],axis=1); out.to_pickle(C/f"X_{split}_v3.pkl"); print(split,out.shape,round(time.time()-t0,1),flush=True)
add("train"); add("test")
