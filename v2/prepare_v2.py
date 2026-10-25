import os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import median_filter
sys.path.insert(0, os.environ['TEMP']); import ph_lib as L
warnings.filterwarnings("ignore")
ROOT=Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters"); D=ROOT/"dataset_and_problem_statement"; C=ROOT/"v2"/"cache"

def process(split):
    t0=time.time(); flux=np.load(D/f"{split}_flux.npy",mmap_mode="r"); cen=np.load(D/f"{split}_centroid.npy",mmap_mode="r")
    stars=pd.read_csv(D/f"{split}_stars.csv"); cad=pd.read_csv(D/f"{split}_cadences.csv")
    valid=~np.isnan(np.asarray(flux[:200])).all(0); n=len(stars)
    print(split,"n",n,"valid",valid.sum(),flush=True)
    f=np.asarray(flux[:,valid],dtype=np.float32); f/=np.median(f,axis=1,keepdims=True); rel=f-1.0; del f
    key=stars.camera.astype(str)+"_"+stars.ccd.astype(str)
    for k,idx in key.groupby(key).groups.items():
        idx=np.asarray(idx); rel[idx]-=np.median(rel[idx],axis=0)
    print("rel common mode done",round(time.time()-t0,1),flush=True)
    hp=rel-median_filter(rel,size=(1,49),mode="nearest")
    c=np.asarray(cen[:,:,valid],dtype=np.float32); c-=np.median(c,axis=2,keepdims=True)
    for k,idx in key.groupby(key).groups.items():
        idx=np.asarray(idx); c[idx]-=np.median(c[idx],axis=0)[None,:,:]
    sigma=np.maximum(1.4826*np.median(np.abs(hp-np.median(hp,axis=1,keepdims=True)),axis=1),1e-6)
    # save full normalized arrays for feature extraction
    np.save(C/f"rel_{split}.npy",rel.astype(np.float32)); np.save(C/f"hp_{split}.npy",hp.astype(np.float32))
    np.save(C/f"cen_{split}.npy",c.astype(np.float32)); np.save(C/f"sigma_{split}.npy",sigma.astype(np.float32))
    np.save(C/f"t_{split}.npy",cad.time_btjd.values[valid].astype(np.float64)); np.save(C/f"q_{split}.npy",cad.quality.values[valid].astype(np.int64))
    # deep tensors: 4 channels x 315 (average 4 cadences)
    Lv=1260; nb=Lv//4
    def ds2(x): return x[:,:Lv].reshape(x.shape[0],nb,4).mean(2)
    def dsc(x): return x[:,:,:Lv].reshape(x.shape[0],x.shape[1],nb,4).mean(3)
    hp_ds=ds2(hp); rel_ds=ds2(rel); c_ds=dsc(c)
    cs=1.4826*np.median(np.abs(c_ds),axis=2)+1e-6
    deep=np.stack([np.clip(hp_ds/sigma[:,None],-20,20),np.clip(rel_ds/sigma[:,None],-20,20),
                   np.clip(c_ds[:,0]/cs[:,0:1],-20,20),np.clip(c_ds[:,1]/cs[:,1:2],-20,20)],axis=1).astype(np.float32)
    np.save(C/f"deep_{split}.npy",deep); del deep,hp_ds,rel_ds,c_ds
    meta=L.metadata_frame(stars); meta["log_sigma"]=np.log10(sigma+1e-9); meta=meta.astype(np.float32)
    meta.to_pickle(C/f"meta_{split}.pkl")
    if split=="train":
        y=stars.label.values; np.save(C/"y.npy",y); np.save(C/"classes.npy",np.array(sorted(np.unique(y))))
        print("class counts",pd.Series(y).value_counts().to_dict(),flush=True)
    print(split,"done sec",round(time.time()-t0,1),"deep",deep.shape if False else "",flush=True)

process("train"); process("test")
# align metadata columns
a=pd.read_pickle(C/"meta_train.pkl"); b=pd.read_pickle(C/"meta_test.pkl"); b=b.reindex(columns=a.columns,fill_value=0).astype(np.float32); b.to_pickle(C/"meta_test.pkl")
print("meta",a.shape,b.shape)
