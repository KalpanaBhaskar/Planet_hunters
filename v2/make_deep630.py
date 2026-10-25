import numpy as np
from pathlib import Path
C=Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters\v2\cache")
for split in ["train","test"]:
    rel=np.load(C/f"rel_{split}.npy",mmap_mode="r"); hp=np.load(C/f"hp_{split}.npy",mmap_mode="r"); c=np.load(C/f"cen_{split}.npy",mmap_mode="r"); sig=np.load(C/f"sigma_{split}.npy")
    Lv=1260; nb=Lv//2
    def ds2(x): return np.asarray(x[:,:Lv]).reshape(x.shape[0],nb,2).mean(2)
    def dsc(x): return np.asarray(x[:,:,:Lv]).reshape(x.shape[0],x.shape[1],nb,2).mean(3)
    hp_ds=ds2(hp); rel_ds=ds2(rel); c_ds=dsc(c); cs=1.4826*np.median(np.abs(c_ds),axis=2)+1e-6
    deep=np.stack([np.clip(hp_ds/sig[:,None],-20,20),np.clip(rel_ds/sig[:,None],-20,20),np.clip(c_ds[:,0]/cs[:,0:1],-20,20),np.clip(c_ds[:,1]/cs[:,1:2],-20,20)],axis=1).astype(np.float32)
    np.save(C/f"deep630_{split}.npy",deep); print(split,deep.shape,deep.nbytes/1e6)
