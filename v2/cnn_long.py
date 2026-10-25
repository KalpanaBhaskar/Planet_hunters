import time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
torch.set_num_threads(6); torch.manual_seed(42); np.random.seed(42)
C=Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters\v2\cache")
X=np.load(C/"deep630_train.npy"); M=pd.read_pickle(C/"meta_train.pkl").values.astype(np.float32); y=np.load(C/"y.npy",allow_pickle=True); classes=np.load(C/"classes.npy",allow_pickle=True)
yi=np.searchsorted(classes,y); mu=M.mean(0); sd=M.std(0)+1e-6; M=(M-mu)/sd
tr,va=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
class Net(nn.Module):
    def __init__(self,nmeta):
        super().__init__()
        self.c=nn.Sequential(nn.Conv1d(4,32,9,padding=4),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,64,7,padding=3),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(128,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU())
        self.m=nn.Sequential(nn.Linear(nmeta,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,64),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(256+64,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,8))
    def forward(self,x,m):
        z=self.c(x); z=torch.cat([z.mean(-1),z.amax(-1)],1); return self.head(torch.cat([z,self.m(m)],1))
net=Net(M.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=3e-4)
EP=20; sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP)
cnt=np.bincount(yi,minlength=8); w=torch.tensor(len(yi)/(8*cnt),dtype=torch.float32); w=w/w.mean(); lossf=nn.CrossEntropyLoss(weight=w,label_smoothing=0.03)
Xtr=torch.tensor(X[tr]); Mtr=torch.tensor(M[tr]); ytr=torch.tensor(yi[tr]); Xva=torch.tensor(X[va]); Mva=torch.tensor(M[va]); yva=yi[va]
def aug(x):
    B=x.shape[0]; out=torch.empty_like(x)
    for i in range(B): out[i]=torch.roll(x[i],int(torch.randint(-40,41,(1,))),dims=-1)
    out=out*(1+0.12*torch.randn(B,1,1)); out=out+0.03*torch.randn_like(out)
    mask=(torch.rand(B,4,1)>0.10).float(); return out*mask
best=0
for ep in range(EP):
    t0=time.time(); net.train(); perm=torch.randperm(len(tr))
    for bi in range(0,len(tr),128):
        b=perm[bi:bi+128]; opt.zero_grad(); loss=lossf(net(aug(Xtr[b]),Mtr[b]),ytr[b]); loss.backward(); opt.step()
    sched.step(); net.eval()
    with torch.no_grad(): pv=torch.cat([net(Xva[i:i+256],Mva[i:i+256]) for i in range(0,len(va),256)]).argmax(1).numpy()
    sc=f1_score(classes[yva],classes[pv],average='macro'); best=max(best,sc)
    print("epoch",ep,"sec",round(time.time()-t0,1),"loss",round(float(loss),4),"valF1",round(sc,4),flush=True)
print("best",round(best,4))
