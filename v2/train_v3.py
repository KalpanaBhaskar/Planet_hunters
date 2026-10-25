import os, sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
warnings.filterwarnings("ignore")
ROOT=Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters"); C=ROOT/"v2"/"cache"; V=ROOT/"v2"
Xtr=pd.read_pickle(C/"X_train_v3.pkl"); Xte=pd.read_pickle(C/"X_test_v3.pkl")
y=np.load(C/"y.npy",allow_pickle=True); classes=np.load(C/"classes.npy",allow_pickle=True)
print("shapes",Xtr.shape,Xte.shape,flush=True)
skf=StratifiedKFold(5,shuffle=True,random_state=42); oof=np.zeros((len(Xtr),8),np.float32); testp=np.zeros((len(Xte),8),np.float32)
params=dict(objective="multiclass",num_class=8,learning_rate=0.04,num_leaves=63,min_child_samples=20,subsample=0.85,colsample_bytree=0.75,reg_lambda=5,n_estimators=900,n_jobs=6,verbosity=-1,class_weight="balanced",random_state=42,force_col_wise=True)
t0=time.time()
for f,(tr,va) in enumerate(skf.split(Xtr,y),1):
    m=lgb.LGBMClassifier(**params); m.fit(Xtr.iloc[tr],y[tr])
    oof[va]=m.predict_proba(Xtr.iloc[va]); testp+=m.predict_proba(Xte)/5
    print("fold",f,"sec",round(time.time()-t0,1),flush=True)
np.save(C/"oof_v3.npy",oof); np.save(C/"test_proba_v3.npy",testp)
base=f1_score(y,classes[oof.argmax(1)],average="macro"); print("base OOF",round(base,4),flush=True)
# coordinate ascent class multipliers
w=np.ones(8); grid=[0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.25,1.4,1.6,1.8,2.0]
best=base
for it in range(3):
    changed=False
    for c in range(8):
        cur=w[c]; local=(best,cur)
        for v in grid:
            w[c]=v; sc=f1_score(y,classes[(oof*w).argmax(1)],average="macro")
            if sc>local[0]: local=(sc,v)
        w[c]=local[1]
        if local[0]>best+1e-7: best=local[0]; changed=True
    print("tune",it,round(best,4),w.tolist(),flush=True)
    if not changed: break
pred=(oof*w).argmax(1); print("tuned OOF",round(best,4)); print(classification_report(y,classes[pred],digits=3))
sub=pd.DataFrame({"id":pd.read_csv(ROOT/"dataset_and_problem_statement"/"test_stars.csv").id,"label":classes[(testp*w).argmax(1)]})
sample=pd.read_csv(ROOT/"dataset_and_problem_statement"/"sample_submission.csv")
assert list(sub.id)==list(sample.id) and len(sub)==7000 and sub.label.isin(classes).all()
sub.to_csv(V/"submission.csv",index=False)
json.dump({"oof_macro_f1":float(best),"class_weights":w.tolist(),"classes":classes.tolist(),"n_features":int(Xtr.shape[1])},open(V/"metrics_v3.json","w"),indent=2)
print("wrote",V/"submission.csv"); print(sub.label.value_counts(),flush=True)

