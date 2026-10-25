import time, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
ROOT=Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters"); C=ROOT/"v2"/"cache"
y=pd.read_csv(ROOT/"dataset_and_problem_statement"/"train_stars.csv").label.values[:3000]; classes=np.array(sorted(np.unique(y)))
Xv2=pd.read_pickle(C/"X_train_v2.pkl"); Xv1=pd.read_pickle(ROOT/"feature_cache"/"X_train.pkl").iloc[:3000]
print(Xv1.shape,Xv2.shape)
def cv(X,name):
    oof=np.zeros((len(y),8),np.float32); t0=time.time()
    skf=StratifiedKFold(5,shuffle=True,random_state=42)
    for tr,va in skf.split(X,y):
        m=lgb.LGBMClassifier(objective='multiclass',num_class=8,learning_rate=0.04,num_leaves=63,min_child_samples=20,subsample=0.85,colsample_bytree=0.75,reg_lambda=5,n_estimators=2000,n_jobs=6,verbosity=-1,class_weight='balanced',random_state=42)
        m.fit(X.iloc[tr],y[tr]); oof[va]=m.predict_proba(X.iloc[va])
    print(name,'OOF',round(f1_score(y,classes[oof.argmax(1)],average='macro'),4),'sec',round(time.time()-t0,1)); return oof
cv(Xv1,'v1'); cv(Xv2,'v2')
