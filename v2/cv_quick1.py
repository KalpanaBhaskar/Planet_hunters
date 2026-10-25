import time, warnings
from pathlib import Path
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
ROOT=Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters"); C=ROOT/"v2"/"cache"
y=pd.read_csv(ROOT/"dataset_and_problem_statement"/"train_stars.csv").label.values[:3000]; classes=np.array(sorted(np.unique(y)))
X=pd.read_pickle(C/"X_train_v2.pkl"); print("loaded",X.shape,flush=True)
tr,va=next(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
t0=time.time(); m=lgb.LGBMClassifier(objective='multiclass',num_class=8,learning_rate=0.05,num_leaves=31,n_estimators=50,n_jobs=1,verbosity=-1,class_weight='balanced',random_state=42)
m.fit(X.iloc[tr],y[tr]); print("fit",round(time.time()-t0,1),flush=True)
p=m.predict_proba(X.iloc[va]); print("f1",f1_score(y[va],classes[p.argmax(1)],average='macro'),flush=True)
