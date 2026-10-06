"""Prepare leakage-controlled sensor and image quality scores."""
import os
os.environ['OMP_NUM_THREADS']='6';os.environ['VECLIB_MAXIMUM_THREADS']='6'
from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parent

def save_stream(name,frame,train,cal,test,scores,extra):
 edges=np.quantile(scores[train],[.6,.85])
 groups=np.digitize(scores,edges)
 frame['score']=scores;frame['group']=groups;frame['split']=''
 frame.loc[train,'split']='train';frame.loc[cal,'split']='calibration';frame.loc[test,'split']='test'
 (ROOT/'data'/'scored').mkdir(parents=True,exist_ok=True)
 frame.to_csv(ROOT/'data'/'scored'/f'{name}.csv',index=False,float_format='%.12g')
 info={'name':name,'n':len(frame),'defects':int(frame.label.sum()),'train':len(train),'calibration':len(cal),'test':len(test),
       'test_defects':int(frame.label.iloc[test].sum()),'test_auroc':roc_auc_score(frame.label.iloc[test],scores[test]),
       'test_ap':average_precision_score(frame.label.iloc[test],scores[test]),'bin_edges':edges.tolist(),
       'bins':[{'g':g,'calibration_n':int(sum(groups[cal]==g)),'calibration_k':int(frame.label.iloc[cal][groups[cal]==g].sum()),
               'test_n':int(sum(groups[test]==g)),'test_k':int(frame.label.iloc[test][groups[test]==g].sum())} for g in range(3)],**extra}
 (ROOT/'data'/'scored'/f'{name}_info.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2),flush=True)

def secom():
 d=ROOT/'data'/'secom';X=np.loadtxt(d/'secom.data');raw=pd.read_csv(d/'secom_labels.data',sep=r'\s+',header=None,names=['label','date','time'])
 raw['timestamp']=pd.to_datetime(raw['date']+' '+raw['time'],dayfirst=True)
 order=np.argsort(raw.timestamp.values,kind='stable');X=X[order];raw=raw.iloc[order].reset_index(drop=True)
 y=(raw.label.values==1).astype(int);n=len(y);train=np.arange(n//2);cal=np.arange(n//2,3*n//4);test=np.arange(3*n//4,n)
 usable=np.isfinite(X[train]).any(axis=0);X=X[:,usable]
 model=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=400,min_samples_leaf=10,max_features=.5,class_weight='balanced_subsample',random_state=20261006,n_jobs=8))
 model.fit(X[train],y[train]);scores=model.predict_proba(X)[:,1]
 save_stream('secom',pd.DataFrame({'id':order,'timestamp':raw.timestamp,'label':y}),train,cal,test,scores,{'features_used':int(usable.sum()),'split_rule':'chronological 50/25/25 percent, stable timestamp order'})

def ksdd2():
 import torch,torchvision
 from PIL import Image
 from torch.utils.data import Dataset,DataLoader
 d=ROOT/'data'/'ksdd2';files=sorted(p for p in d.rglob('*.png') if not p.stem.endswith('_GT'))
 # Dataset uses *_GT.png annotation masks.
 records=[]
 for p in files:
  mask=p.with_name(p.stem+'_GT.png')
  if not mask.exists():continue
  records.append((p,int(np.asarray(Image.open(mask)).max()>0), 'test' if 'test' in p.parts else 'train'))
 feature_file=ROOT/'data'/'ksdd2_features.npz'
 manifest=ROOT/'data'/'scored'/'ksdd2.csv'
 refresh='--refresh-features' in __import__('sys').argv
 if feature_file.exists() and manifest.exists() and not refresh:
  cached=pd.read_csv(manifest)
  records=[(d/row.id,int(row.label),'test' if 'test' in Path(row.id).parts else 'train') for row in cached.itertuples()]
 if not records:raise RuntimeError('Run download_data.py before raw image preparation')
 class Images(Dataset):
  def __len__(self):return len(records)
  def __getitem__(self,i):
   img=Image.open(records[i][0]).convert('RGB').resize((256,640),Image.Resampling.BILINEAR)
   x=torch.from_numpy(np.asarray(img).copy()).permute(2,0,1).float()/255
   return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
 torch.set_num_threads(6);torch.manual_seed(20261006)
 dev='mps' if torch.backends.mps.is_available() else 'cpu';print('Feature device',dev,'images',len(records),flush=True)
 feature_file=ROOT/'data'/'ksdd2_features.npz'
 if feature_file.exists() and not refresh:features=np.load(feature_file)['features']
 else:
  torch.hub.set_dir(str(ROOT/'data'/'weights'))
  model=torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
  encoder=torch.nn.Sequential(*list(model.children())[:-1]).to(dev).eval()
  feats=[];t=time.monotonic()
  # Batch extraction uses the available GPU; filesystem workers stay in the main process for portability.
  with torch.inference_mode():
   for j,x in enumerate(DataLoader(Images(),batch_size=16,num_workers=0,shuffle=False)):
    feats.append(encoder(x.to(dev)).flatten(1).cpu().numpy())
    if j%25==0:print('feature batch',j,'seconds',time.monotonic()-t,flush=True)
  features=np.vstack(feats);np.savez_compressed(feature_file,features=features)
 y=np.array([x[1] for x in records]);pool=np.array([i for i,x in enumerate(records) if x[2]=='train']);test=np.array([i for i,x in enumerate(records) if x[2]=='test'])
 train,cal=train_test_split(pool,test_size=.4,random_state=20261006,stratify=y[pool]);train=np.sort(train);cal=np.sort(cal)
 estimator=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=3000,solver='liblinear',random_state=20261006))
 cv=GridSearchCV(estimator,{'logisticregression__C':[.01,.1,1.,10.]},scoring='roc_auc',cv=StratifiedKFold(5,shuffle=True,random_state=20261006),n_jobs=6)
 cv.fit(features[train],y[train]);scores=cv.predict_proba(features)[:,1]
 save_stream('ksdd2',pd.DataFrame({'id':[str(r[0].relative_to(d)) for r in records],'label':y}),train,cal,test,scores,{'feature_dimension':features.shape[1],'feature_device':dev,'selected_C':cv.best_params_['logisticregression__C'],'split_rule':'official test; stratified 60/40 split of official training set'})
if __name__=='__main__':
 import sys
 if len(sys.argv)==1 or sys.argv[1]=='secom':secom()
 if len(sys.argv)==1 or sys.argv[1]=='ksdd2':ksdd2()
