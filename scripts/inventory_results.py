"""Build a source-linked inventory and independently check saved confusion metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
rows=[];checked=0

def collect(obj,source,path=''):
    global checked
    if isinstance(obj,dict):
        if 'accuracy' in obj and 'macro_f1' in obj:
            cm=obj.get('confusion_matrix');kappa=None
            if cm is not None:
                cm=np.asarray(cm,dtype=float);n=cm.sum();tp=np.diag(cm)
                recall=np.divide(tp,cm.sum(1),out=np.zeros_like(tp),where=cm.sum(1)!=0)
                f1=np.divide(2*tp,cm.sum(0)+cm.sum(1),out=np.zeros_like(tp),where=cm.sum(0)+cm.sum(1)!=0)
                pe=(cm.sum(0)*cm.sum(1)).sum()/n**2
                acc=tp.sum()/n;kappa=(acc-pe)/(1-pe)
                for key,value in [('accuracy',acc),('macro_f1',f1[(cm.sum(0)+cm.sum(1))>0].mean()),('balanced_accuracy',recall[cm.sum(1)>0].mean())]:
                    if key in obj and not np.isclose(obj[key],value,atol=1e-9):
                        raise ValueError(f'Metric mismatch: {source}:{path}:{key}')
                checked+=1
            rows.append({'source':str(source),'metric_path':path,'accuracy':obj['accuracy'],
                'macro_f1':obj['macro_f1'],'balanced_accuracy':obj.get('balanced_accuracy'),
                'kappa_recomputed':kappa,'scope':'smoke' if 'smoke' in str(source) else 'historical'})
        for key,value in obj.items():
            if key!='history':collect(value,source,f'{path}/{key}')
    elif isinstance(obj,list):
        for i,value in enumerate(obj):collect(value,source,f'{path}/{i}')

for p in sorted((root/'SNN/results').rglob('fold_*.json')):
    collect(json.loads(p.read_text()),p.relative_to(root))
for p in sorted((root/'results/shallowconvnet').glob('*.json')):
    collect(json.loads(p.read_text()),p.relative_to(root),'/test_metrics')
(root/'docs').mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(root/'docs/experiment_inventory.csv',index=False)
(root/'docs/inventory_validation.json').write_text(json.dumps({'metric_rows':len(rows),
    'confusion_matrices_recomputed':checked,'tolerance':1e-9},indent=2)+'\n')
print(f'{len(rows)} metric rows; {checked} confusion matrices independently verified')
table=pd.DataFrame(rows)
test=table[(table.scope!='smoke') & table.metric_path.str.endswith('/test_metrics')].copy()
test['experiment']=test.source.map(lambda x:str(Path(x).parent))
summary=test.groupby(['experiment','metric_path']).agg(
    folds=('accuracy','count'), accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),
    macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'),
    balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_std=('balanced_accuracy','std'),
    kappa_mean=('kappa_recomputed','mean'),kappa_std=('kappa_recomputed','std')).reset_index()
summary.to_csv(root/'docs/experiment_summary.csv',index=False)
