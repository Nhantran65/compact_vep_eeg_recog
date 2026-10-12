"""Reconstruct historical epoch folds using saved notebook file order.

For historical reproduction only. These folds do not hold subjects out.
Run from the project root. Existing folds are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('trainfold'))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite {args.output}')
    root = Path(__file__).resolve().parents[1]
    inputs = json.loads((root/'docs/legacy_input_order.json').read_text())
    frames=[]; origins=[]
    for item in inputs:
        df=pd.read_csv(root/item['path'])
        assert len(df)==item['epochs']
        frames.append(df)
        origins.extend([item['path']]*len(df))
    data=pd.concat(frames,ignore_index=True)
    assert data.shape==(3612,1795)
    x=data.iloc[:,:-3].to_numpy(); y=data['Class'].to_numpy()
    assert np.isfinite(x).all()
    meta=data.iloc[:,-3:].copy(); meta['source_file']=origins
    meta['row_id']=np.arange(len(data))
    args.output.mkdir(parents=True)
    info=[]
    for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y),1):
        folder=args.output/f'fold_{fold}'; folder.mkdir()
        hashes={}
        for label,indices in [('train',train),('test',test)]:
            df=pd.DataFrame(x[indices]);df['Class']=y[indices]
            path=folder/f'{label}.csv';df.to_csv(path,index=False)
            hashes[label]=hashlib.sha256(path.read_bytes()).hexdigest()
            meta.iloc[indices].to_csv(folder/f'{label}_metadata.csv',index=False)
        # Subject fields in source may use local IDs; physical source recording overlap is unambiguous.
        shared=set(np.array(origins)[train]) & set(np.array(origins)[test])
        info.append({'fold':fold,'train_samples':len(train),'test_samples':len(test),
                     'shared_source_recordings':len(shared),'sha256':hashes})
    (args.output/'fold_info.json').write_text(json.dumps({'protocol':'legacy_epoch_stratified_seed42',
       'input_order':'saved SVM notebook console output, 59 CSVs',
       'not_subject_disjoint':True,'folds':info},indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
