"""Freeze nested subsets of original internal training indices before fitting."""
import json
import numpy as np
from common import HERE,CLEAN,load_full_inner,sha256
from model import SIZES


def main():
    arrays={};records=[]
    for fold in range(3):
        data=load_full_inner(fold);rng=np.random.default_rng(20261007+fold)
        orders={label:rng.permutation(np.flatnonzero(data['train_y']==label)) for label in [0,1]}
        previous=set()
        for size,fraction in SIZES.items():
            local=np.sort(np.concatenate([order[:int(np.floor(len(order)*fraction))] for order in orders.values()]))
            indices=data['train_indices'][local]
            assert previous.issubset(set(indices));previous=set(indices)
            assert not np.intersect1d(indices,data['val_indices']).size
            arrays[f'{size}_local_f{fold}']=local;arrays[f'{size}_global_f{fold}']=indices
            records.append(dict(fold=fold,size=size,fraction=fraction,train_count=len(local),
                train_positive=int(data['train_y'][local].sum()),val_count=len(data['val_y']),
                full_train_count=len(data['train_y']),seed=20261007+fold))
        arrays[f'val_global_f{fold}']=data['val_indices']
    path=HERE/'subsets.npz'
    if path.exists():
        with np.load(path) as z:
            assert set(z.files)==set(arrays)
            for name,array in arrays.items():np.testing.assert_array_equal(z[name],array)
    else:np.savez_compressed(path,**arrays)
    manifest=dict(status='frozen',records=records,subsets_sha256=sha256(path),
                  original_data_sha256=sha256(CLEAN/'inner_data.npz'),prepare_sha256=sha256(__file__),
                  no_new_holdout_split=True,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    mp=HERE/'subset_manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:mp.write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
