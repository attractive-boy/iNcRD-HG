"""Re-encode the original 216 SMILES; no additional entities or data source."""
import json
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,rdBase
from rdkit.Chem import AllChem,MACCSkeys
from model import HERE,RELEASE,sha256
from incrd_hg_m123.data import load_graph


def main():
    raw=HERE.parents[3]/"incrd-orig/2.construct_network"
    ids=pd.read_csv(raw/"Drug.csv")["CID"].astype(int).tolist()
    smiles=pd.read_csv(raw/"drug_smiles.csv").set_index("drug_cid").loc[ids,"smiles"].tolist()
    features=pd.read_csv(raw/"drug_features-chemBertA.csv",index_col=0)
    assert features.index.astype(int).tolist()==ids
    x=features.values.astype(np.float32)
    x=(x-x.min(1,keepdims=True))/(x.max(1,keepdims=True)-x.min(1,keepdims=True)+1e-8)
    graph=load_graph("MiDrug")
    np.testing.assert_array_equal(x,graph["drug"].x.numpy())
    mols=[Chem.MolFromSmiles(s) for s in smiles]
    assert all(m is not None for m in mols)
    fingerprints={
        "morgan2":[AllChem.GetMorganFingerprint(m,2) for m in mols],
        "morgan3":[AllChem.GetMorganFingerprint(m,3) for m in mols],
        "maccs":[MACCSkeys.GenMACCSKeys(m) for m in mols],
        "pharm2":[AllChem.GetMorganFingerprint(m,2,useFeatures=True) for m in mols],
    }
    kernels={k:np.array([DataStructs.BulkTanimotoSimilarity(fp,fps) for fp in fps],dtype=np.float32)
             for k,fps in fingerprints.items()}
    original=pd.read_csv(raw/"drugSimMat.csv",header=None).values
    np.testing.assert_allclose(kernels["morgan2"],original,atol=1e-7,rtol=1e-7)
    np.savez_compressed(HERE/"chemical_kernels.npz",**kernels,drug_cids=ids)
    report=dict(status="passed",source_files={str(raw/f):sha256(raw/f) for f in
        ("Drug.csv","drug_smiles.csv","drug_features-chemBertA.csv","drugSimMat.csv")},
        source_sha256=sha256(__file__),kernel_sha256=sha256(HERE/"chemical_kernels.npz"),
        rdkit_version=rdBase.rdkitVersion,n_drugs=len(ids),original_feature_order_exact=True,
        original_morgan2_similarity_max_error=float(abs(kernels["morgan2"]-original).max()),
        labels_used=False,external_data_used=False)
    (HERE/"chemical_provenance.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
