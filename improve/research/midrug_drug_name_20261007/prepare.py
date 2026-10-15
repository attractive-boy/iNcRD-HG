"""Original Drug_Name only, label-free shared character TF-IDF and an independent oracle."""
import json, re
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from common import HERE, CLEAN, sha256

ORIGINAL=HERE.parents[3]/'incrd-orig/2.construct_network'
def independent_ngrams(text):
    result=[]
    for token in text.lower().split():
        token=' '+token+' '
        for n in range(3,6):
            if len(token)<=n:
                result.append(token);break
            result += [token[i:i+n] for i in range(len(token)-n+1)]
    return Counter(result)

def main():
    assert not (HERE/'covariates.npz').exists()
    expected=json.loads((CLEAN/'groups_provenance.json').read_text())['original_source_hashes']
    files=['Drug.csv','drug_features-chemBertA.csv']
    for f in files: assert sha256(ORIGINAL/f)==expected[f]
    table=pd.read_csv(ORIGINAL/'Drug.csv');names=table.Drug_Name.str.strip().tolist()
    ids=table.CID.astype(int).tolist();assert len(set(ids))==216
    chem=pd.read_csv(ORIGINAL/'drug_features-chemBertA.csv',index_col=0)
    assert chem.index.astype(int).tolist()==ids
    x=chem.values.astype(np.float32)
    normalized=(x-x.min(1,keepdims=True))/(np.ptp(x,axis=1,keepdims=True)+1e-8)
    graph=torch.load(CLEAN/'graph.pt',map_location='cpu')
    np.testing.assert_array_equal(normalized,graph['drug'].x.numpy())
    vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,
                              norm='l2',sublinear_tf=True,lowercase=True)
    features=vectorizer.fit_transform(names).toarray().astype(np.float32)
    terms=vectorizer.get_feature_names_out().tolist()
    counters=[independent_ngrams(name) for name in names]
    df=Counter()
    for row in counters:df.update(row.keys())
    assert terms==sorted([term for term,count in df.items() if count>=2])
    oracle=np.zeros_like(features,dtype=np.float64)
    for j,term in enumerate(terms):
        idf=1+np.log((1+len(names))/(1+df[term]))
        for i,row in enumerate(counters):
            if row[term]:oracle[i,j]=(1+np.log(row[term]))*idf
    oracle/=np.maximum(np.linalg.norm(oracle,axis=1,keepdims=True),1e-12)
    np.testing.assert_allclose(features,oracle.astype(np.float32),atol=5e-8,rtol=1e-6)
    rng=np.random.RandomState(20261007);permutation=rng.permutation(len(names))
    while np.any(permutation==np.arange(len(names))):permutation=rng.permutation(len(names))
    np.savez_compressed(HERE/'covariates.npz',features=features,permutation=permutation)
    similarity=features@features.T;np.fill_diagonal(similarity,0)
    neighbors={name:[dict(name=names[j],cosine=float(similarity[i,j])) for j in np.argsort(-similarity[i])[:5]]
               for i,name in enumerate(names)}
    (HERE/'vocabulary.json').write_text(json.dumps(dict(terms=terms,document_frequencies=[df[t] for t in terms],
        idf=vectorizer.idf_.tolist(),names=names,cids=ids,neighbors=neighbors),indent=2))
    meta=dict(status='passed',shape=list(features.shape),nonzero_rows=int((features!=0).any(1).sum()),
        zero_rows=[names[i] for i in np.flatnonzero(~(features!=0).any(1))],
        independent_tfidf_max_error=float(abs(features-oracle).max()),
        original_drug_feature_alignment_exact=True,permutation_fixed_points=0,
        original_source_hashes={str(ORIGINAL/f):expected[f] for f in files},
        covariates_sha256=sha256(HERE/'covariates.npz'),vocabulary_sha256=sha256(HERE/'vocabulary.json'),
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        prepare_sha256=sha256(__file__),no_lncrna=True,no_labels_used=True,
        no_external_vocabulary_or_pretrained_text_model=True)
    (HERE/'input_manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
