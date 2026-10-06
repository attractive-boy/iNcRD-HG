"""dataset-internal similarity views: drug ECFP4 (radius 2, 2048 bits) Tanimoto from the release SMILES,
miRNA Smith-Waterman local alignment of the mature sequences (match +1, mismatch -1, gap -2),
normalised by the shorter self-score.  Saved to cache/sims_{DS}.npz."""
import sys, numpy as np
sys.path.insert(0, '../mphgnn_plmf_20260921'); from plmf_splits import load_dataset
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def ecfp_tanimoto(smiles):
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            m = Chem.MolFromSmiles(s, sanitize=False); m.UpdatePropertyCache(strict=False); Chem.GetSymmSSSR(m)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    n = len(fps); S = np.zeros((n, n), np.float32)
    for i in range(n):
        S[i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    return S


def sw_matrix(seqs, match=1, mis=-1, gap=-2):
    L = max(len(s) for s in seqs); n = len(seqs)
    code = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    X = np.full((n, L), -1, np.int8); lens = np.array([len(s) for s in seqs])
    for i, s in enumerate(seqs):
        X[i, :len(s)] = [code.get(c, 4) for c in s]
    S = np.zeros((n, n), np.float32)
    for i in range(n):
        a = X[i, :lens[i]]; H = np.zeros((n, L + 1), np.float32); best = np.zeros(n, np.float32)
        for p in range(len(a)):
            Hn = np.zeros((n, L + 1), np.float32)
            sub = np.where(X == a[p], match, mis).astype(np.float32); sub[X < 0] = -1e4
            diag = H[:, :-1] + sub
            up = H[:, 1:] + gap
            cur = np.maximum(0, np.maximum(diag, up))
            # left gaps: sequential scan along the row
            for q in range(L):
                left = Hn[:, q] + gap
                Hn[:, q + 1] = np.maximum(cur[:, q], left)
            H = Hn; best = np.maximum(best, H.max(1))
        S[i] = best
    self_ = np.diag(S).copy()
    return S / np.minimum(self_[:, None], self_[None, :])


for ds in ['MDR', 'MDS']:
    mir, dru, _, _ = load_dataset(ds)
    Sd = ecfp_tanimoto(dru['SMILES'].astype(str).tolist())
    seqs = mir['miRNA_Sequence'].astype(str).str.upper().str.replace('T', 'U').tolist()
    Sr = sw_matrix(seqs)
    np.savez(f'cache/sims_{ds}.npz', drug_ecfp=Sd, mirna_sw=Sr)
    print(ds, Sd.shape, Sr.shape, float(Sd.mean()), float(Sr.mean()), flush=True)
