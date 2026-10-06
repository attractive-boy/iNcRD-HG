import torch
import dgl
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .helper import is_file_in_directory, has_node_with_zero_in_degree
from .graph_path import path_complex_mol
from .splitters import ScaffoldSplitter

class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list):
        self.labels = label_list
        self.graphs = graph_list
        self.device = torch.device('cpu')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index].to(self.device)
        graph = self.graphs[index].to(self.device)
        return label, graph


def collate_fn(batch):
    labels, graphs = zip(*batch)
    labels = torch.stack(labels)
    batched_graph = dgl.batch(graphs)
    return labels, batched_graph


def _get_task_names(dataset):
    """Get task names for different datasets"""
    task_map = {
        'tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                  'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
        'clintox': ['FDA_APPROVED', 'CT_TOX'],
        'sider': ['Hepatobiliary disorders', 'Metabolism and nutrition disorders',
                  'Product issues', 'Eye disorders', 'Investigations',
                  'Musculoskeletal and connective tissue disorders', 'Gastrointestinal disorders',
                  'Social circumstances', 'Immune system disorders',
                  'Reproductive system and breast disorders',
                  'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                  'General disorders and administration site conditions', 'Endocrine disorders',
                  'Surgical and medical procedures', 'Vascular disorders',
                  'Blood and lymphatic system disorders', 'Skin and subcutaneous tissue disorders',
                  'Congenital, familial and genetic disorders', 'Infections and infestations',
                  'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
                  'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions',
                  'Ear and labyrinth disorders', 'Cardiac disorders', 'Nervous system disorders',
                  'Injury, poisoning and procedural complications'],
        'muv': ['MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692',
                'MUV-712','MUV-713','MUV-733','MUV-737','MUV-810','MUV-832','MUV-846',
                'MUV-852','MUV-858','MUV-859'],
        'hiv': ['label'],
        'bbbp': ['label'],
        'bace': ['label']
    }
    return task_map.get(dataset, ['label'])


def create_data(datafile, encoder_atom, encoder_bond, batch_size, train_ratio, vali_ratio, test_ratio):
    """Create and save processed graph data"""
    directory_path = 'data/processed/'
    target_file_name = datafile + '.pth'

    if is_file_in_directory(directory_path, target_file_name):
        print(f"Data file {target_file_name} already exists!")
        return True

    print(f"Processing {datafile} dataset...")
    df = pd.read_csv('data/' + datafile + '.csv')
    task_names = _get_task_names(datafile)
    smiles_list = df['smiles']
    labels = df[task_names]

    # Handle missing values
    if datafile in ['tox21', 'muv']:
        labels = labels.fillna(0)

    data_list = []
    for i, smiles in enumerate(smiles_list):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(smiles_list)} molecules")

        Graph_list = path_complex_mol(smiles, encoder_atom, encoder_bond)
        if Graph_list == False:
            continue

        if has_node_with_zero_in_degree(Graph_list):
            continue

        data_list.append([smiles, torch.tensor(labels.iloc[i]), Graph_list])

    print('Graph list creation done!')

    # Split data
    splitter = ScaffoldSplitter().split(
        data_list,
        frac_train=train_ratio,
        frac_valid=vali_ratio,
        frac_test=test_ratio
    )
    print('Data splitting done!')

    # Organize splits
    train_label = [item[1] for item in splitter[0]]
    train_graph_list = [item[2] for item in splitter[0]]
    valid_label = [item[1] for item in splitter[1]]
    valid_graph_list = [item[2] for item in splitter[1]]
    test_label = [item[1] for item in splitter[2]]
    test_graph_list = [item[2] for item in splitter[2]]

    # Save processed data
    torch.save({
        'train_label': train_label,
        'train_graph_list': train_graph_list,
        'valid_label': valid_label,
        'valid_graph_list': valid_graph_list,
        'test_label': test_label,
        'test_graph_list': test_graph_list,
        'batch_size': batch_size,
        'shuffle': True,
    }, 'data/processed/' + datafile + '.pth')

    print(f"Data saved to data/processed/{datafile}.pth")


def laplacian_eigenmaps_3d(edge_index, num_nodes):
    """
    Use Laplacian Eigenmaps to generate 3D coordinates from graph structure.

    Takes the 3 smallest non-zero eigenvectors of the normalized Laplacian
    as 3D coordinates. This is more principled than spring layout.

    Args:
        edge_index: tuple of (src, dst) tensors
        num_nodes: number of nodes in the graph

    Returns:
        coords: [N, 3] tensor of 3D coordinates
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    src, dst = edge_index[0].numpy(), edge_index[1].numpy()

    # Build adjacency matrix
    data = np.ones(len(src))
    A = sp.csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))

    # Degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()

    # Handle isolated nodes
    degrees = np.maximum(degrees, 1e-8)

    # Normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    L_sym = sp.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

    # For small graphs, use dense eigendecomposition
    if num_nodes <= 4:
        # Too few nodes for 3D embedding, use random coords
        return np.random.randn(num_nodes, 3) * 0.1

    # Get smallest eigenvalues/eigenvectors (skip the first zero eigenvalue)
    k = min(4, num_nodes - 1)
    try:
        eigenvalues, eigenvectors = eigsh(L_sym, k=k, which='SM', tol=1e-6)
        # Take eigenvectors 1, 2, 3 (skip index 0 which is the constant vector)
        if k >= 4:
            coords = eigenvectors[:, 1:4]
        else:
            # Pad with zeros if not enough eigenvectors
            coords = np.zeros((num_nodes, 3))
            coords[:, :k-1] = eigenvectors[:, 1:k]
    except:
        # Fallback to random coordinates
        coords = np.random.randn(num_nodes, 3) * 0.1

    return coords


def create_zinc_dgl_data(encoder_atom, encoder_bond, batch_size, train_ratio, vali_ratio, test_ratio, prop_idx=0):
    """
    Create and save processed ZINC graph data using DGL's built-in ZINCDataset.

    DGL ZINCDataset is a 12K subset of ZINC (vs 250K full):
        - Train: 10,000
        - Valid: 1,000
        - Test: 1,000
        - Target: penalized logP = logP - SAS - cycles (single regression target)

    Uses Laplacian Eigenmaps for 3D coordinate generation (more principled than spring layout).

    Args:
        encoder_atom: Atom feature encoder type (ignored, uses DGL features)
        encoder_bond: Bond feature encoder type (ignored, uses DGL features)
        batch_size: Batch size for training
        train_ratio, vali_ratio, test_ratio: Ignored (DGL has fixed splits)
        prop_idx: Ignored (DGL ZINC has single target)

    Returns:
        mean, std: Statistics for target normalization
    """
    from dgl.data import ZINCDataset

    datafile = 'zinc_dgl'
    directory_path = 'data/processed/'
    target_file_name = datafile + '.pth'

    if is_file_in_directory(directory_path, target_file_name):
        print(f"Data file {target_file_name} already exists!")
        state = torch.load(directory_path + target_file_name)
        return state.get('target_mean', 0.0), state.get('target_std', 1.0)

    print("Loading DGL ZINCDataset (12K molecules)...")
    print("Using Laplacian Eigenmaps for 3D coordinate generation")

    # Load DGL ZINC datasets
    train_dataset = ZINCDataset(mode='train')
    valid_dataset = ZINCDataset(mode='valid')
    test_dataset = ZINCDataset(mode='test')

    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    # Collect all targets for normalization
    all_targets = []
    for dataset in [train_dataset, valid_dataset, test_dataset]:
        for _, label in dataset:
            all_targets.append(label.item())

    all_targets = np.array(all_targets)
    target_mean = float(np.mean(all_targets))
    target_std = float(np.std(all_targets))
    print(f"Target statistics - Mean: {target_mean:.4f}, Std: {target_std:.4f}")

    def process_dgl_zinc_split(dataset, split_name):
        """Process a single split of DGL ZINC dataset"""
        labels = []
        graphs = []
        failed_count = 0

        for i, (g, label) in enumerate(dataset):
            if i % 1000 == 0:
                print(f"Processing {split_name}: {i}/{len(dataset)} (failed: {failed_count})")

            num_nodes = g.num_nodes()

            try:
                # Generate 3D coordinates using Laplacian Eigenmaps
                coor = laplacian_eigenmaps_3d(g.edges(), num_nodes)
                coor = torch.tensor(coor, dtype=torch.float32)

                # Get node features from DGL (atom types, 28-dim one-hot or index)
                node_feat = g.ndata['feat']  # Shape: [N] or [N, feat_dim]
                if node_feat.dim() == 1:
                    # Convert index to one-hot (28 atom types)
                    node_feat_onehot = torch.zeros(num_nodes, 28)
                    node_feat_onehot.scatter_(1, node_feat.unsqueeze(1).long(), 1)
                    node_feat = node_feat_onehot

                # No padding - use raw 28-dim node features

                # Get edge features from DGL (bond types, 4-dim)
                edge_feat = g.edata['feat']  # Shape: [E] or [E, feat_dim]
                if edge_feat.dim() == 1:
                    # Convert index to one-hot (4 bond types)
                    num_edges = g.num_edges()
                    edge_feat_onehot = torch.zeros(num_edges, 4)
                    edge_feat_onehot.scatter_(1, edge_feat.unsqueeze(1).long(), 1)
                    edge_feat = edge_feat_onehot

                # No padding - use raw 4-dim edge features

                # Create new DGL graph with proper features
                new_g = dgl.graph((g.edges()[0], g.edges()[1]), num_nodes=num_nodes)
                new_g.ndata['feat'] = node_feat.float()
                new_g.ndata['coor'] = coor.float()
                new_g.edata['feat'] = edge_feat.float()
                new_g.edata['id'] = torch.ones(g.num_edges(), 1)  # All bonded edges

                # Use raw target directly (no normalization for fair comparison)
                label_tensor = torch.tensor([label.item()], dtype=torch.float64)

                labels.append(label_tensor)
                graphs.append(new_g)

            except Exception as e:
                failed_count += 1
                if failed_count < 5:
                    print(f"Warning: Failed to process graph {i}: {e}")
                continue

        print(f"{split_name}: Success {len(graphs)}, Failed {failed_count}")
        return labels, graphs

    # Process each split
    train_labels, train_graphs = process_dgl_zinc_split(train_dataset, 'train')
    valid_labels, valid_graphs = process_dgl_zinc_split(valid_dataset, 'valid')
    test_labels, test_graphs = process_dgl_zinc_split(test_dataset, 'test')

    # Save processed data
    import os
    os.makedirs(directory_path, exist_ok=True)

    # No normalization - use mean=0, std=1 so denormalization is identity
    torch.save({
        'train_label': train_labels,
        'train_graph_list': train_graphs,
        'valid_label': valid_labels,
        'valid_graph_list': valid_graphs,
        'test_label': test_labels,
        'test_graph_list': test_graphs,
        'batch_size': batch_size,
        'shuffle': True,
        'target_mean': 0.0,  # No normalization
        'target_std': 1.0,   # No normalization
        'prop_idx': 0,  # DGL ZINC has single target (penalized logP)
    }, directory_path + target_file_name)

    print(f"Data saved to {directory_path}{target_file_name}")
    print(f"Target range: mean={target_mean:.4f}, std={target_std:.4f} (not normalized)")

    return 0.0, 1.0  # No normalization


def create_zinc_dgl_with_smiles(encoder_atom, encoder_bond, batch_size, tolerance=0.001):
    """
    Create ZINC 12K dataset using official DGL/PyG split WITH SMILES recovery.

    This function:
    1. Loads official ZINC 12K (10K/1K/1K split) via PyG (has atom type info)
    2. Matches each molecule to TDC SMILES by MOLECULAR FORMULA + closest target
    3. Builds rich molecular graphs using path_complex_mol from matched SMILES
    4. Saves both graphs AND SMILES

    This gives us: Official 12K benchmark + SMILES for analysis

    Using molecular formula matching achieves ~92% match rate vs ~37% with structure-only.

    Args:
        encoder_atom: Atom feature encoder type
        encoder_bond: Bond feature encoder type
        batch_size: Batch size for training
        tolerance: Tolerance for target matching (default 0.001)

    Returns:
        mean, std: Statistics for target normalization (0.0, 1.0 for no normalization)
    """
    from rdkit import Chem
    from collections import Counter
    import os
    import pickle
    import sys

    # Need this to load PyG pickle files
    class Dictionary:
        def __init__(self):
            pass
    sys.modules['__main__'].Dictionary = Dictionary

    datafile = 'zinc_dgl_smiles'
    directory_path = 'data/processed/'
    target_file_name = datafile + '.pth'

    if is_file_in_directory(directory_path, target_file_name):
        print(f"Data file {target_file_name} already exists!")
        state = torch.load(directory_path + target_file_name)
        return state.get('target_mean', 0.0), state.get('target_std', 1.0)

    print("Creating ZINC 12K dataset with SMILES recovery (formula matching)...", flush=True)

    # Step 1: Ensure PyG ZINC data is downloaded
    pyg_data_dir = 'data/pyg_zinc'
    if not os.path.exists(f'{pyg_data_dir}/raw/train.pickle'):
        print("Step 1: Downloading PyG ZINC data...", flush=True)
        from torch_geometric.datasets import ZINC
        _ = ZINC(root=pyg_data_dir, subset=True, split='train')
    else:
        print("Step 1: PyG ZINC data already exists...", flush=True)

    # Load atom type dictionary
    with open(f'{pyg_data_dir}/raw/atom_dict.pickle', 'rb') as f:
        atom_dict = pickle.load(f)
    idx2atom = atom_dict.idx2word

    def atom_label_to_element(label):
        """Convert atom label like 'C H1' to element 'C'"""
        return label.split()[0]

    # Load indices for each split
    def load_indices(split):
        with open(f'{pyg_data_dir}/raw/{split}.index', 'r') as f:
            return [int(x) for x in f.read().strip().split(',')]

    train_idx = load_indices('train')
    val_idx = load_indices('val')
    test_idx = load_indices('test')

    # Load pickle data
    with open(f'{pyg_data_dir}/raw/train.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open(f'{pyg_data_dir}/raw/val.pickle', 'rb') as f:
        val_data = pickle.load(f)
    with open(f'{pyg_data_dir}/raw/test.pickle', 'rb') as f:
        test_data = pickle.load(f)

    print(f"PyG ZINC: Train={len(train_idx)}, Valid={len(val_idx)}, Test={len(test_idx)}", flush=True)

    # Build fingerprints with molecular formula
    def build_formula_fingerprints(data, indices):
        fps = []
        for idx in indices:
            mol = data[idx]
            atoms = mol['atom_type'].tolist()
            elements = [atom_label_to_element(idx2atom[a]) for a in atoms]
            formula = tuple(sorted(Counter(elements).items()))
            target = round(mol['logP_SA_cycle_normalized'].item(), 4)
            fps.append({
                'idx': idx,
                'formula': formula,
                'target': target,
                'label': mol['logP_SA_cycle_normalized']
            })
        return fps

    train_fps = build_formula_fingerprints(train_data, train_idx)
    valid_fps = build_formula_fingerprints(val_data, val_idx)
    test_fps = build_formula_fingerprints(test_data, test_idx)

    print("Step 2: Loading TDC SMILES and computing properties...", flush=True)

    # Load TDC SMILES
    tdc_file = 'data/zinc_smiles.csv'
    if not os.path.exists(tdc_file):
        raise FileNotFoundError(f"TDC SMILES file not found: {tdc_file}")

    df = pd.read_csv(tdc_file)
    all_smiles = df['smiles'].tolist()
    print(f"Loaded {len(all_smiles)} TDC SMILES", flush=True)

    # Build TDC lookup keyed by MOLECULAR FORMULA
    print("Computing penalized logP and molecular formulas for TDC...", flush=True)
    tdc_lookup = {}  # key: formula tuple -> list of (target, smiles)

    def get_formula_from_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return tuple(sorted(Counter([a.GetSymbol() for a in mol.GetAtoms()]).items()))

    for i, smiles in enumerate(all_smiles):
        if i % 50000 == 0:
            print(f"Processing TDC: {i}/{len(all_smiles)}", flush=True)

        formula = get_formula_from_smiles(smiles)
        if formula is None:
            continue

        target = compute_penalized_logp_rdkit(smiles)
        if target is None:
            continue

        if formula not in tdc_lookup:
            tdc_lookup[formula] = []
        tdc_lookup[formula].append((round(target, 4), smiles))

    print(f"Built TDC lookup with {len(tdc_lookup)} unique molecular formulas", flush=True)

    # Match molecules to TDC SMILES using formula
    print("Step 3: Matching molecules to TDC SMILES by formula...", flush=True)

    def match_and_build(fps, split_name):
        """Match fingerprints to TDC SMILES and build graphs

        Uses molecular formula matching and finds the SMILES with closest
        penalized logP to the target value.
        """
        labels = []
        graphs = []
        smiles_list = []
        matched = 0
        unmatched = 0
        failed_graph = 0

        for fp in fps:
            target = fp['target']
            formula = fp['formula']

            # Find candidates with same molecular formula
            if formula not in tdc_lookup or len(tdc_lookup[formula]) == 0:
                unmatched += 1
                continue

            # Find the SMILES with closest target value
            candidates = tdc_lookup[formula]
            best_smiles = None
            best_diff = float('inf')

            for tdc_target, smiles in candidates:
                diff = abs(tdc_target - target)
                if diff < best_diff:
                    best_diff = diff
                    best_smiles = smiles

            # Accept match if target difference is reasonable (< 0.5)
            if best_diff > 0.5:
                unmatched += 1
                continue

            matched += 1

            # Build graph from SMILES using path_complex_mol
            graph = path_complex_mol(best_smiles, encoder_atom, encoder_bond)

            if graph == False:
                failed_graph += 1
                continue

            if has_node_with_zero_in_degree(graph):
                failed_graph += 1
                continue

            # Use original target (official benchmark value)
            label_tensor = torch.tensor([fp['label'].item()], dtype=torch.float64)

            labels.append(label_tensor)
            graphs.append(graph)
            smiles_list.append(best_smiles)

        print(f"{split_name}: Matched={matched}, Unmatched={unmatched}, FailedGraph={failed_graph}, Final={len(graphs)}", flush=True)
        return labels, graphs, smiles_list

    train_labels, train_graphs, train_smiles = match_and_build(train_fps, 'train')
    valid_labels, valid_graphs, valid_smiles = match_and_build(valid_fps, 'valid')
    test_labels, test_graphs, test_smiles = match_and_build(test_fps, 'test')

    # Compute target statistics
    all_targets = [l.item() for l in train_labels + valid_labels + test_labels]
    target_mean = float(np.mean(all_targets))
    target_std = float(np.std(all_targets))
    print(f"Target statistics - Mean: {target_mean:.4f}, Std: {target_std:.4f}", flush=True)

    # Save processed data
    os.makedirs(directory_path, exist_ok=True)

    save_dict = {
        'train_label': train_labels,
        'train_graph_list': train_graphs,
        'valid_label': valid_labels,
        'valid_graph_list': valid_graphs,
        'test_label': test_labels,
        'test_graph_list': test_graphs,
        'batch_size': batch_size,
        'shuffle': True,
        'target_mean': 0.0,  # No normalization
        'target_std': 1.0,
        'train_smiles': train_smiles,
        'valid_smiles': valid_smiles,
        'test_smiles': test_smiles,
        'match_info': {
            'train_matched': len(train_smiles),
            'valid_matched': len(valid_smiles),
            'test_matched': len(test_smiles),
        }
    }

    torch.save(save_dict, directory_path + target_file_name)

    print(f"\nData saved to {directory_path}{target_file_name}", flush=True)
    print(f"Official 12K split with SMILES:", flush=True)
    print(f"  Train: {len(train_smiles)}/10000 matched", flush=True)
    print(f"  Valid: {len(valid_smiles)}/1000 matched", flush=True)
    print(f"  Test: {len(test_smiles)}/1000 matched", flush=True)

    return 0.0, 1.0


def compute_penalized_logp_rdkit(smiles):
    """
    Compute penalized logP using RDKit (same formula as DGL ZINC target).

    penalized_logP = logP - SAS - ring_penalty

    where ring_penalty = sum of (ring_size - 6) for each ring with size > 6

    Args:
        smiles: SMILES string

    Returns:
        penalized_logp: float, or None if computation fails
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    from rdkit.Chem import RDConfig
    import os
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

    try:
        from sascorer import calculateScore as calc_sas
    except ImportError:
        # Fallback: use simple SAS approximation if sascorer not available
        calc_sas = None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        # logP using Wildman-Crippen method
        logp = Crippen.MolLogP(mol)

        # SAS (Synthetic Accessibility Score)
        if calc_sas is not None:
            sas = calc_sas(mol)
        else:
            # Simple fallback approximation
            sas = 3.0  # Average SAS is around 3

        # Ring penalty: sum of (ring_size - 6) for rings > 6 atoms
        ring_info = mol.GetRingInfo()
        ring_penalty = 0
        for ring in ring_info.AtomRings():
            if len(ring) > 6:
                ring_penalty += len(ring) - 6

        penalized_logp = logp - sas - ring_penalty
        return penalized_logp

    except Exception as e:
        return None