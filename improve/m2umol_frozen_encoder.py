"""Extract only the fully pretrained 2D branch from pinned upstream M2UMol.

The release checkpoint uses OGB categorical atom/bond encoders. The public
downstream example instead has new linear input layers and a new fusion head;
those layers cannot be treated as pretrained. This adapter exposes `feats2d`
from the original pretraining forward and requires its complete state.
"""
import ast
import copy
import inspect
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import Dropout, Linear, Sequential
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
import torch_geometric.transforms as T
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


def upstream_definitions(source):
    """Compile exact upstream definitions without importing 3D/text trainers."""
    source = Path(source)
    path = source/'pretraining_M2UMol/layer.py'
    tree = ast.parse(path.read_text())
    gps = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPSConv1')
    parent = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='M2UMol')
    forward = next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    body = []
    for statement in forward.body:
        body.append(copy.deepcopy(statement))
        if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='feats2d' for t in statement.targets):
            break
    else:
        raise ValueError('Cannot locate original pretraining 2D output')
    body.append(ast.Return(value=ast.Name(id='feats2d',ctx=ast.Load())))
    sliced = ast.FunctionDef(name='_pretraining_2d_forward',
        args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='twograph')],vararg=None,
            kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=body,decorator_list=[])
    namespace = globals().copy()
    code = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(gps),sliced],type_ignores=[]))
    exec(compile(code,str(path),'exec'),namespace)
    return namespace['GPSConv1'],namespace['_pretraining_2d_forward']


class M2UMolFrozen2D(nn.Module):
    def __init__(self,source):
        super().__init__()
        gps,forward = upstream_definitions(source)
        self._upstream_forward = forward
        # These are exactly the modules referenced before feats2d in the
        # pinned pretraining M2UMol.forward; names also match the checkpoint.
        self.transform = T.AddRandomWalkPE(walk_length=20,attr_name='pe')
        self.encoder2D = nn.ModuleList()
        for _ in range(3):
            network = Sequential(Linear(128,128),nn.ReLU(),Linear(128,128))
            self.encoder2D.append(gps(128,GINEConv(network),heads=4,attn_type='multihead',
                attn_kwargs={'dropout':0.5},dropout=0.1))
        self.pe_lin = Linear(20,8)
        self.pe_norm = nn.BatchNorm1d(20)
        self.linx = AtomEncoder(emb_dim=120)
        self.linedgea = BondEncoder(emb_dim=128)

    def load_pretrained(self,state):
        expected = self.state_dict()
        missing = sorted(set(expected)-set(state))
        if missing:
            raise ValueError('Missing pretrained 2D state: '+repr(missing))
        selected = {key:state[key] for key in expected}
        self.load_state_dict(selected,strict=True)
        self.requires_grad_(False)
        self.eval()
        return dict(loaded_state_entries=len(selected),ignored_other_modalities=len(state)-len(selected),missing=[])

    def forward(self,twograph):
        return self._upstream_forward(self,twograph)


def graph_builder(source):
    path = Path(source)/'pretraining_M2UMol/molepre.py'
    tree = ast.parse(path.read_text())
    names = ['edge_features','get_mol_edge_list_and_feat_mtx']
    definitions = [copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(definitions)!=2:
        raise ValueError('Pretraining graph functions changed')
    namespace = globals().copy()
    code = ast.fix_missing_locations(ast.Module(body=definitions,type_ignores=[]))
    exec(compile(code,str(path),'exec'),namespace)
    build = namespace['get_mol_edge_list_and_feat_mtx']
    def create(mol,smiles):
        edge_index,x,edge_attr = build(mol,smiles,[])
        if x.ndim!=2 or x.shape[1]!=9 or edge_index.ndim!=2 or edge_index.shape[0]!=2 or edge_attr.ndim!=2 or edge_attr.shape[1]!=3:
            raise ValueError('Unsupported original pretraining graph shape (including edgeless molecules)')
        return Data(x=x,edge_index=edge_index,edge_attr=edge_attr)
    return create
