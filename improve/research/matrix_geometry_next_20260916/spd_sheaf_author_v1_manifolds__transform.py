# SPDIsometryQR performs isometry X->QXQ^T    iso_params is a learnable isometry matrix

import torch
from torch import nn
from torch.nn import Parameter
from .spd import SPDManifold
from abc import ABC


class SPDTransform(nn.Module, ABC):
    def __init__(self, input_dims):
        super().__init__()

        self.dims = input_dims
        self.n_isom = input_dims * (input_dims - 1) // 2

        self.isom_params = Parameter(torch.Tensor(self.dims, self.dims))

        self.isom_params.data.copy_(torch.eye(self.dims))

    def forward(self, node_feats):
        raise NotImplementedError("forward is not implemented in SPDTransform")

    @classmethod
    def get(cls, type_str: str, input_dims: int):
        classes = {"qr": SPDIsometryQR, "cayley": SPDIsometryCayley, "residual": SPDTransformTangentResidual}
        return classes[type_str](input_dims)


### Learnable isometry
class SPDIsometryQR(SPDTransform):
    def __init__(self, input_dims):
        SPDTransform.__init__(self, input_dims)

    def forward(self, mat_feats, M=None):
        """Apply learnable isometry; optional M allows composition with an external frame."""
        W = self.isom_params

        q, r = torch.linalg.qr(W)
        d = torch.diag(r, 0)
        ph = d.sign()
        q = q * ph

        # We need q to be (1, d, d) for broadcasting if mat_feats is (B, d, d)
        Q = q.unsqueeze(0)

        if M is not None:
            # If M provided (B, d, d) or (d, d), fold it into the isometry.
            Q = Q @ M.transpose(-2, -1)

        mat_feats = Q @ mat_feats @ Q.transpose(-1, -2)

        return mat_feats


class SPDIsometryCayley(SPDTransform):
    def __init__(self, input_dims):
        SPDTransform.__init__(self, input_dims)

    def forward(self, mat_feats, M=None):
        W = self.isom_params

        W = W - W.t()  # skew-symmetric A^T = -A
        Id = torch.eye(W.size(0), device=W.device)

        # See https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv
        Q = torch.linalg.solve(
            Id - W, Id + W
        ).t()  # faster than (Id - W) @ torch.linalg.inv(Id + W)
        Q = Q.unsqueeze(0)

        if M is not None:
            Q = Q @ M.transpose(-2, -1)

        mat_feats = Q @ mat_feats @ Q.transpose(-1, -2)
        return mat_feats


class SPDTransformTangentResidual(SPDTransform):
    """
    SPD Transform with Tangent Space Residual Connection.

    Pipeline:
    1. (Optional) Base Isometry: X -> Q X Q^T
    2. Log Map: T = log(X)
    3. Residual: T_new = T + MLP(vec(T))
    4. Exp Map: X_new = exp(T_new)

    This allows the model to learn non-isometric deformations (scaling, shearing)
    while maintaining SPD structure via the exponential map.
    """
    def __init__(self, input_dims, hidden_dim=32, use_isometry=True):
        super().__init__(input_dims)
        self.use_isometry = use_isometry
        self.dims = input_dims

        # 1. Base Isometry Parameters (Rotation)
        # We use the QR parametrization from SPDIsometryQR
        if self.use_isometry:
            self.isom_params = Parameter(torch.Tensor(self.dims, self.dims))
            self.isom_params.data.copy_(torch.eye(self.dims))

        # 2. Tangent Space Residual Parameters
        # Vector dimension for symmetric matrix: d(d+1)/2
        self.vec_dim = input_dims * (input_dims + 1) // 2

        # Helper indices for vectorization (upper triangle)
        r, c = torch.triu_indices(self.dims, self.dims)
        self.register_buffer('triu_rows', r)
        self.register_buffer('triu_cols', c)

        # Residual MLP: vec(T) -> vec(T)
        # Initialize to identity (residual=0) for stability
        self.tangent_mlp = nn.Sequential(
            nn.Linear(self.vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.vec_dim)
        )

        # Zero initialization for the last layer ensures the transform starts as identity (or pure isometry)
        nn.init.zeros_(self.tangent_mlp[-1].weight)
        nn.init.zeros_(self.tangent_mlp[-1].bias)

    def _vectorize(self, sym_mat):
        # [B, d, d] -> [B, vec_dim]
        return sym_mat[..., self.triu_rows, self.triu_cols]

    def _unvectorize(self, vec):
        # [B, vec_dim] -> [B, d, d]
        B = vec.shape[0]
        mat = torch.zeros(B, self.dims, self.dims, device=vec.device, dtype=vec.dtype)
        mat[..., self.triu_rows, self.triu_cols] = vec
        # Fill lower triangle
        mat = mat + mat.transpose(-1, -2)
        # Diagonal elements were added twice, so subtract them once
        diag_ind = torch.arange(self.dims, device=vec.device)
        mat[..., diag_ind, diag_ind] /= 2
        return mat

    def forward(self, mat_feats, M=None):
        # 1. Apply Base Isometry (Optional)
        if self.use_isometry:
            W = self.isom_params
            q, r = torch.linalg.qr(W)
            d = torch.diag(r, 0)
            ph = d.sign()
            q = q * ph
            Q = q.unsqueeze(0) # [1, d, d]

            if M is not None:
                Q = Q @ M.transpose(-2, -1)

            mat_feats = Q @ mat_feats @ Q.transpose(-1, -2)

        # 2. Tangent Space Residual
        # Log map: SPD -> Sym
        tan_feats = SPDManifold.logmap_id(mat_feats)

        # Vectorize
        vec_feats = self._vectorize(tan_feats)

        # Apply Residual MLP
        residual = self.tangent_mlp(vec_feats)
        vec_feats_new = vec_feats + residual

        # Unvectorize
        tan_feats_new = self._unvectorize(vec_feats_new)

        # Exp map: Sym -> SPD
        mat_feats_new = SPDManifold.expmap_id(tan_feats_new)

        return mat_feats_new