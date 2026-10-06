# SPD operations definition

from typing import Tuple
import torch
from . import linalg as lalg

class SPDManifold:
    def __init__(self):
        super().__init__()

    @staticmethod
    def dist(self, a: torch.Tensor, b: torch.Tensor, keepdim=True) -> Tuple[torch.Tensor, torch.Tensor]:

        raise NotImplementedError("dist is not implemented in SPDManifold")

    @staticmethod
    def expmap_id(x: torch.Tensor) -> torch.Tensor:
        """
        Performs an exponential map using the Identity as basepoint :math:`\operatorname{Exp}_{Id}(u)`.
        """
        x = lalg.sym(x)
        return lalg.sym_funcm(x, torch.exp)

    @staticmethod
    def logmap_id(x: torch.Tensor) -> torch.Tensor:
        """
        Perform an logarithmic map using the Identity as basepoint :math:`\operatorname{Log}_{Id}(y)`.
        """
        x = lalg.sym(x)
        return lalg.sym_funcm(x, torch.log)

    @staticmethod
    def addition_id(a: torch.Tensor, b: torch.Tensor):
        """
        Performs addition using the Identity as basepoint.

        The addition on SPD using the identity as basepoint is :math:`A \oplus_{Id} B = sqrt(A) B sqrt(A)`.
        """
        sqrt_a = lalg.sym_sqrtm(a)
        return sqrt_a @ b @ sqrt_a
    '''
    def addition_id(a, b, eps=1e-6, max_val=1e6):
        a = a.clamp(min=eps, max=max_val)
        b = b.clamp(min=eps, max=max_val)
        sqrt_a = lalg.sym_sqrtm(a)
        return sqrt_a @ b @ sqrt_a

    '''

# Replace SPDManifoldOps with this safer version
class SPDManifoldOps:
    """SPD manifold operations using affine-invariant Riemannian metric.
    Added explicit symmetrization and eps-regularization for numerical stability.
    """

    @staticmethod
    def _sym(A):
        return 0.5 * (A + A.transpose(-2, -1))

    @staticmethod
    def _add_eps_eye(A, eps):
        d = A.shape[-1]
        return A + eps * torch.eye(d, device=A.device).unsqueeze(0).expand_as(A)

    @staticmethod
    def parallel_transport(X, Y, U, epsilon=1e-6):
        # X, Y, U: [B, d, d] or [d,d] (we assume batch)
        # Return symmetric transported_U
        # Compute Y * X^{-1}
        # Add small reg to X to avoid inversion issues
        X_reg = SPDManifoldOps._add_eps_eye(X, epsilon)
        X_inv = torch.linalg.inv(X_reg)
        YX_inv = torch.bmm(Y, X_inv)

        # eigendecompose YX_inv (Hermitian), clamp and form sqrt
        eigvals, eigvecs = torch.linalg.eigh(SPDManifoldOps._sym(YX_inv))
        eigvals = torch.clamp(eigvals, min=epsilon)
        sqrt_eigvals = torch.sqrt(eigvals)

        YX_inv_sqrt = torch.bmm(eigvecs, torch.diag_embed(sqrt_eigvals))
        YX_inv_sqrt = torch.bmm(YX_inv_sqrt, eigvecs.transpose(-2, -1))
        YX_inv_sqrt = SPDManifoldOps._sym(YX_inv_sqrt)

        transported_U = torch.bmm(torch.bmm(YX_inv_sqrt, U), YX_inv_sqrt)
        transported_U = SPDManifoldOps._sym(transported_U)
        #transported_U = transported_U + epsilon * torch.eye(transported_U.shape[-1], device=transported_U.device).unsqueeze(0)
        return transported_U

    @staticmethod
    def frechet_mean(X, Y, epsilon=1e-6):
        # safer Frechet mean between two SPD matrices X and Y (batchable)
        X_sym = SPDManifoldOps._sym(X)
        eigvals_X, eigvecs_X = torch.linalg.eigh(X_sym)
        eigvals_X = torch.clamp(eigvals_X, min=epsilon)
        sqrt_eigvals_X = torch.sqrt(eigvals_X)
        inv_sqrt_eigvals_X = 1.0 / sqrt_eigvals_X

        X_sqrt = torch.bmm(eigvecs_X, torch.diag_embed(sqrt_eigvals_X))
        X_sqrt = torch.bmm(X_sqrt, eigvecs_X.transpose(-2, -1))
        X_inv_sqrt = torch.bmm(eigvecs_X, torch.diag_embed(inv_sqrt_eigvals_X))
        X_inv_sqrt = torch.bmm(X_inv_sqrt, eigvecs_X.transpose(-2, -1))

        temp = torch.bmm(torch.bmm(X_inv_sqrt, SPDManifoldOps._sym(Y)), X_inv_sqrt)
        # eigendecompose temp
        eigvals_temp, eigvecs_temp = torch.linalg.eigh(SPDManifoldOps._sym(temp))
        eigvals_temp = torch.clamp(eigvals_temp, min=epsilon)
        sqrt_eigvals_temp = torch.sqrt(eigvals_temp)

        temp_sqrt = torch.bmm(eigvecs_temp, torch.diag_embed(sqrt_eigvals_temp))
        temp_sqrt = torch.bmm(temp_sqrt, eigvecs_temp.transpose(-2, -1))
        temp_sqrt = SPDManifoldOps._sym(temp_sqrt)

        M = torch.bmm(torch.bmm(X_sqrt, temp_sqrt), X_sqrt)
        M = SPDManifoldOps._sym(M)
        #M = M + epsilon * torch.eye(M.shape[-1], device=M.device).unsqueeze(0)
        return M

