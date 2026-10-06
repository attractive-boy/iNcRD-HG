import torch.sparse
import torch.autograd as autograd
from solvers import *


class MEGNN_m_iter(nn.Module):
    def __init__(self, n, dv, nhid, threshold, max_iter, gamma):
        super(MEGNN_m_iter, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(dv, dv), requires_grad=True)
        self.I = nn.Parameter(torch.eye(dv))
        self.dv = dv
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        self.max_iter = max_iter
        self.n = n
        self.nhid = nhid
        self.threshold = threshold
        self.f_solver = fwd_solver
        self.b_solver = fwd_solver
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)

    def _inner_func(self, Z, X, adj):
        adj = adj.requires_grad_(False)
        Z_new = self.gamma * torch.sparse.mm(adj, Z) @ g(self.F) + X
        return Z_new

    def forward(self, X, adj):
        with torch.no_grad():
            Z, abs_diff, n_step = self.f_solver(lambda Z: self._inner_func(Z.cuda(), X, adj),
                                                z_init=torch.zeros((self.n, self.dv)).cuda(),
                                                threshold=self.threshold,
                                                max_iter=self.max_iter)
        new_Z = Z
        if self.training:
            new_Z = self._inner_func(Z.requires_grad_(), X, adj)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()
                result, b_abs_diff, n_step = self.b_solver(
                    lambda y: autograd.grad(new_Z, Z.cuda(), y, retain_graph=True)[0] + grad,
                    z_init=torch.zeros((self.n, self.dv)).cuda(), threshold=self.threshold,
                    max_iter=self.max_iter)
                return result

            self.hook = new_Z.register_hook(backward_hook)
        return new_Z, n_step


epsilon_F = 10 ** (-12)


def g(F):
    FF = F.t() @ F
    FF_norm = torch.norm(FF, p='fro')
    return (1 / (FF_norm + epsilon_F)) * FF


def ortho(F, I):
    temp1 = I - F + F.t()
    temp2 = torch.inverse(I + F - F.t())
    return temp1 @ temp2
