from function_transformer_attention import SpGraphTransAttentionLayer
from utils import manual_dropout, cbrt
from layer import GATConv_complex

from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import torch.nn as nn
import torch

from functools import partial


class SLGCN(torch.nn.Module):
    """ 
    Stuart-Landau Graph Convolutional Network with polar IMEX integration
    dot X = -(zeta + nu*|X|^p)X + sigma(AXW)
    """
    def __init__(self, args, dataset):
        super().__init__()

        self.args = args
        self.implicit_r_tol_break = False
        
        self.dt = getattr(args, 'dt', None)
        self.nlayer = args.nlayer
        self.k = args.k
        self.p = args.p
        self.tol = args.tol
        self.activation = partial(F.leaky_relu, negative_slope=args.leaky_relu_slope) 
        self.dropout = args.dropout
        self.input_dropout = args.input_dropout

        self.enc = nn.Linear(dataset.num_node_features, args.nhid, dtype=torch.cfloat)
        self.dec = nn.Linear(args.nhid, 
                             dataset.num_classes if not args.dataset in ['ESOL', 'FreeSolv', 'Lipo'] else 1)

        self.zeta = args.zeta
        self.nu = args.nu

        self.convs = torch.nn.ModuleList()
        for _ in range(self.nlayer):
            self.convs.append(GCNConv(args.nhid, args.nhid, bias=args.bias))
            self.convs[-1].lin.weight = nn.Parameter(self.convs[-1].lin.weight.detach().cfloat())

    def solve_implicit_r(self, r_prev, max_iter=20): 
        r_next = r_prev.clone()
        
        for _ in range(max_iter):
            residual = r_next - r_prev - self.dt * (self.zeta.real*r_next - self.nu.real * r_next**(self.p+1))
            deriv = 1 -self.dt * (self.zeta.real - (self.p+1)*self.nu.real*r_next**self.p)
            update = residual / deriv
            r_next = r_next - update

            if torch.max(torch.abs(update)) < self.tol:
                return r_next
        else:
            print('Implicit r early exit: max_iter reached', flush=True)
            self.implicit_r_tol_break = True
            return r_next
        
    def solve_explicit_inverse_r(self, r_prev):
        p = (1 - self.dt * self.zeta.real) / (3 * self.nu.real * self.dt)
        q = -r_prev / (2 * self.nu.real * self.dt)
        
        delta = q**2 + p**3

        if torch.all(delta > 0): # if only one real root
            delta = delta**0.5
            u1 = cbrt(-q + delta)
            u2 = cbrt(-q - delta)

            root1 = u1 + u2
            return root1
        else:
            raise NotImplementedError('delta < 0 is not implemented')

    def cartesian_implicit_step(self, z, tol=1e-1, max_iter=20):
        x, y = z.real, z.imag
        
        for _ in range(max_iter):
            x3y = x**2 + 3*y**2
            y3x = 3*x**2 + y**2
            xy2 = 2*x*y

            J11 = 1 - self.dt * (self.zeta.real - self.nu.real * y3x + self.nu.imag * xy2)
            J12 = self.dt * (self.zeta.imag + self.nu.real * xy2 - self.nu.imag * x3y)
            J21 = self.dt * (-self.zeta.imag + self.nu.real * xy2 + self.nu.imag * y3x)
            J22 = 1 - self.dt * (self.zeta.real - self.nu.real * x3y - self.nu.imag * xy2)

            jac = torch.stack([
                torch.stack([J11, J12], axis=-1),
                torch.stack([J21, J22], axis=-1)
            ], axis=-2)

            F = x - z.real - self.dt * (self.zeta.real*x - self.zeta.imag*y - (self.nu.real*x - self.nu.imag*y) * (x**2 + y**2))
            G = y - z.imag - self.dt * (self.zeta.imag*x + self.zeta.real*y - (self.nu.real*y + self.nu.imag*x) * (x**2 + y**2))
            RHS = torch.stack([F, G], axis=-1)[..., None]

            delta = torch.linalg.solve(jac, RHS).squeeze()

            x = x - delta[..., 0]
            y = y - delta[..., 1]

            if torch.max(torch.abs(delta)) < tol:
                z_new = x + 1j*y
                return z_new
        else:
            print('Implicit early exit: max_iter reached')
            z_new = x + 1j*y
            return z_new

    def ode_step(self, x, edge_index, conv):
        theta = torch.angle(x)
        r = torch.abs(x)
        r = self.solve_implicit_r(r) # Newton
        # r = self.solve_explicit_inverse_r(r) 
        theta = theta + self.dt * (self.zeta.imag - self.nu.imag * r**2)
        x_new = torch.polar(r, theta)

        # x_new = self.cartesian_implicit_step(x) # 2D Newton

        agg = torch.view_as_complex(self.activation(torch.view_as_real(conv(x, edge_index))))
        x_new = x_new + self.k*self.dt*agg
        return x_new

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = manual_dropout(x, p=self.input_dropout, training=self.training)
        
        x = x.to(torch.cfloat)
        x = self.enc(x)
        
        for i in range(self.nlayer):
            x = self.ode_step(x, edge_index, self.convs[i])
            
        x = manual_dropout(x, p=self.dropout, training=self.training)
        
        x = self.dec(x.real)
        return x


class SLGAT(SLGCN):
    def __init__(self, args, dataset):
        super(SLGAT, self).__init__(args, dataset)

        # OVERWRITE the inherited convs
        self.convs = torch.nn.ModuleList()
        for _ in range(args.nlayer):
            self.convs.append(GATConv_complex(args.nhid, args.nhid, bias=args.bias, heads=args.nhead, concat=False))
            self.convs[-1].lin.weight = nn.Parameter(self.convs[-1].lin.weight.detach().cfloat())
            

class SLTrans(SLGCN):
    def __init__(self, args, dataset):
        super(SLTrans, self).__init__(args, dataset)

        self.convs = torch.nn.ModuleList()
        for _ in range(args.nlayer):
            self.convs.append(SpGraphTransAttentionLayer(args.nhid, args.natt, args.nhead, add_source=False, bias=args.bias, dtype=torch.cfloat))



class GraphCONGCN(nn.Module):
    def __init__(self, args, dataset):
        super(GraphCONGCN, self).__init__()
        self.dropout = args.dropout
        self.nhid = args.nhid
        self.nlayer = args.nlayer
        self.enc = nn.Linear(dataset.num_features, self.nhid)
        self.convs = nn.ModuleList([GCNConv(self.nhid, self.nhid, bias=args.bias) for _ in range(args.nlayer)])
        self.dec = nn.Linear(self.nhid, dataset.num_classes if not args.dataset in ['ESOL', 'FreeSolv', 'Lipo'] else 1)
        self.dt = args.dt
        self.act_fn = partial(F.leaky_relu, negative_slope=args.leaky_relu_slope) 
        self.alpha = args.alpha
        self.gamma = args.gamma

        self.input_dropout = args.input_dropout
        self.implicit_r_tol_break = False

    def forward(self, data):
        input = data.x.to(torch.float)
        edge_index = data.edge_index
        input = F.dropout(input, self.input_dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayer):
            Y = Y + self.dt*(self.act_fn(self.convs[i](X,edge_index)) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)
        return X


class GraphCONGAT(nn.Module):
    def __init__(self, args, dataset):
        super(GraphCONGAT, self).__init__()
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.dropout = args.dropout
        self.nhead = args.nhead
        self.nhid = args.nhid
        self.nlayer = args.nlayer
        self.act_fn = partial(F.leaky_relu, negative_slope=args.leaky_relu_slope) 
        self.enc = nn.Linear(dataset.num_features, self.nhid)
        self.convs = nn.ModuleList([GATConv(self.nhid, self.nhid, heads=self.nhead, bias=args.bias) for _ in range(args.nlayer)])
        self.dec = nn.Linear(self.nhid, dataset.num_classes if not args.dataset in ['ESOL', 'FreeSolv', 'Lipo'] else 1)
        self.dt = args.dt

        self.input_dropout = args.input_dropout
        self.implicit_r_tol_break = False

    def forward(self, data):
        input = data.x.to(torch.float)
        n_nodes = input.size(0)
        edge_index = data.edge_index
        input = F.dropout(input, self.input_dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayer):
            Y = Y + self.dt*(self.act_fn(self.convs[i](X, edge_index)).view(n_nodes, -1, self.nhead).mean(dim=-1) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)

        return X


class GraphCONTrans(nn.Module):
    def __init__(self, args, dataset):
        super(GraphCONTrans, self).__init__()
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.dropout = args.dropout
        self.nhead = args.nhead
        self.nhid = args.nhid
        self.nlayer = args.nlayer
        self.act_fn = partial(F.leaky_relu, negative_slope=args.leaky_relu_slope) 
        self.enc = nn.Linear(dataset.num_features, self.nhid)
        self.dec = nn.Linear(self.nhid, dataset.num_classes if not args.dataset in ['ESOL', 'FreeSolv', 'Lipo'] else 1)
        self.dt = args.dt

        self.k = args.k
        self.convs = nn.ModuleList([
            SpGraphTransAttentionLayer(args.nhid, args.natt, args.nhead, add_source=False, bias=args.bias, dtype=torch.float) 
            for _ in range(args.nlayer)
        ])
        self.input_dropout = args.input_dropout
        self.implicit_r_tol_break = False

    def forward(self, data):
        input = data.x.to(torch.float)
        edge_index = data.edge_index
        input = F.dropout(input, self.input_dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayer):
            Y = Y + self.dt*(self.k*self.act_fn(self.convs[i](X, edge_index)) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)
        return X
    


class KuramotoGCN(nn.Module):
    def __init__(self, args, dataset):
        super(KuramotoGCN, self).__init__()
        self.dropout = args.dropout
        self.nlayer = args.nlayer
        self.act_fn = partial(F.leaky_relu, negative_slope=args.leaky_relu_slope) 
        self.enc = nn.Linear(dataset.num_features, args.nhid, dtype=torch.cfloat)
        self.dec = nn.Linear(args.nhid, dataset.num_classes if not args.dataset in ['ESOL', 'FreeSolv', 'Lipo'] else 1)
        self.dt = args.dt

        self.k = args.k
        self.omega = args.zeta.imag
        self.input_dropout = args.input_dropout
        self.implicit_r_tol_break = False

        self.convs = nn.ModuleList()
        for _ in range(args.nlayer):
            self.convs.append(GCNConv(args.nhid, args.nhid, bias=args.bias))
            self.convs[-1].lin.weight = nn.Parameter(self.convs[-1].lin.weight.detach().cfloat())

    def forward(self, data):
        input = data.x.to(torch.cfloat)
        edge_index = data.edge_index
        input = manual_dropout(input, self.input_dropout, training=self.training)
        X = self.enc(input)
        X = manual_dropout(X, self.dropout, training=self.training)

        X = X / (X.abs() + 1e-8)

        for i in range(self.nlayer):
            dx = self.k*(torch.view_as_complex(self.act_fn(torch.view_as_real(self.convs[i](X,edge_index)))))
            X = X + self.dt*dx
            X = torch.polar(torch.ones_like(X, dtype=torch.float), torch.angle(X) + self.omega*self.dt)
            X = manual_dropout(X, self.dropout, training=self.training)

        X = self.dec(X.real)
        return X


class KuramotoGAT(KuramotoGCN):
    def __init__(self, args, dataset):
        super(KuramotoGAT, self).__init__(args, dataset)
        self.convs = nn.ModuleList()
        for _ in range(args.nlayer):
            self.convs.append(GATConv_complex(args.nhid, args.nhid, heads=args.nhead, bias=args.bias, concat=False))
            self.convs[-1].lin.weight = nn.Parameter(self.convs[-1].lin.weight.detach().cfloat())


class KuramotoTrans(KuramotoGCN):
    def __init__(self, args, dataset):
        super(KuramotoTrans, self).__init__(args, dataset)
        self.convs = nn.ModuleList([SpGraphTransAttentionLayer(args.nhid, args.natt, args.nhead, bias=args.bias, add_source=False, dtype=torch.cfloat) for _ in range(args.nlayer)])






class BaseGCN(nn.Module):
    def __init__(self, args, dataset):
        super(BaseGCN, self).__init__()
        self.dropout = args.dropout
        self.nhid = args.nhid
        self.nlayer = args.nlayer
        self.enc = nn.Linear(dataset.num_features, self.nhid)
        self.convs = nn.ModuleList([GCNConv(self.nhid, self.nhid, bias=args.bias) for _ in range(self.nlayer)])
        self.dec = nn.Linear(self.nhid, dataset.num_classes if not args.dataset in ['ESOL', 'FreeSolv', 'Lipo'] else 1)
        self.act_fn = partial(F.leaky_relu, negative_slope=args.leaky_relu_slope)
        
        self.input_dropout = args.input_dropout
        self.implicit_r_tol_break = False

    def forward(self, data):
        input = data.x.to(torch.float)
        edge_index = data.edge_index
        input = F.dropout(input, self.input_dropout, training=self.training)
        X = self.act_fn(self.enc(input))
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayer):
            X = self.act_fn(self.convs[i](X, edge_index)) + X # add residual connection for consistency
            X = F.dropout(X, self.dropout, training=self.training) 

        X = self.dec(X)
        return X

class BaseGAT(nn.Module):
    def __init__(self, args, dataset):
        super(BaseGAT, self).__init__()
        self.dropout = args.dropout
        self.nhead = args.nhead
        self.nhid = args.nhid
        self.nlayer = args.nlayer
        self.enc = nn.Linear(dataset.num_features, self.nhid)
        self.convs = nn.ModuleList([GATConv(self.nhid, self.nhid, heads=self.nhead, bias=args.bias) for _ in range(self.nlayer)])
        self.dec = nn.Linear(self.nhid, dataset.num_classes if not args.dataset in ['ESOL', 'FreeSolv', 'Lipo'] else 1)
        self.act_fn = partial(F.leaky_relu, negative_slope=args.leaky_relu_slope)
        
        self.input_dropout = args.input_dropout
        self.implicit_r_tol_break = False

    def forward(self, data):
        input = data.x.to(torch.float)
        n_nodes = input.size(0)
        edge_index = data.edge_index
        input = F.dropout(input, self.input_dropout, training=self.training)
        X = self.act_fn(self.enc(input))
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayer):
            X = self.act_fn(self.convs[i](X, edge_index)).view(n_nodes, -1, self.nhead).mean(dim=-1) + X # add residual connection for consistency
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)
        return X
    
class BaseTrans(nn.Module):
    def __init__(self, args, dataset):
        super(BaseTrans, self).__init__()
        self.dropout = args.dropout
        self.nhead = args.nhead
        self.nhid = args.nhid
        self.nlayer = args.nlayer
        self.enc = nn.Linear(dataset.num_features, self.nhid)
        self.dec = nn.Linear(self.nhid, dataset.num_classes if not args.dataset in ['ESOL', 'FreeSolv', 'Lipo'] else 1)
        self.act_fn = partial(F.leaky_relu, negative_slope=args.leaky_relu_slope)
        
        self.k = args.k
        self.convs = nn.ModuleList([
            SpGraphTransAttentionLayer(args.nhid, args.natt, args.nhead, add_source=False, bias=args.bias, dtype=torch.float) 
            for _ in range(args.nlayer)
        ])
        self.input_dropout = args.input_dropout
        self.implicit_r_tol_break = False

    def forward(self, data):
        input = data.x.to(torch.float)
        edge_index = data.edge_index
        input = F.dropout(input, self.input_dropout, training=self.training)
        X = self.act_fn(self.enc(input))
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayer):
            X = self.k * self.act_fn(self.convs[i](X, edge_index)) + X  # add residual connection for consistency
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)
        return X