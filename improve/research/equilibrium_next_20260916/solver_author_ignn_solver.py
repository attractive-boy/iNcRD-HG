"""
IGNN-Solver: Improved IGNN with advanced solvers.

This module implements IGNN with Anderson acceleration and other
advanced fixed-point solvers for better convergence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.nn import Module, Parameter

from models.solvers import anderson_solver
from utils import get_spectral_radius, projection_norm_inf


class ImplicitGraphLayer(Module):
    """
    Improved implicit graph layer for IGNN-Solver.
    
    Uses more efficient formulation for fixed-point iteration.
    """
    
    def __init__(self, in_features, out_features, num_nodes, kappa=0.99):
        """
        Initialize implicit graph layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_nodes: Number of nodes
            kappa: Spectral constraint parameter
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.kappa = kappa
        
        self.W = Parameter(torch.FloatTensor(out_features, out_features))
        self.B = Parameter(torch.FloatTensor(in_features, out_features))
        
        self.init_parameters()
    
    def init_parameters(self):
        """Initialize layer parameters."""
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)
    
    def forward(self, X_prev, A, U, phi=F.relu):
        """
        Forward pass: one iteration of fixed-point equation.
        
        Args:
            X_prev: Previous iteration
            A: Adjacency matrix
            U: Input features
            phi: Activation function
            
        Returns:
            Next iteration
        """
        # Compute bias term
        A_U_B = torch.spmm(torch.spmm(A, U), self.B)
        
        # Apply spectral constraint
        if self.kappa != 0:
            A_rho = get_spectral_radius(A)
            self.W = projection_norm_inf(self.W, kappa=self.kappa / A_rho)
        
        # Fixed-point iteration step
        X_W = torch.spmm(X_prev, self.W)
        A_X_W = torch.spmm(A, X_W)
        X_new = phi(A_X_W + A_U_B)
        
        return X_new


class IGNNSolver(nn.Module):
    """
    IGNN-Solver: IGNN with advanced fixed-point solvers.
    
    Supports multiple solver types and gradient computation modes.
    """
    
    def __init__(self, nfeat, nhid, nclass, num_node, kappa, dropout,
                 model_mode="implicit",
                 grad_mode="solver_grad",
                 fw_solver=None, bw_solver=None,
                 **kwargs):
        """
        Initialize IGNN-Solver.
        
        Args:
            nfeat: Number of input features
            nhid: Number of hidden units
            nclass: Number of output classes
            num_node: Number of nodes
            kappa: Spectral constraint parameter
            dropout: Dropout rate
            model_mode: 'implicit' or 'explicit'
            grad_mode: 'solver_grad' or 'explicit_grad'
            fw_solver: Forward solver (default: anderson_solver)
            bw_solver: Backward solver (default: anderson_solver)
        """
        super().__init__()
        
        self.hook = None
        self.kwargs = kwargs
        self.dropout = dropout
        
        # Core components
        self.implicit_layer = ImplicitGraphLayer(nfeat, nhid, num_node, kappa)
        self.Z_0 = Parameter(torch.zeros(num_node, nhid), requires_grad=False)
        self.classifier = nn.Linear(nhid, nclass, bias=False)
        
        # Model settings
        self.model_mode = model_mode
        
        if model_mode == "implicit":
            self.grad_mode = grad_mode
            if grad_mode == "solver_grad":
                self.fw_solver = fw_solver if fw_solver else anderson_solver
                self.bw_solver = bw_solver if bw_solver else anderson_solver
    
    def forward(self, U, A, **kwargs):
        """
        Forward pass of IGNN-Solver.
        
        Args:
            U: Input features
            A: Adjacency matrix
            **kwargs: Additional arguments (threshold, n_layer, etc.)
            
        Returns:
            dict: Predictions and embeddings
        """
        # Initialize fixed point estimate
        z_init = torch.zeros_like(self.Z_0)
        z_star = z_init
        threshold = kwargs.get('threshold', 20)
        
        if self.model_mode == "explicit":
            # Explicit mode: fixed number of iterations
            n_layer = kwargs.get('n_layer', 3)
            for _ in range(n_layer):
                z_star = self.implicit_layer(z_star, A, U)
            new_z_star = z_star
            
        elif self.model_mode == "implicit":
            # Implicit mode: solve to fixed point
            with torch.no_grad():
                result_fw = self.fw_solver(
                    lambda z: self.implicit_layer(z, A, U),
                    x0=z_init,
                    threshold=threshold
                )
                new_z_star = result_fw['result']
            
            # Backward pass setup
            if self.training:
                new_z_star = self.implicit_layer(z_star.requires_grad_(), A, U)
                
                if self.grad_mode == "explicit_grad":
                    pass  # Use standard backprop
                    
                elif self.grad_mode == "solver_grad":
                    # Use solver for backward pass
                    def backward_hook(grad):
                        if self.hook is not None:
                            self.hook.remove()
                            torch.cuda.synchronize()
                        
                        # Solve adjoint equation
                        f_g = lambda y: autograd.grad(
                            new_z_star, z_star, y, retain_graph=True
                        )[0] + grad
                        
                        result_bw = self.bw_solver(
                            f_g, x0=torch.zeros_like(grad), **self.kwargs
                        )
                        new_grad = result_bw['result']
                        
                        return new_grad
                    
                    self.hook = new_z_star.register_hook(backward_hook)
        
        # Apply dropout and classifier
        output = F.dropout(new_z_star, self.dropout, training=self.training)
        output = self.classifier(output)
        
        return {
            "label_pred": output,
            "emb": new_z_star,
        }
