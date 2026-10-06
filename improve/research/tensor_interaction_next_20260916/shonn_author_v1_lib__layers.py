import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, train_eigenvectors=True):
        super(SpectralLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.train_eigenvectors = train_eigenvectors
        
        if self.bias:
            # Add bias neuron to input features
            self.phi = nn.Parameter(torch.Tensor(out_features, in_features + 1), requires_grad=train_eigenvectors)
            self.l_in = nn.Parameter(torch.Tensor(in_features + 1))
        else:
            self.phi = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=train_eigenvectors)
            self.l_in = nn.Parameter(torch.Tensor(in_features))
        
        self.l_out = nn.Parameter(torch.Tensor(out_features))
            
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.phi)
        nn.init.uniform_(self.l_in, -1, 1)
        nn.init.uniform_(self.l_out, -1, 1)

    def forward(self, input):
        # Weight matrix W = phi * diag(l_in) - diag(l_out) * phi
        # So w_ij = (l_j - l_i) * phi_ij

        if self.bias:
            ones = torch.ones(input.size(0), 1, device=input.device)
            input = torch.cat([input, ones], dim=1)
        
        weight = (self.l_in.unsqueeze(0) - self.l_out.unsqueeze(1)) * self.phi
        return F.linear(input, weight, bias=None)

class SpectralTriadic(nn.Module):
    def __init__(self, in_features, out_features, bias=True, train_eigenvectors=True, train_triadic_eigenvectors=False):
        super(SpectralTriadic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.train_eigenvectors = train_eigenvectors
        
        if self.bias:
            # Add bias neuron to input features
            self.phi = nn.Parameter(torch.Tensor(out_features, in_features + 1), requires_grad=train_eigenvectors)
            self.l_in = nn.Parameter(torch.Tensor(in_features + 1))
        else:
            self.phi = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=train_eigenvectors)
            self.l_in = nn.Parameter(torch.Tensor(in_features))
        
        self.l_out = nn.Parameter(torch.Tensor(out_features))
        
        # Triadic parameters
        num_pairs = in_features * (in_features + 1) // 2 # Binomial(in_features, 2) + in_features
        self.phi_triadic = nn.Parameter(torch.Tensor(out_features, num_pairs), requires_grad=train_triadic_eigenvectors)
        self.l_in_triadic = nn.Parameter(torch.Tensor(num_pairs))
        self.l_out_triadic = nn.Parameter(torch.Tensor(out_features))
        
        # Register upper triangular indices for triadic interactions
        # We use register_buffer to store these indices as they are not parameters
        # but must be on the same device as the module
        # Note that torch.triu_indices in practice does the lexicographical ordering
        # with the i' being the column number!
        indices = torch.triu_indices(in_features, in_features, offset=0)
        self.register_buffer('triu_rows', indices[0])
        self.register_buffer('triu_cols', indices[1])
            
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.phi)
        nn.init.uniform_(self.l_in, -1, 1)
        nn.init.uniform_(self.l_out, -1, 1)
        nn.init.xavier_uniform_(self.phi_triadic)
        nn.init.uniform_(self.l_in_triadic, -1, 1)
        nn.init.uniform_(self.l_out_triadic, -1, 1)

    def forward(self, input):
        if self.bias:
            ones = torch.ones(input.size(0), 1, device=input.device)
            input_linear = torch.cat([input, ones], dim=1)
        else:
            input_linear = input
        
        # Linear weight
        # w_kj = (l_j - l_k) * phi_kj
        weight_linear = (self.l_in.unsqueeze(0) - self.l_out.unsqueeze(1)) * self.phi
        out_linear = F.linear(input_linear, weight_linear, bias=None)
        
        # Triadic weight
        # w_kij = (L_ij - L_k) * phi_kij
        weight_triadic = (self.l_in_triadic.unsqueeze(0) - self.l_out_triadic.unsqueeze(1)) * self.phi_triadic
        
        # Compute bilinear term: sum_{j>=i} w_kij * x_i * x_j
        x_i = input[:, self.triu_rows]
        x_j = input[:, self.triu_cols]
        x_pairs = x_i * x_j
        
        out_triadic = F.linear(x_pairs, weight_triadic, bias=None)
        
        return out_linear + out_triadic

class SpectralTriadicOnly(nn.Module):
    def __init__(self, in_features, out_features, train_triadic_eigenvectors=False):
        super(SpectralTriadicOnly, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        num_pairs = in_features * (in_features + 1) // 2 # Binomial(in_features, 2) + in_features
        self.phi_triadic = nn.Parameter(torch.Tensor(out_features, num_pairs), requires_grad=train_triadic_eigenvectors)
        self.l_in_triadic = nn.Parameter(torch.Tensor(num_pairs))
        self.l_out_triadic = nn.Parameter(torch.Tensor(out_features))
        
        # Register upper triangular indices for triadic interactions
        # We use register_buffer to store these indices as they are not parameters
        # but must be on the same device as the module
        # Note that torch.triu_indices in practice does the lexicographical ordering
        # with the i' being the column number!
        indices = torch.triu_indices(in_features, in_features, offset=0)
        self.register_buffer('triu_rows', indices[0])
        self.register_buffer('triu_cols', indices[1])
            
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.phi_triadic)
        nn.init.uniform_(self.l_in_triadic, -1, 1)
        nn.init.uniform_(self.l_out_triadic, -1, 1)

    def forward(self, input):
        
        # Triadic weight
        # w_kij = (L_ij - L_k) * phi_kij
        weight_triadic = (self.l_in_triadic.unsqueeze(0) - self.l_out_triadic.unsqueeze(1)) * self.phi_triadic
        
        # Compute bilinear term: sum_{j>=i} w_kij * x_i * x_j
        x_i = input[:, self.triu_rows]
        x_j = input[:, self.triu_cols]
        x_pairs = x_i * x_j
        
        out_triadic = F.linear(x_pairs, weight_triadic, bias=None)
        
        return out_triadic
    
class DirectSpaceTriadic(nn.Module):
    def __init__(self, in_features, out_features):
        super(DirectSpaceTriadic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)
        
        num_pairs = in_features * (in_features + 1) // 2 # Binomial(in_features, 2) + in_features
        self.weight_triadic = nn.Parameter(torch.Tensor(out_features, num_pairs))

        # Register upper triangular indices for triadic interactions
        # We use register_buffer to store these indices as they are not parameters
        # but must be on the same device as the module
        # Note that torch.triu_indices in practice does the lexicographical ordering
        # with the i' being the column number!
        indices = torch.triu_indices(in_features, in_features, offset=0)
        self.register_buffer('triu_rows', indices[0])
        self.register_buffer('triu_cols', indices[1])

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.weight_triadic)

    def forward(self, input):
        out_linear = self.linear(input)
        
        # Compute bilinear term: sum_{j>=i} w_kij * x_i * x_j
        x_i = input[:, self.triu_rows]
        x_j = input[:, self.triu_cols]
        x_pairs = x_i * x_j
        
        out_triadic = F.linear(x_pairs, self.weight_triadic, bias=None)
        
        return out_linear + out_triadic
            