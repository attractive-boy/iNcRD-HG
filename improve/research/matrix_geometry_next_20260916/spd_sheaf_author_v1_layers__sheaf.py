import torch
import torch.nn as nn
from manifolds import SPDManifold, linalg as lalg
from mappers import Vec2SymMat

class SPDSheafLayer(nn.Module):
    def __init__(self, stalk_dim):
        super().__init__()
        self.stalk_dim = stalk_dim

        # Instead of reflection_net over vectors, learn SPD transformation
        self.proj = nn.Linear(2 * stalk_dim * stalk_dim, stalk_dim * stalk_dim)

    def forward(self, x, edge_index):
        num_nodes, n, _ = x.size()
        src, dst = edge_index

        # Move SPD to tangent space at Identity
        log_x = SPDManifold.logmap_id(x)

        # Learnable transformation (similar to reflection, but SPD-aware)
        v_s = self.proj(torch.cat([log_x[src].reshape(len(src), -1),
                                   log_x[dst].reshape(len(dst), -1)], dim=1))
        v_s = v_s.reshape(-1, n, n)
        v_s = lalg.sym(v_s)

        # Back to SPD
        F_s = SPDManifold.expmap_id(v_s)

        # Diffusion: SPD addition (schematic)
        out = x.clone()
        new_out = out.clone()  # Use new tensor to save updates

        for i in range(len(src)):
            s, d = src[i], dst[i]

            msg_s_to_d = F_s[i] @ out[d] @ F_s[i].transpose(-1, -2)
            msg_d_to_s = F_s[i] @ out[s] @ F_s[i].transpose(-1, -2)

            new_out[s] = SPDManifold.addition_id(out[s], msg_s_to_d)
            new_out[d] = SPDManifold.addition_id(out[d], msg_d_to_s)

        out = new_out


        return out
