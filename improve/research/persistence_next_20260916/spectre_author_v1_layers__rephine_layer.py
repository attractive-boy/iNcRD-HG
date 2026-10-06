import torch
import torch.nn as nn
from ph_cpu import compute_persistence_homology_batched_mt
from rephine_mt import compute_rephine_batched_mt
from spectre import compute_spectre_batched_mt as compute_spectre
#from spectre_power import compute_spectre_power_batched_mt as compute_fast_spectre

#from spectre_lobpcg import compute_spectre_lobpcg_batched_mt as compute_fast_spectre

from spectre_scheduling import compute_spectre_scheduling_batched_mt as compute_fast_spectre

from layers.deepsets import DeepSetLayer0, DeepSetLayer1
from utils.utils import remove_duplicate_edges

class RephineLayer(nn.Module):
    def __init__(
        self,
        n_features,
        n_filtrations,
        filtration_hidden,
        out_dim,
        diagram_type="rephine",
        dim1=True,
        sig_filtrations=True,
        reduce_tuples=False,
        out_dim_eigen_deep_set=4
    ):
        super().__init__()

        final_filtration_activation = nn.Sigmoid() if sig_filtrations else nn.Identity()

        if diagram_type == "rephine":
            self.persistence_fn = compute_rephine_batched_mt
        elif diagram_type == "fast_spectre":
            self.persistence_fn = compute_fast_spectre
        elif diagram_type == "spectre":
            self.persistence_fn = compute_spectre
        elif diagram_type == "standard":
            self.persistence_fn = compute_persistence_homology_batched_mt

        self.diagram_type = diagram_type
        self.dim1 = dim1
        self.out_dim = out_dim

        self.filtrations = nn.Sequential(
            nn.Linear(n_features, filtration_hidden),
            nn.ReLU(),
            nn.Linear(filtration_hidden, n_filtrations),
            final_filtration_activation,
        )

        if self.diagram_type == "rephine" or self.diagram_type == "spectre" or self.diagram_type == "fast_spectre":
            self.edge_filtrations = nn.Sequential(
                nn.Linear(n_features, filtration_hidden),
                nn.ReLU(),
                nn.Linear(filtration_hidden, n_filtrations),
                final_filtration_activation,
            )

        if self.diagram_type == "spectre" or self.diagram_type == "fast_spectre":
            self.eigen_deep_set0 = nn.Sequential(nn.Linear(1, out_dim_eigen_deep_set),
                                                 nn.ReLU(),
                                                 nn.Linear(out_dim_eigen_deep_set, out_dim_eigen_deep_set))
        self.num_filtrations = n_filtrations
        self.reduce_tuples = reduce_tuples

        self.tuple_size = 3 if self.reduce_tuples else 4

        if diagram_type == "rephine":
            diagram_size = self.tuple_size
        elif diagram_type == "standard":
            self.tuple_size = 2
            diagram_size = 2
        elif diagram_type == "spectre" or diagram_type == "fast_spectre":
            diagram_size = self.tuple_size + out_dim_eigen_deep_set

        self.deepset_fn = DeepSetLayer0(
            in_dim=n_filtrations * diagram_size, out_dim=out_dim
        )

        if dim1:
            self.deepset_fn_dim1 = DeepSetLayer1(in_dim=diagram_size, out_dim=out_dim)
            if self.diagram_type == "spectre" or diagram_type == "fast_spectre":
                self.eigen_deep_set1 = nn.Sequential(nn.Linear(1, out_dim_eigen_deep_set),
                                       nn.ReLU(),
                                       nn.Linear(out_dim_eigen_deep_set, out_dim_eigen_deep_set))

        self.out = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.bn = nn.BatchNorm1d(out_dim)

    def compute_persistence(self, x, edge_index, vertex_slices, edge_slices):
        filtered_v = self.filtrations(x)
        if self.diagram_type == "rephine" or self.diagram_type == "spectre" or self.diagram_type == "fast_spectre":
            filtered_e = self.edge_filtrations(
                x[edge_index[0]] + x[edge_index[1]]
            )  # multiple ways of doing this.
        elif self.diagram_type == "standard":
            filtered_e, _ = torch.max(
                torch.stack((filtered_v[edge_index[0]], filtered_v[edge_index[1]])),
                axis=0,
            )

        vertex_slices = vertex_slices.cpu().long()
        edge_slices = edge_slices.cpu().long()
        filtered_v = filtered_v.transpose(1, 0).cpu().contiguous()
        filtered_e = filtered_e.transpose(1, 0).cpu().contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()

        persistence0, persistence1 = self.persistence_fn(
            filtered_v, filtered_e, edge_index, vertex_slices, edge_slices
        )

        persistence0 = persistence0.to(x.device)
        persistence1 = persistence1.to(x.device)

        if self.diagram_type == "rephine" or self.diagram_type == 'spectre' or self.diagram_type == "fast_spectre":
            full_size = persistence0.shape[2]
            indices = list(range(3, full_size, 1))
            persistence0 = persistence0[:, :, [0, 2, 1] + indices]

            if not self.reduce_tuples:
                persistence0 = torch.cat(
                    (
                        torch.zeros((persistence0.shape[0], persistence0.shape[1], 1)).to(
                            x.device
                        ),
                        persistence0,
                    ),
                    dim=-1,
                )
                persistence1 = torch.cat(
                    (
                        torch.zeros((persistence1.shape[0], persistence1.shape[1], 1)).to(
                            x.device
                        ),
                        persistence1,
                    ),
                    dim=-1,
                )

            persistence0[persistence0.isnan()] = 1.0

        return persistence0, persistence1

    def process_eigenvalues(self, persistence, dim):
        eigen_v = persistence[:, :, self.tuple_size+1:]
        eigen_v = eigen_v.permute(1, 0, 2).unsqueeze(-1)
        mask = eigen_v != 0.0
        if dim == 0:
            x = self.eigen_deep_set0(eigen_v)
        else:
            x = self.eigen_deep_set1(eigen_v)
        h = torch.sum(x * mask, dim=2)   # We could also compute the mean
        return h

    def forward(self, x, data):
        edge_index, vertex_slices, edge_slices, batch = remove_duplicate_edges(data)
        pers0, pers1 = self.compute_persistence(
            x, edge_index, vertex_slices, edge_slices
        )
        x0 = pers0[:, :, :self.tuple_size].permute(1, 0, 2).reshape(pers0.shape[1], -1)

        if self.diagram_type == 'spectre' or self.diagram_type == 'fast_spectre':
            eig0 = self.process_eigenvalues(pers0, dim=0)
            eig0 = eig0.view(eig0.shape[0], -1)
            x0 = torch.cat((x0, eig0), dim=-1)

        x0 = self.deepset_fn(x0, batch)

        if self.dim1:
            x1 = pers1[:, :, :self.tuple_size]
            pers1_mask = ~(x1 == 0).all(dim=-1)
            if self.diagram_type == 'spectre' or self.diagram_type == 'fast_spectre':
                eig1 = self.process_eigenvalues(pers1, dim=1)
                eig1 = eig1.permute(1, 0, 2)
                x1 = torch.cat((x1, eig1), dim=-1)
            x1 = self.deepset_fn_dim1(x1, edge_slices, mask=pers1_mask)
            x1 = x1.mean(dim=0)
            x0 = x0 + x1
        x = x0
        x = self.bn(x)
        x = self.out(x)
        return x
