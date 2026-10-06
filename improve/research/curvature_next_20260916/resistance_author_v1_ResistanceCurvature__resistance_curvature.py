import torch
from typing import Tuple

class ResistanceCurvature():
    """
        A class to compute resistance curvature for graphs using PyTorch.

        This class implements methods to calculate node and edge curvatures based on
        resistance distance metrics derived from graph Laplacians.
    """
    def __init__(self,device: torch.device = None):
        """
        Initialize the ResistanceCurvature calculator.

        Args:
            device: The computation device (e.g., 'cpu' or 'cuda').
        """
        self.device=device if device is not None else \
                 torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def cal_laplacian(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
            Compute the unnormalized graph Laplacian matrix.

            Laplacian L = D - W, where D is degree matrix and W is weight matrix.

            Args:
                weight_matrix: Input weight matrix.

            Returns:
                Laplacian matrix of the graph.
        """
        degree_matrix=torch.diag(torch.sum(weight_matrix,dim=1))
        return degree_matrix-weight_matrix

    def cal_laplacian_inverse_disturb(self,
                                    weight_matrix: torch.Tensor,
                                    disturb_factor: float = 1e-3) -> torch.Tensor:
        """
        Compute the inverse of Laplacian with small diagonal disturbance for numerical stability.

        Args:
            weight_matrix: Input weight matrix.
            disturb_factor: Small positive value to add to diagonal for numerical stability.

        Returns:
            Inverse of the disturbed Laplacian matrix.
        """
        laplacian = self.cal_laplacian(weight_matrix)
        disturbed_laplacian = laplacian + disturb_factor * torch.eye(
            laplacian.shape[0], device=self.device
        )
        # Cholesky decomposition for efficient inversion
        L = torch.linalg.cholesky(disturbed_laplacian)
        return torch.cholesky_inverse(L)

    def cal_laplacian_pseudo_inverse(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the Moore-Penrose pseudo-inverse of the Laplacian matrix.

        Args:
            weight_matrix: Input weight matrix.

        Returns:
            Pseudo-inverse of the Laplacian matrix.
        """
        laplacian = self.cal_laplacian(weight_matrix)
        return torch.linalg.pinv(laplacian)
    def cal_resistance_matrix(self, laplacian_inverse: torch.Tensor, affinity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the resistance distance matrix from Laplacian inverse.

        Resistance distance between nodes i and j is:
        R_ij = L^+_ii + L^+_jj - L^+_ij - L^+_ji

        Args:
            laplacian_inverse: Inverse or pseudo-inverse of Laplacian matrix.
            affinity_matrix: Binary adjacency matrix to mask non-existent edges.

        Returns:
            Resistance distance matrix masked by adjacency.
        """
        diag = torch.diag(laplacian_inverse).reshape(1, laplacian_inverse.shape[0])
        resistance=(diag + diag.T) - (laplacian_inverse + laplacian_inverse.T)
        return resistance*affinity_matrix
    def cal_node_curvature(self,weight_matrix: torch.Tensor, resistence_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute node curvature from resistance distances.

        Node curvature: k_i = 1 - 0.5 * sum_j w_ij R_ij

        Args:
            weight_matrix: Original weight matrix.
            resistance_matrix: Resistance distance matrix.

        Returns:
            Vector of node curvatures.
        """
        return 1-0.5*torch.sum(resistence_matrix*weight_matrix,dim=1)

    def cal_edge_curvature(self,node_curvature: torch.Tensor,resistance_matrix: torch.Tensor,affinity_matrix: torch.Tensor)-> torch.Tensor:
        """
        Compute edge curvature from node curvatures and resistance distances.

        Edge curvature: k_ij = 2(k_i + k_j)/R_ij for existing edges

        Args:
            node_curvature: Node curvature values.
            resistance_matrix: Resistance distance matrix.
            affinity_matrix: Binary adjacency matrix.

        Returns:
            Matrix of edge curvatures.
        """
        # Broadcast node curvatures to pairwise matrix
        k_i = node_curvature.unsqueeze(1)
        k_j = node_curvature.unsqueeze(0)
        pairwise_sum = k_i + k_j

        # Compute reciprocal of resistance with masking
        inv_resistance = torch.where(
            resistance_matrix != 0,
            1.0 / resistance_matrix,
            torch.zeros_like(resistance_matrix)
        )
        return 2 * pairwise_sum * inv_resistance * affinity_matrix
    def adjust_weights_for_leaf_nodes_matrix(self,matrix):
        """
        Adjust weights for edges connected to leaf nodes (degree=1).

        Ensures weights connected to leaf nodes are positive by flipping signs if negative.

        Args:
            matrix: Weight matrix to adjust.

        Returns:
            Adjusted weight matrix.
        """
        degrees = torch.sum(matrix != 0, dim=1)
        is_leaf = degrees == 1

        # Find edges connected to leaf nodes
        is_leaf = is_leaf.unsqueeze(1)
        leaf_edges = is_leaf | is_leaf.T

        # Flip sign of negative weights connected to leaves
        to_adjust = (matrix < 0) & leaf_edges
        matrix[to_adjust] = -matrix[to_adjust]
        return matrix

    def cal_curvature(
                self,
                weight_matrix: torch.Tensor,
                use_disturb: bool = True,
                disturb_factor: float = 1e-3
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Compute both node and edge curvatures.

        Args:
            weight_matrix: Input weight matrix.
            use_disturb: Whether to use disturbed Laplacian inverse for stability.
            disturb_factor: Disturbance factor for numerical stability.

        Returns:
            Tuple of (node_curvatures, edge_curvatures)
        """
        affinity_matrix = (weight_matrix != 0).float()

        if use_disturb:
            laplacian_inverse = self.cal_laplacian_inverse_disturb(
                weight_matrix, disturb_factor
            )
        else:
            laplacian_inverse = self.cal_laplacian_pseudo_inverse(weight_matrix)

        resistance_matrix = self.cal_resistance_matrix(
            laplacian_inverse, affinity_matrix
        )

        node_curvature = self.cal_node_curvature(weight_matrix, resistance_matrix)
        edge_curvature = self.cal_edge_curvature(
            node_curvature, resistance_matrix, affinity_matrix
        )

        # Scale curvatures by minimum resistance for stability
        resistance_matrix_min = resistance_matrix[resistance_matrix > 0].min() if (resistance_matrix > 0).any() else None
        edge_curvature = edge_curvature * resistance_matrix_min / 2

        # Adjust leaf node connections
        edge_curvature = self.adjust_weights_for_leaf_nodes_matrix(edge_curvature)
        return node_curvature, edge_curvature
    def cal_curvature_flow(
                self,
                weight_matrix: torch.Tensor,
                n_iter: int = 1,
                step: float = 1.0,
                flow_norm: bool = False,
                dtype: torch.dtype = torch.float32,
                use_disturb: bool = True,
                disturb_factor: float = 1e-3
        ) -> torch.Tensor:
        """
        Perform curvature flow iterations to update edge weights.

        Args:
            weight_matrix: Initial weight matrix (similarity matrix).
            n_iter: Number of flow iterations.
            step: Learning rate/step size for updates.
            flow_norm: Whether to use normalized flow.
            dtype: Data type for computation.
            use_disturb: Whether to use disturbed Laplacian inverse for stability.
            disturb_factor: Disturbance factor for numerical stability.

        Returns:
            updated_weight_matrix
        """

        weight_matrix=weight_matrix.to(dtype)
        affinity_matrix=(weight_matrix != 0).float()
        for _ in range(1,n_iter+1):
            print(weight_matrix)
            if use_disturb:
                laplacian_inverse=self.cal_laplacian_inverse_disturb(weight_matrix,disturb_factor)
            else:
                laplacian_inverse=self.cal_laplacian_pseudo_inverse(weight_matrix)
            resistance_matrix = self.cal_resistance_matrix(laplacian_inverse,affinity_matrix)

            node_curvature = self.cal_node_curvature(weight_matrix, resistance_matrix)
            edge_curvature = self.cal_edge_curvature(node_curvature, resistance_matrix, affinity_matrix)

            # Scale curvatures by minimum resistance for stability
            resistance_matrix_min = resistance_matrix[resistance_matrix > 0].min() if (resistance_matrix > 0).any() else None
            edge_curvature = edge_curvature * resistance_matrix_min /2

            # Adjust leaf node connections
            edge_curvature = self.adjust_weights_for_leaf_nodes_matrix(edge_curvature)

            # Convert weights to conductances (reciprocal)
            conductance  = torch.where(weight_matrix != 0, 1 / weight_matrix, 0.0)

            # Update conductances based on curvature flow
            if flow_norm==False:
                conductance -=step*edge_curvature*conductance
            else:
                conductance  -= step *conductance *(edge_curvature-(torch.sum(edge_curvature*conductance )/torch.sum(conductance)))
            # Convert back to weights
            weight_matrix = torch.where(conductance  != 0, 1 / conductance , 0.0)
        return weight_matrix
