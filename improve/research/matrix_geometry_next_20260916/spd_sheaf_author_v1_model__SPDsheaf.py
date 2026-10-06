import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

from loguru import logger
from manifolds import linalg as lalg
from layers import SPDNonLinearity
from manifolds import SPDManifold, SPDTransform, SPDPooling
from .sheaf_models import LocalConcatSheafLearner, QuadraticFormSheafLearner, GatedLocalConcatSheafLearner
from .orthogonal import Orthogonal

from dgl.nn import SAGEConv
from dgl.nn import SumPooling, AvgPooling, MaxPooling


class SPDFeatureInteraction(nn.Module):
    """
    Layer-wise interaction between SPD manifold features and Euclidean features

    Three interaction modes:
    - 'attention': Use attention to weight SPD and feature contributions
    - 'gating': Use gating mechanism to control information flow
    - 'fusion': Direct fusion through projection and combination
    """
    def __init__(self, spd_dim=3, feature_dim=64, interaction_type='attention'):
        super().__init__()
        self.spd_dim = spd_dim
        self.feature_dim = feature_dim
        self.interaction_type = interaction_type

        if interaction_type == 'attention':
            # Attention-based interaction
            # Project SPD tangent space to attention scores
            self.spd_to_attn = nn.Linear(spd_dim * spd_dim, feature_dim)
            self.feature_to_attn = nn.Linear(feature_dim, feature_dim)
            self.attn_combine = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                #nn.LayerNorm(feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, 1)
            )

            # Update features based on SPD
            self.spd_to_feature = nn.Linear(spd_dim * spd_dim, feature_dim)
            #self.temperature = nn.Parameter(torch.ones(1))

        elif interaction_type == 'gating':
            # Gating mechanism
            self.spd_gate = nn.Sequential(
                nn.Linear(spd_dim * spd_dim, feature_dim),
                nn.Sigmoid()
            )
            self.feature_gate = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid()
            )
            self.spd_to_feature = nn.Linear(spd_dim * spd_dim, feature_dim)

        elif interaction_type == 'fusion':
            # Simple fusion
            self.spd_proj = nn.Linear(spd_dim * spd_dim, feature_dim)
            self.feature_proj = nn.Linear(feature_dim, feature_dim)
            self.fusion = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )

        # elif interaction_type == 'cross_attention':
        #     # Cross-attention between SPD and features
        #     self.spd_proj = nn.Linear(spd_dim * spd_dim, feature_dim)
        #     self.query = nn.Linear(feature_dim, feature_dim)
        #     self.key = nn.Linear(feature_dim, feature_dim)
        #     self.value = nn.Linear(feature_dim, feature_dim)
        #     self.scale = feature_dim ** -0.5

        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")

        logger.info(f"SPDFeatureInteraction: type={interaction_type}, spd_dim={spd_dim}, feat_dim={feature_dim}")

    def forward(self, spd_mats, features):
        """
        Args:
            spd_mats: [N, d, d] SPD matrices
            features: [N, feature_dim] node features

        Returns:
            updated_features: [N, feature_dim]
            updated_spd: [N, d, d] (optional, depends on interaction type)
        """
        N = features.shape[0]

        # Flatten SPD matrices to vectors (tangent space)
        spd_tangent = SPDManifold.logmap_id(spd_mats)  # [N, d, d]
        spd_vec = spd_tangent.reshape(N, -1)  # [N, d*d]

        if self.interaction_type == 'attention':
            # Compute attention scores
            spd_attn = self.spd_to_attn(spd_vec)  # [N, feature_dim]
            feat_attn = self.feature_to_attn(features)  # [N, feature_dim]

            combined = torch.cat([spd_attn, feat_attn], dim=-1)  # [N, 2*feature_dim]

            attn_scores = torch.softmax(self.attn_combine(combined), dim=0)  # [N, 1]

            # Update features with SPD information
            spd_features = self.spd_to_feature(spd_vec)  # [N, feature_dim]
            updated_features = features + attn_scores * spd_features

            return updated_features, spd_mats

        elif self.interaction_type == 'gating':
            # Compute gates
            spd_gate = self.spd_gate(spd_vec)  # [N, feature_dim]
            feat_gate = self.feature_gate(features)  # [N, feature_dim]

            # Update features
            spd_features = self.spd_to_feature(spd_vec)
            updated_features = feat_gate * features + spd_gate * spd_features

            return updated_features, spd_mats

        elif self.interaction_type == 'fusion':
            # Project and fuse
            spd_proj = self.spd_proj(spd_vec)  # [N, feature_dim]
            feat_proj = self.feature_proj(features)  # [N, feature_dim]

            fused = torch.cat([spd_proj, feat_proj], dim=-1)  # [N, 2*feature_dim]
            updated_features = self.fusion(fused) + features  # Residual connection

            return updated_features, spd_mats

        # elif self.interaction_type == 'cross_attention':
        #     # Cross-attention mechanism
        #     spd_emb = self.spd_proj(spd_vec)  # [N, feature_dim]

        #     Q = self.query(features)  # Features query SPD
        #     K = self.key(spd_emb)
        #     V = self.value(spd_emb)

        #     # WARNING: This creates [N, N] attention matrix across ALL nodes in batch!
        #     # This causes different graphs to attend to each other, which is incorrect.
        #     # Proper fix requires batch_indices to mask attention within each graph.
        #     attn = torch.softmax(Q @ K.T * self.scale, dim=-1)  # [N, N]
        #     attended = attn @ V  # [N, feature_dim]

        #     updated_features = features + attended

        #     return updated_features, spd_mats


# ============= Modified Model =============
class SPDGNNModel(nn.Module):
    """
    SPD Sheaf Neural Network for graph-structured data
    Uses SPD manifold geometry for feature representation and message passing
    """
    def __init__(self, args, num_nodes, num_edges):
        super().__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.interaction_type = args.interaction_type

        self.sheaf = args.sheaf
        self.transforms = nn.ModuleList([
                SPDTransform.get(args.transform, input_dims=3)
                for _ in range(args.num_layers)
            ])
        if self.sheaf:
            logger.info("Using SPD Sheaf layers")

            self.sheaf_learners = nn.ModuleList()
            self.sheaf_act = args.sheaf_act


            learner_map = {
            'local_concat': LocalConcatSheafLearner,
            'quadratic': QuadraticFormSheafLearner,
            'gated_local_concat': GatedLocalConcatSheafLearner}

            SheafLearnerClass = learner_map[args.sheaf_learner_type]

            for _ in range(args.num_layers):
                self.sheaf_learners.append(
                    SheafLearnerClass(args.feat_dim, out_shape=(6,), sheaf_act=self.sheaf_act)
                    ) #args.feat_dim

            self.orthogonal_map = args.orthogonal_map
            self.orth = Orthogonal(d=3, orthogonal_map=self.orthogonal_map)

        self.nonlinear = SPDNonLinearity.get(args.nonlinear)
        self.feat_dim = args.feat_dim
        self.spd_to_feature_interactions = nn.ModuleList([
            SPDFeatureInteraction(
                spd_dim=3,
                feature_dim=self.feat_dim,
                interaction_type=self.interaction_type
            )
            for _ in range(args.num_layers)
        ])

        self.dropout = nn.Dropout(args.dropout)

        self.num_layers = args.num_layers
        self.feat_pooling = args.feat_pooling
        self.lin_in = nn.Linear(args.num_node_features, self.feat_dim, bias=True)
        self.feature_layers = nn.ModuleList([
                    SAGEConv(self.feat_dim, self.feat_dim, 'mean') for _ in range(args.num_layers)
                ])
        self.leaky_relu = nn.LeakyReLU()

        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

        d = args.hidden_dims

        self.register_buffer('triu_rows', torch.triu_indices(d, d)[0])
        self.register_buffer('triu_cols', torch.triu_indices(d, d)[1])

        self.epsilon = float(args.epsilon)
        self.coord_method = args.coord_method

        self.use_mlp_head = bool(args.use_mlp_head)
        if self.use_mlp_head:
            # Dimensions of pooled SPD streams
            d1, d2 = 3, 16

            rows1, cols1 = torch.triu_indices(d1, d1)
            rows2, cols2 = torch.triu_indices(d2, d2)
            self.register_buffer('tri_rows1', rows1)
            self.register_buffer('tri_cols1', cols1)
            self.register_buffer('tri_rows2', rows2)
            self.register_buffer('tri_cols2', cols2)

            # Triangular vector dims for each SPD stream
            t1 = d1 * (d1 + 1) // 2  # = 6
            t2 = self.feat_dim
            self.t1, self.t2 = t1, t2

            # Head type: 'fbp' (default) or 'attention'
            self.mlp_head_type = args.mlp_head_type

            if self.mlp_head_type == 'fbp':
                # Factorized Bilinear Pooling (FBP)
                self.fbp_r = int(args.fbp_rank)
                self.fbp_U = nn.Linear(t1, self.fbp_r, bias=False)
                self.fbp_V = nn.Linear(t2, self.fbp_r, bias=False)

                fused_in_dim = t1 + t2 + self.fbp_r
                self.bn = nn.BatchNorm1d(fused_in_dim)

                hidden = int(args.mlp_hidden_dim)
                drop = float(args.mlp_dropout)
                self.mlp_head = nn.Sequential(
                    nn.Linear(fused_in_dim, hidden),
                    nn.ELU(),
                    nn.Dropout(drop),
                    nn.Linear(hidden, args.num_classes),
                )

                logger.info(
                    f"MLP head (FBP) enabled: rank={self.fbp_r}, fused_dim={fused_in_dim}, "
                    f"hidden={hidden}, dropout={drop}"
                )
            elif self.mlp_head_type == 'cross_attention':
                # Cross-Attention Fusion Head
                # Features query geometry (x1=geometry, x2=features)
                ca_d_model = int(args.ca_d_model)
                ca_num_heads = int(args.ca_num_heads)
                drop = float(args.mlp_dropout)
                hidden = int(args.mlp_hidden_dim)

                # Ensure d_model is divisible by num_heads
                if ca_d_model % ca_num_heads != 0:
                    logger.warning(
                        f"ca_d_model ({ca_d_model}) not divisible by ca_num_heads ({ca_num_heads}); "
                        f"adjusting to ca_num_heads=1"
                    )
                    ca_num_heads = 1

                self.ca_d_model = ca_d_model
                self.ca_num_heads = ca_num_heads

                # Project geometry (x1) and features (x2) to d_model
                self.ca_proj_geo = nn.Linear(t1, ca_d_model)  # 6 -> d_model
                self.ca_proj_feat = nn.Linear(t2, ca_d_model)  # feat_dim -> d_model

                # Multi-head cross-attention: features query geometry
                self.ca_multihead_attn = nn.MultiheadAttention(
                    embed_dim=ca_d_model,
                    num_heads=ca_num_heads,
                    dropout=drop,
                    batch_first=True
                )

                # Layer norm after attention
                self.ca_norm1 = nn.LayerNorm(ca_d_model)
                self.ca_norm2 = nn.LayerNorm(ca_d_model)

                # Feed-forward after attention
                self.ca_ffn = nn.Sequential(
                    nn.Linear(ca_d_model, ca_d_model * 2),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(ca_d_model * 2, ca_d_model),
                )

                # Final classifier: concatenate attended features + original features
                fused_in_dim = ca_d_model + t2
                self.ca_bn = nn.BatchNorm1d(fused_in_dim)
                self.ca_head = nn.Sequential(
                    nn.Linear(fused_in_dim, hidden),
                    nn.ELU(),
                    nn.Dropout(drop),
                    nn.Linear(hidden, args.num_classes),
                )

                logger.info(
                    f"MLP head (CrossAttention) enabled: d_model={ca_d_model}, heads={ca_num_heads}, "
                    f"fused_dim={fused_in_dim}, hidden={hidden}, dropout={drop}"
                )
            else:
                raise ValueError(f"Unknown mlp_head_type: {self.mlp_head_type}")


        # === NEW: SPD-specific pooling ===
        pooling_type = args.coor_pooling
        pooling_power = args.coor_pooling_power  # θ=0.5 recommended

        self.coor_spd_pooling = SPDPooling(
            pooling_type=pooling_type,
            power=pooling_power,
            eps=self.epsilon
        )

        # self.feature_spd_pooling = SPDPooling(
        #     pooling_type=pooling_type,
        #     power=pooling_power,
        #     eps=self.epsilon
        # )

        init_msg = (
            f"SPDGNNModel init: sheaf={self.sheaf}\n"
            f"layers={args.num_layers} hidden={args.hidden_dims}\n"
            f"nonlin={args.nonlinear} transform={args.transform}\n"
            f"dropout={float(args.dropout):.3f}\n"
            f"coor_pooling={pooling_type}(power={float(pooling_power):.3f})\n"
            f"feat_pooling={self.feat_pooling}\n"
            f"classifier={args.classifier} mlp_head={self.use_mlp_head}"
        )
        logger.info(init_msg)

        # Additional details at INFO level
        logger.info(
            f"in_dim={args.num_node_features} -> hidden={args.hidden_dims};\n"
            f"coord_method={self.coord_method} epsilon={self.epsilon}"
        )


    def _construct_coordinate_spd(self, pos, features):
        """
        Construct SPD matrices from node coordinates/features
        (unchanged from original)
        """
        device = features.device

        if self.coord_method == 'outer_product':
            pos_norm = torch.norm(pos, dim=-1, keepdim=True)  # [N, 1]
            pos_normalized = pos / (pos_norm + 1e-8)  # [N, 3]，norm=1
            coord_spd = torch.bmm(pos_normalized.unsqueeze(2), pos_normalized.unsqueeze(1))
            coord_spd = coord_spd + self.epsilon * torch.eye(3, device=device).unsqueeze(0)
        return coord_spd

    def generate_equivariant_orthogonal_matrix(self, pos: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """
        Generate a 3x3 orthogonal matrix Q for each node, satisfying global rotation equivariance: Q' = R Q.
        Construct local frame using node coordinates and neighbor edge directions.

        Args:
            pos: [N, 3] node coordinates (already centered)
            src, dst: [E] edge endpoint indices

        Returns:
            Q: [N, 3, 3] node local orthogonal frame
        """
        # Aggregate neighbor edge directions (add outgoing, subtract incoming, direction is rotation equivariant)
        e = pos[dst] - pos[src]  # [E, 3]
        agg = torch.zeros_like(pos)
        agg.scatter_add_(0, src.unsqueeze(-1).expand_as(e), e)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(e), -e)

        # v1: node position direction
        v1 = pos / torch.clamp(torch.linalg.norm(pos, dim=-1, keepdim=True), min=1e-6)  # [N,3]
        # v2_raw: aggregated neighbor edge direction
        v2_raw = agg / torch.clamp(torch.linalg.norm(agg, dim=-1, keepdim=True), min=1e-6)

        # v3: v1 x v2_raw, if nearly parallel, use the standard axis least correlated with v1
        v3 = torch.cross(v1, v2_raw, dim=-1)
        v3_norm = torch.linalg.norm(v3, dim=-1, keepdim=True)

        # Degenerate case handling: select the unit axis least aligned with v1 as auxiliary vector
        min_idx = torch.argmin(torch.abs(v1), dim=1)  # [N]
        fallback = torch.zeros_like(pos)
        fallback[torch.arange(pos.size(0)), min_idx] = 1.0
        v3_fallback = torch.cross(v1, fallback, dim=-1)

        v3 = torch.where(v3_norm > 1e-6, v3, v3_fallback)
        v3 = v3 / torch.clamp(torch.linalg.norm(v3, dim=-1, keepdim=True), min=1e-6)

        # v2: orthogonal to v1 and v3
        v2 = torch.cross(v3, v1, dim=-1)

        Q = torch.stack([v1, v2, v3], dim=-1)  # [N, 3, 3]
        return Q


    def forward(self, g, features):
        """
        Dual-path forward: coor + node features
        """
        device = features.device

        # ========== Initialize both paths ==========
        # SPD path: Coordinate-based
        #pos = g.ndata['coor']
        pos = g.ndata['coor'].double()
        batch = g.batch_num_nodes()       # [num_graphs], number of nodes per graph
        graph_idx = torch.repeat_interleave(torch.arange(len(batch), device=device), batch)
        graph_sum = torch.zeros((len(batch), pos.size(1)), device=device)
        graph_sum.index_add_(0, graph_idx, pos)           # Sum node coordinates for each graph
        graph_mean = graph_sum / batch.unsqueeze(1)       # Mean for each graph
        pos = pos - graph_mean[graph_idx]
        src, dst = g.edges()

        M = self.generate_equivariant_orthogonal_matrix(pos, src, dst)


        features = features.double()

        coord_spd = self._construct_coordinate_spd(pos, features)
        coord_spd = lalg.sym(self.dropout(coord_spd))
        coord_spd = SPDManifold.expmap_id(coord_spd)

        # Feature path: Initial projection
        h = self.lin_in(features)
        # ========== Layer-wise processing with interaction ==========
        for layer_idx in range(self.num_layers):
            # ===== Step 1: Process SPD path =====
            
            if layer_idx == 0:
                coord_spd_new = self.transforms[layer_idx](coord_spd, M)
            else:
                coord_spd_new = self.transforms[layer_idx](coord_spd)

            if self.sheaf:

                maps = self.sheaf_learners[layer_idx](h, src, dst)
                maps = self.orth(maps)  # [2E, 3, 3]


                delta_U = spd_sheaf_laplacian(
                        g, coord_spd_new, maps, device, self.epsilon
                    )

                if layer_idx == 0:
                    delta_U = SPDManifold.expmap_id(SPDManifold.logmap_id(coord_spd_new) + delta_U)
                else:
                    delta_U = SPDManifold.expmap_id(SPDManifold.logmap_id(coord_spd) + delta_U)


                coord_spd = self.nonlinear(delta_U)
            else:
                coord_spd = self.nonlinear(coord_spd_new)

            # ===== Step 2: Process feature path =====
            h = self.feature_layers[layer_idx](g, h)
            h = self.leaky_relu(h)
            #h = self.layer_norms[layer_idx](h)

            # ===== Step 3: Interaction =====
            # SPD → Feature interaction
            h, coord_spd = self.spd_to_feature_interactions[layer_idx](
                coord_spd, h
            )

            # ===== Step 4: Dropout (except last layer) =====
            if layer_idx < self.num_layers - 1:
                h = self.dropout(h)

                mat_feats_tan = SPDManifold.logmap_id(coord_spd)
                mat_feats_tan = self.dropout(mat_feats_tan)
                coord_spd = SPDManifold.expmap_id(mat_feats_tan)

        # Pooling for coordinate path
        batch_indices = torch.cat([
            torch.full((n,), i, dtype=torch.long, device=device)
            for i, n in enumerate(g.batch_num_nodes())
        ])
        graph_feats_coor = self.coor_spd_pooling(coord_spd, batch_indices)

        # Feature pooling
        if self.feat_pooling == 'avg':
            y = self.avgpool(g, h)
        elif self.feat_pooling == 'max':
            y = self.maxpool(g, h)
        elif self.feat_pooling == 'sum':
            y = self.sumpool(g, h)
        else:
            y = self.avgpool(g, h)


        # ========== Classification with dual features ==========
        if self.use_mlp_head:
            # Pooled SPD → tangent → upper-tri vectors for both streams
            spd_small = SPDManifold.logmap_id(graph_feats_coor)

            x1 = spd_small[:, self.tri_rows1, self.tri_cols1]  # [B, t1]
            x2 = y
            if self.mlp_head_type == 'fbp':
                # Factorized bilinear pooling cross term
                cross = self.fbp_U(x1) * self.fbp_V(x2)  # [B, r]
                x = torch.cat([x1, x2, cross], dim=-1)   # [B, t1+t2+r]
                x = self.bn(x)
                predictions = self.mlp_head(x)
            elif self.mlp_head_type == 'cross_attention':
                # Cross-Attention: features query geometry
                # Project to d_model
                geo_emb = self.ca_proj_geo(x1)    # [B, d_model]
                feat_emb = self.ca_proj_feat(x2)  # [B, d_model]

                # Reshape for multi-head attention: [B, 1, d_model] (single token per stream)
                geo_emb = geo_emb.unsqueeze(1)    # [B, 1, d_model] - Key/Value
                feat_emb = feat_emb.unsqueeze(1)  # [B, 1, d_model] - Query

                # Cross-attention: features attend to geometry
                # Q=features, K=V=geometry
                attn_out, _ = self.ca_multihead_attn(
                    query=feat_emb,
                    key=geo_emb,
                    value=geo_emb
                )  # [B, 1, d_model]

                # Residual + LayerNorm
                attn_out = self.ca_norm1(feat_emb + attn_out)

                # Feed-forward + Residual + LayerNorm
                ffn_out = self.ca_ffn(attn_out)
                attn_out = self.ca_norm2(attn_out + ffn_out)

                # Remove sequence dimension and concatenate with original features
                attn_out = attn_out.squeeze(1)  # [B, d_model]
                fused = torch.cat([attn_out, x2], dim=-1)  # [B, d_model + feat_dim]
                fused = self.ca_bn(fused)
                predictions = self.ca_head(fused)
            else:
                raise ValueError(f"Unknown mlp_head_type: {self.mlp_head_type}")

        outputs = torch.sigmoid(predictions)
        return outputs


def spd_sheaf_laplacian(
    g,
    coord_spd,
    maps,
    device,
    epsilon=1e-6
):
    """
    Sheaf convolution on SPD: Delta_U = δ^T δ(U)

    Two usage modes are supported:
    1) Restriction-map mode (preferred):
       - maps is a tuple/list (A_src_to_edge, A_dst_to_edge) or a dict with
         keys ('src_to_edge','dst_to_edge') / ('12','21'). Each has shape
         [E, d, d] (or broadcastable to that) acting by congruence A X A^T.
       - Implements δ(U)_e = exp(log(A_i X_i A_i^T) - log(A_j X_j A_j^T)), and
         δ^T aggregates back in the log-domain with pullbacks A^T (⋅) A and
         incidence signs (+ for src, − for dst).


    Returns
    -------
    delta_U : [N, d, d]
        Symmetric matrices in the tangent at identity (caller may expmap).
    """

    num_nodes = g.num_nodes()
    d = coord_spd.shape[-1]
    src, dst = g.edges()

    E = src.shape[0]

    # --- maps normalization / interpretation ---
    # Accept formats:
    # 1) (A_src_to_e, A_dst_to_e) tuple/list -> keep
    # 2) dict with keys ('src_to_edge','dst_to_edge') or ('12','21') -> extract
    # 3) tensor with shape [E, d, d] -> treat as A_src_to_e (and optionally set A_dst_to_e = A_src_to_e)
    # 4) tensor with shape [2*E, d, d] -> split into two halves: first E -> A_src_to_e, last E -> A_dst_to_e
    A_src_to_e = None
    A_dst_to_e = None


    A_src_to_e = maps[0::2]  # even indices

    A_dst_to_e = maps[1::2]  # odd indices

    src = src[0::2]
    dst = dst[0::2]
    X_src = coord_spd[src]
    X_dst = coord_spd[dst]

    # Node SPD to edge SPD by congruence
    X_src_e = A_src_to_e @ X_src @ A_src_to_e.transpose(-1, -2)
    X_dst_e = A_dst_to_e @ X_dst @ A_dst_to_e.transpose(-1, -2)


    # δ on edges in log-domain
    t_src = SPDManifold.logmap_id(X_src_e)
    t_dst = SPDManifold.logmap_id(X_dst_e)
    T_e = t_src - t_dst  # [E, d, d]
    T_e = 0.5 * (T_e + T_e.transpose(-2, -1))

    pb_src = A_src_to_e.transpose(-1, -2) @ T_e @ A_src_to_e  # [E, d, d]
    pb_dst = A_dst_to_e.transpose(-1, -2) @ -T_e @ A_dst_to_e  # [E, d, d]

    delta_log = torch.zeros(num_nodes, d, d, device=device)
    delta_log.scatter_add_(0, src.view(-1,1,1).expand(-1,d,d),  pb_src)  # +  i→j

    delta_log.scatter_add_(0, dst.view(-1,1,1).expand(-1,d,d),  pb_dst)  # −  j→i

    N, d, _ = delta_log.shape
    eigvals, eigvecs = torch.linalg.eigh(delta_log)
    eigvals = torch.clamp(eigvals, min=epsilon)
    λ_min = eigvals.min(dim=1, keepdim=True).values
    λ_max = eigvals.max(dim=1, keepdim=True).values
    range_val = λ_max - λ_min

    min_range = 0.01
    needs_norm = (range_val > min_range).squeeze(-1)  # [N]

    a, b = (-1, 1)
    eigvals_normalized = eigvals.clone()

    idx = needs_norm
    eigvals_normalized[idx] = a + (eigvals[idx] - λ_min[idx]) / (range_val[idx] + epsilon) * (b - a)

    delta_log = eigvecs @ torch.diag_embed(eigvals_normalized) @ eigvecs.transpose(-1, -2)

    return delta_log
