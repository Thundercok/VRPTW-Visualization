"""
GNN edge predictor for VRPTW.

Message passing runs over a k-nearest-neighbour edge set rather than the complete
graph. The dense formulation materialised ``(B, N, N, hidden)`` edge tensors —
about 0.26 GB for the hidden state plus 0.77 GB for the per-layer concatenation at
n=1000, times three layers — which is what put the 600/800/1000-customer shards out
of reach. With K=16 that becomes ``(B, N, K, hidden)``, roughly 60x smaller at
n=1000.

The *output* stays a dense ``(N, N)`` probability matrix: every consumer indexes it
as ``heatmap[i, j]`` for arbitrary pairs (including depot edges), and the pruning
kernels treat a low probability as "skip this position", so leaving non-kNN pairs
unscored would silently forbid every depot-adjacent insertion. The dense matrix is
produced by a bilinear head over the final node embeddings, which costs one
``N x N`` matmul and no ``N x N x hidden`` intermediate.
"""

import torch
import torch.nn as nn

from .core import Inst, Plan

# Neighbours per node for message passing. Slot 0 is reserved for the depot so
# depot connectivity is always modelled.
DEFAULT_K = 16


def build_knn_edges(inst: Inst, k: int = DEFAULT_K) -> torch.Tensor:
    """Neighbour indices of shape ``(N, k)`` for message passing.

    Unlike ``Inst.neighbors_k`` this includes the depot (pinned at slot 0 for
    customers) and provides a row for the depot itself, so the graph stays
    connected through it.
    """
    n_nodes = inst.n + 1
    k = min(k, n_nodes - 1)
    dist = torch.tensor(inst.dist, dtype=torch.float32)
    masked = dist.clone()
    masked.fill_diagonal_(float("inf"))

    # Depot row: simply its k nearest customers.
    depot_nbrs = torch.topk(masked[0], k, largest=False).indices

    # Customer rows: depot pinned first, then the k-1 nearest other customers.
    cust = masked[1:].clone()
    cust[:, 0] = float("inf")  # depot handled separately
    cust_nbrs = torch.topk(cust, k - 1, largest=False).indices
    depot_col = torch.zeros((inst.n, 1), dtype=torch.long)
    cust_nbrs = torch.cat([depot_col, cust_nbrs], dim=1)

    return torch.cat([depot_nbrs.unsqueeze(0), cust_nbrs], dim=0)


class GNNLayer(nn.Module):
    """Joint node/edge update over a sparse neighbour set."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, h_nodes: torch.Tensor, h_edges: torch.Tensor, nbr_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # h_nodes: (B, N, H) | h_edges: (B, N, K, H) | nbr_idx: (B, N, K)
        B, N, H = h_nodes.shape
        K = nbr_idx.shape[-1]

        # Gather the embedding of each neighbour j of i by flat indexing — going
        # through an expanded (B, N, N, H) view would reintroduce the quadratic
        # tensor this layer exists to avoid.
        flat = h_nodes.reshape(B * N, H)
        offsets = (torch.arange(B, device=h_nodes.device) * N).view(B, 1, 1)
        h_nodes_j = flat[(nbr_idx + offsets).reshape(-1)].view(B, N, K, H)
        h_nodes_i = h_nodes.unsqueeze(2).expand(B, N, K, H)

        edge_in = torch.cat([h_edges, h_nodes_i, h_nodes_j], dim=-1)
        h_edges_new = h_edges + self.edge_mlp(edge_in)

        # Aggregate over the K neighbours rather than all N nodes.
        agg_msg = (h_edges_new + h_nodes_j).mean(dim=2)  # (B, N, H)

        node_in = torch.cat([h_nodes, agg_msg], dim=-1)
        h_nodes_new = h_nodes + self.node_mlp(node_in)

        return h_nodes_new, h_edges_new


class GNNEdgePredictor(nn.Module):
    """Predicts a dense edge-probability matrix from sparse message passing."""

    def __init__(self, node_dim: int = 6, edge_dim: int = 1, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        self.layers = nn.ModuleList([GNNLayer(hidden_dim) for _ in range(num_layers)])

        # Bilinear source/target head. Scoring pairs from node embeddings keeps the
        # output dense without ever building an (N, N, hidden) tensor.
        # Bilinear source/target head. Scoring pairs from node embeddings keeps the
        # output dense without ever building an (N, N, hidden) tensor.
        self.src_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dst_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Parameter(torch.zeros(1))

        # Contrastive projection head for Contrastive Graph RL (InfoNCE Loss)
        self.contrastive_dim = 32
        self.contrastive_src_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.contrastive_dim),
        )
        self.contrastive_dst_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.contrastive_dim),
        )

    def forward(self, x_nodes: torch.Tensor, x_edges: torch.Tensor, nbr_idx: torch.Tensor) -> torch.Tensor:
        # x_nodes: (B, N, node_dim) | x_edges: (B, N, K, edge_dim) | nbr_idx: (B, N, K)
        h_nodes = self.node_embed(x_nodes)
        h_edges = self.edge_embed(x_edges)

        for layer in self.layers:
            h_nodes, h_edges = layer(h_nodes, h_edges, nbr_idx)

        src = self.src_proj(h_nodes)  # (B, N, H)
        dst = self.dst_proj(h_nodes)  # (B, N, H)
        scale = self.hidden_dim**0.5
        logits = torch.matmul(src, dst.transpose(-2, -1)) / scale + self.edge_bias
        return logits  # (B, N, N)

    def get_contrastive_embeddings(
        self, x_nodes: torch.Tensor, x_edges: torch.Tensor, nbr_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns L2-normalized source and target node embeddings for InfoNCE loss."""
        h_nodes = self.node_embed(x_nodes)
        h_edges = self.edge_embed(x_edges)
        for layer in self.layers:
            h_nodes, h_edges = layer(h_nodes, h_edges, nbr_idx)

        z_src = nn.functional.normalize(self.contrastive_src_proj(h_nodes), p=2, dim=-1)
        z_dst = nn.functional.normalize(self.contrastive_dst_proj(h_nodes), p=2, dim=-1)
        return z_src, z_dst

    def compute_contrastive_loss(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
        tau: float = 0.1,
    ) -> torch.Tensor:
        """Computes InfoNCE Contrastive Loss between positive (elite) and negative (violating) edges."""
        # z_src, z_dst: (1, N, 32)
        # pos_edges: (M_pos, 2), neg_edges: (M_neg, 2)
        if pos_edges.shape[0] == 0:
            return torch.tensor(0.0, device=z_src.device, requires_grad=True)

        src_pos = z_src[0, pos_edges[:, 0]]  # (M_pos, 32)
        dst_pos = z_dst[0, pos_edges[:, 1]]  # (M_pos, 32)
        pos_sim = (src_pos * dst_pos).sum(dim=-1) / tau  # (M_pos,)

        if neg_edges.shape[0] > 0:
            src_neg = z_src[0, neg_edges[:, 0]]  # (M_neg, 32)
            dst_neg = z_dst[0, neg_edges[:, 1]]  # (M_neg, 32)
            neg_sim = (src_neg * dst_neg).sum(dim=-1) / tau  # (M_neg,)
            denom = torch.exp(pos_sim) + torch.exp(neg_sim).mean()
        else:
            denom = torch.exp(pos_sim)

        loss = -torch.log(torch.exp(pos_sim) / (denom + 1e-8)).mean()
        return loss


def get_gnn_features(inst: Inst, k: int = DEFAULT_K) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalised instance features.

    Returns ``(node_feats, edge_feats, nbr_idx)`` with batch dimension 1 and shapes
    ``(1, N, 6)``, ``(1, N, K, 1)`` and ``(1, N, K)``.
    """
    coords = inst.coords
    demands = inst.demands
    ready = inst.ready_times
    due = inst.due_times
    service = inst.service_times

    # Coordinate normalization to [0, 1]
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    coords_range = max_coords - min_coords
    coords_range[coords_range == 0] = 1.0
    norm_coords = (coords - min_coords) / coords_range

    norm_demands = demands / max(inst.capacity, 1.0)
    max_time = max(due[0], 1.0)  # depot due time

    node_feats = torch.zeros((inst.n + 1, 6), dtype=torch.float32)
    node_feats[:, 0:2] = torch.tensor(norm_coords, dtype=torch.float32)
    node_feats[:, 2] = torch.tensor(norm_demands, dtype=torch.float32)
    node_feats[:, 3] = torch.tensor(ready / max_time, dtype=torch.float32)
    node_feats[:, 4] = torch.tensor(due / max_time, dtype=torch.float32)
    node_feats[:, 5] = torch.tensor(service / max_time, dtype=torch.float32)

    # Sparse edge features: normalised distance to each of the k neighbours.
    nbr_idx = build_knn_edges(inst, k)
    max_dist = max(float(inst.dist.max()), 1.0)
    dist_t = torch.tensor(inst.dist, dtype=torch.float32) / max_dist
    edge_feats = torch.gather(dist_t, 1, nbr_idx).unsqueeze(-1)  # (N, K, 1)

    return node_feats.unsqueeze(0), edge_feats.unsqueeze(0), nbr_idx.unsqueeze(0)


def load_edge_predictor(model_path: str, device) -> "GNNEdgePredictor | None":
    """Load a trained edge predictor, or return None if the file is absent.

    Raises a pointed error on an architecture mismatch: checkpoints trained
    against the previous dense formulation carry an ``edge_predictor.*`` head that
    the sparse model replaces with ``src_proj``/``dst_proj``, so they cannot be
    reused and must be retrained.
    """
    import os

    if not os.path.exists(model_path):
        return None

    model = GNNEdgePredictor(node_dim=6, edge_dim=1, hidden_dim=64, num_layers=3).to(device)
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location=device)

    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        if any(k.startswith("edge_predictor.") for k in state_dict):
            raise RuntimeError(
                f"{model_path} was trained against the dense GNN architecture and is "
                "incompatible with the sparse-kNN edge predictor. Retrain it with "
                "`python -m vrptw.train_gnn`."
            ) from e
        raise
    return model


def plan_to_adj_matrix(plan: Plan) -> torch.Tensor:
    """
    Converts a plan (list of routes) into an adjacency matrix of shape (N+1, N+1).
    BKS/Elite solutions are encoded as targets.
    """
    n_nodes = plan.inst.n + 1
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    for r in plan.routes:
        if not r:
            continue
        # depot -> first node
        adj[0, r[0]] = 1.0
        # consecutive customer node sequences
        for i in range(len(r) - 1):
            adj[r[i], r[i + 1]] = 1.0
        # last node -> depot
        adj[r[-1], 0] = 1.0
    return adj


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) with multi-head attention over spatial-temporal customer nodes.
    Computes node embeddings weighted by dynamic spatial-temporal attention scores.
    """

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_nodes: torch.Tensor) -> torch.Tensor:
        # h_nodes: (B, N, hidden_dim)
        B, N, H = h_nodes.shape
        q = self.q_proj(h_nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Multi-head self-attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, N, H)
        return self.out_proj(out) + h_nodes


def get_gat_embeddings(inst: Inst, hidden_dim: int = 64, weights: dict | None = None) -> torch.Tensor:
    """
    Extracts 64-dimensional Graph Attention (GAT) spatial-temporal node embeddings for an instance.

    ``weights`` must supply trained ``state_dict``s under the keys
    ``"node_embedder"`` and ``"gat_layer"``. Without them there is nothing to
    run: this function used to build both modules inline, so the returned
    "embeddings" were freshly-random projections that differed on every call for
    the same ``inst``. Raising is deliberate — silently handing back that noise
    is how it ends up as somebody's feature vector.
    """
    if weights is None:
        raise NotImplementedError(
            "get_gat_embeddings requires trained weights: the GAT layer is "
            "roadmap scaffolding and no checkpoint is wired up yet. Pass "
            "weights={'node_embedder': state_dict, 'gat_layer': state_dict} "
            "once one exists."
        )

    node_feats, _edge_feats, _nbr_idx = get_gnn_features(inst)
    node_embedder = nn.Linear(6, hidden_dim)
    gat_layer = GraphAttentionLayer(hidden_dim=hidden_dim)
    node_embedder.load_state_dict(weights["node_embedder"])
    gat_layer.load_state_dict(weights["gat_layer"])
    node_embedder.eval()
    gat_layer.eval()

    with torch.no_grad():
        h = node_embedder(node_feats)
        gat_h = gat_layer(h)
    return gat_h.squeeze(0)  # (N+1, hidden_dim)
