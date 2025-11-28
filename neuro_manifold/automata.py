"""
Neural Cellular Automata on Manifold

This module defines the update rules for the cells.
D_state = Update(State, Neighbors)

It uses Graph Neural Networks (GNN) principles adapted for the manifold.
"""

import torch
import torch.nn as nn
from .geometry import RiemannianManifold, GeodesicFlow

class ManifoldAutomata(nn.Module):
    def __init__(self, num_cells: int, state_dim: int, geometry: RiemannianManifold):
        super().__init__()
        self.num_cells = num_cells
        self.state_dim = state_dim
        self.geometry = geometry

        # Communication (Geodesic Message Passing)
        self.msg_fn = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim)
        )

        # Update Rule (Cellular Dynamics)
        self.update_fn = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim * 2),
            nn.LayerNorm(state_dim * 2),
            nn.SiLU(),
            nn.Linear(state_dim * 2, state_dim),
            nn.Tanh() # Bounded updates
        )

        # Fast Weights (Hebbian Plasticity)
        self.plasticity_rate = 0.01

        # Lock 3: Gated Learning (Circuit Breaker)
        self.learning_rate_gate = 1.0 # 1.0 = Open, 0.0 = Locked

    def forward(self, x: torch.Tensor, hebbian_trace: torch.Tensor = None, steps: int = 1):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape

        # Initialize Hebbian Trace if None
        if hebbian_trace is None:
            hebbian_trace = torch.zeros(B, N, N, device=x.device)

        curr_x = x

        for _ in range(steps):
            # 1. Perception (Message Passing)
            # We use attention-like mechanism but weighted by Manifold Distance
            # For efficiency in this demo, we use all-to-all with distance mask

            # Compute Pairwise Distances on Manifold
            # dist_matrix: (B, N, N)
            # We act as if cells are at their feature coordinates (Concept Space)
            x_query = curr_x.unsqueeze(2) # (B, N, 1, D)
            x_key = curr_x.unsqueeze(1)   # (B, 1, N, D)

            # Use Geometry to compute distance
            dists = self.geometry.compute_distance(x_query, x_key)

            # Neighborhood kernel (Gaussian)
            adjacency = torch.exp(-dists) # (B, N, N)

            # Apply Hebbian trace (Short-term memory of connection)
            effective_adj = adjacency + 0.1 * hebbian_trace

            # Aggregate messages
            # msg = sum(adj * Value(neighbor))
            # Simple average of neighbors
            # (B, N, N) @ (B, N, D) -> (B, N, D)
            neighborhood_agg = torch.bmm(effective_adj, curr_x) / (N + 1e-5)

            # 2. Update
            # Input to update rule: [Self, Neighborhood]
            combined = torch.cat([curr_x, neighborhood_agg], dim=-1)
            delta = self.update_fn(combined)

            curr_x = curr_x + delta

            # 3. Plasticity Update (Hebbian)
            # If two cells fire together (high cosine sim), strengthen connection
            # Normalized correlation
            norm_x = torch.nn.functional.normalize(curr_x, p=2, dim=-1)
            correlation = torch.bmm(norm_x, norm_x.transpose(1, 2))

            # Lock 3: Gated Learning
            # Update trace: trace = decaying_trace + rate * correlation
            # We apply the gate to the update term.
            update_term = self.plasticity_rate * correlation * self.learning_rate_gate
            hebbian_trace = 0.95 * hebbian_trace + update_term

        return curr_x, hebbian_trace
