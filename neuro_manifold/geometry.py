"""
Riemannian Geometry for Neuro-Manifold Automata

This module implements the geometric core of the NMA.
It provides a numerically stable Riemannian manifold using Cholesky parameterization
for the metric tensor and implements Geodesic Flow logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RiemannianManifold(nn.Module):
    """
    A dynamic manifold represented by a learnable metric field.
    Uses Cholesky factors for robust Positive Definite metric tensors.
    """
    def __init__(self, dim: int, hidden_dim: int = 32):
        super().__init__()
        self.dim = dim

        # Predicts the Lower Triangular matrix L such that G = L @ L.T
        self.cholesky_field = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * dim)
        )

        # Bias towards Identity matrix (Euclidean space)
        # We assume flat view for addition, but register as buffer for device management
        self.register_buffer("I_bias", torch.eye(dim).view(-1))

        # Mask for lower triangular matrix
        self.register_buffer("tril_mask", torch.tril(torch.ones(dim, dim)))
        self.register_buffer("diag_mask", torch.eye(dim))

    def get_cholesky_factor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Cholesky factor L(x) such that G(x) = L(x) @ L(x)^T.
        """
        # x shape: (B, N, D) or (B, N, M, D) etc.
        # We process last dim D.
        # Flatten arbitrary leading dims for processing
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)

        # Predict raw factors
        raw = self.cholesky_field(x_flat) # (Batch*N, D*D)

        # Add Identity bias
        L_flat = raw + self.I_bias
        L = L_flat.view(-1, self.dim, self.dim)

        # Force lower triangular
        L = L * self.tril_mask

        # Ensure positive diagonal
        L = L * (1 - self.diag_mask) + F.softplus(L) * self.diag_mask

        # Reshape back: (B, N, D, D)
        return L.view(*original_shape[:-1], self.dim, self.dim)

    def get_metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Riemannian Metric Tensor G(x) = L(x) @ L(x)^T
        Guaranteed to be Symmetric Positive Definite (SPD).
        """
        L = self.get_cholesky_factor(x)
        # G = L @ L.T
        G = torch.matmul(L, L.transpose(-1, -2))
        return G

    def compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mahalanobis distance induced by the local metric G.
        Supports broadcasting.
        x1: (B, N, 1, D) (Query)
        x2: (B, 1, M, D) (Key)
        Output: (B, N, M)
        """
        # Broadcasting difference
        # x1: (..., N, 1, D)
        # x2: (..., 1, M, D)
        dx = x2 - x1 # (..., N, M, D)

        # Compute metric at midpoint
        mid = (x1 + x2) / 2 # (..., N, M, D)
        G = self.get_metric_tensor(mid) # (..., N, M, D, D)

        # Distance squared: dx^T G dx
        dx_un = dx.unsqueeze(-1) # (..., N, M, D, 1)
        G_dx = torch.matmul(G, dx_un) # (..., N, M, D, 1)
        dist_sq = torch.matmul(dx_un.transpose(-1, -2), G_dx).squeeze(-1).squeeze(-1) # (..., N, M)

        return torch.sqrt(torch.clamp(dist_sq, min=1e-6))

class GeodesicFlow(nn.Module):
    """
    Simulates the flow of a particle (information) along a geodesic.
    Used for long-range communication between cells without direct edges.
    """
    def __init__(self, manifold: RiemannianManifold, step_size: float = 0.1):
        super().__init__()
        self.manifold = manifold
        self.step_size = step_size

    def forward(self, x: torch.Tensor, v: torch.Tensor, steps: int = 3) -> torch.Tensor:
        """
        Euler integration of the Geodesic Equation approx (Natural Gradient Flow).
        v_{new} = G^{-1} v_{old}
        """
        curr_x = x
        curr_v = v

        for _ in range(steps):
            L = self.manifold.get_cholesky_factor(curr_x)
            # We want to solve G * v_new = v_old
            # G = L @ L.T
            # L @ L.T @ v_new = v_old
            # torch.cholesky_solve computes A^{-1} b given chol(A).

            # Prepare v for solve: (B, N, D, 1)
            v_in = curr_v.unsqueeze(-1)

            try:
                # cholesky_solve(b, L) solves A x = b
                v_out = torch.cholesky_solve(v_in, L)
                curr_v = v_out.squeeze(-1)
            except RuntimeError:
                pass

            curr_x = curr_x + curr_v * self.step_size

        return curr_x
