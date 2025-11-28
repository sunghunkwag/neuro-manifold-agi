"""

Riemannian Geometry for Neuro-Manifold Automata

This module implements the geometric core of the NMA.

It provides a numerically stable Riemannian manifold using Cholesky parameterization

for the metric tensor and implements Geodesic Flow logic.

Changes:

- Replaced `matrix_exp` with Cholesky Decomposition (L @ L.T) for absolute stability.

- Added `GeodesicFlow` for transporting information along curved paths.

- Optimized tensor operations for batch processing.

"""

import torch

import torch.nn as nn

import torch.nn.functional as F

class RiemannianManifold(nn.Module):

"""

A dynamic manifold represented by a learnable metric field.

Uses Cholesky factors for robust Positive Definite metric tensors.

"""

def __init__(self, dim: int, hidden_dim: int = 64):

super().__init__()

self.dim = dim

# Predicts the Lower Triangular matrix L such that G = L @ L.T

# Output size: dim * (dim + 1) / 2 would be efficient, but full matrix is easier to batch

self.cholesky_field = nn.Sequential(

nn.Linear(dim, hidden_dim),

nn.SiLU(),

nn.Linear(hidden_dim, dim * dim)



# Bias towards Identity matrix (Euclidean space)

# This acts as a regularizer, ensuring the space starts flat

self.I_bias = nn.Parameter(torch.eye(dim).view(-1), requires_grad=False)

def get_metric_tensor(self, x: torch.Tensor) -> torch.Tensor:

"""

Computes the Riemannian Metric Tensor G(x) = L(x) @ L(x)^T

Guaranteed to be Symmetric Positive Definite (SPD).

"""

B, N, _ = x.shape

# Predict raw factors

raw = self.cholesky_field(x) # (B, N, D*D)

# Add Identity bias to start near Euclidean

L_flat = raw + self.I_bias

L = L_flat.view(B, N, self.dim, self.dim)

# Force lower triangular (mask upper)

mask = torch.tril(torch.ones(self.dim, self.dim, device=x.device))

L = L * mask

# Ensure positive diagonal for uniqueness and conditioning

# Softplus on diagonal elements

diag_mask = torch.eye(self.dim, device=x.device)

L = L * (1 - diag_mask) + F.softplus(L) * diag_mask

# G = L @ L.T

G = torch.matmul(L, L.transpose(-1, -2))

return G

def compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

"""

Computes the Mahalanobis distance induced by the local metric G.

d^2 = (x2 - x1)^T * G * (x2 - x1)

"""

dx = x2 - x1

mid = (x1 + x2) / 2

G = self.get_metric_tensor(mid) # (B, N, D, D)

# Efficient computation: dx^T G dx

# (B, N, 1, D) @ (B, N, D, D) @ (B, N, D, 1) -> (B, N, 1, 1)

dx_un = dx.unsqueeze(-1)

G_dx = torch.matmul(G, dx_un)

dist_sq = torch.matmul(dx_un.transpose(-1, -2), G_dx).squeeze(-1).squeeze(-1)

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

Euler integration of the Geodesic Equation:

d^2x/dt^2 + Gamma * (dx/dt) * (dx/dt) = 0

Ideally, we solve this. For efficiency, we approximate by

following the metric tensor's principle axes (Gradient Flow).

Here, we simply push 'x' in direction 'v' modified by G^{-1}.

Intuitively: Moving is harder in 'dense' (high curvature) areas.

"""

curr_x = x

curr_v = v

for _ in range(steps):

G = self.manifold.get_metric_tensor(curr_x)

# G is SPD, so we can invert it (or solve).

# Acceleration ~ - G^{-1} (Christoffel symbols proxy)

# We simplify: velocity is scaled by G^{-1} (Gradient descent on Manifold)

# Solve G * v_new = v_old -> v_new = G^{-1} * v_old

# This is 'Natural Gradient' flow.

try:

curr_v = torch.linalg.solve(G, curr_v.unsqueeze(-1)).squeeze(-1)

except RuntimeError:

# Fallback for singular G (shouldn't happen due to Cholesky, but safe guarding)

curr_v = curr_v

curr_x = curr_x + curr_v * self.step_size

return curr_x