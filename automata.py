"""

Neural Cellular Automata (NCA) with Hebbian Plasticity

The "Living Cells" of the architecture.

Each cell is a recurrent unit that interacts with neighbors.

Crucially, the connection strengths (Attention Weights) are NOT static.

They evolve via Hebbian Plasticity (Fire together, wire together).

Key Concepts:

- Fast Weights: Attention matrix updates dynamically during inference.

- DNA (Stable): Fixed parameters defining the update rule.

- Plasticity: Short-term adaptation to the current context.

"""

import torch

import torch.nn as nn

import torch.nn.functional as F

from typing import Tuple

from .geometry import RiemannianManifold

class NeuralCell(nn.Module):

"""

A single cell's metabolic rule.

"""

def __init__(self, state_dim: int, msg_dim: int = 16):

super().__init__()

self.state_dim = state_dim

# Perception: Process neighbor messages

self.perceive = nn.Linear(msg_dim, state_dim * 2)

# Adaptation: Update internal state based on perception

self.update_gate = nn.GRUCell(state_dim * 2, state_dim)

# Broadcasting: Generate message for neighbors

self.broadcast = nn.Linear(state_dim, msg_dim)

# Cell "DNA" - bias

self.dna = nn.Parameter(torch.randn(1, state_dim) * 0.1)

def forward(self, state: torch.Tensor, neighbor_agg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

# 1. Perception

perception = self.perceive(neighbor_agg)

# 2. State Update

new_state = self.update_gate(perception, state + self.dna)

# 3. Message Generation

message = self.broadcast(new_state)

return new_state, message

class ManifoldAutomata(nn.Module):

"""

The colony of cells with Plastic Connectivity.

"""

def __init__(self, num_cells: int, state_dim: int, geometry: RiemannianManifold):

super().__init__()

self.num_cells = num_cells

self.state_dim = state_dim

self.geometry = geometry

self.cell_rule = NeuralCell(state_dim, msg_dim=state_dim)

# Query/Key projections for Plastic Attention

self.W_q = nn.Linear(state_dim, state_dim)

self.W_k = nn.Linear(state_dim, state_dim)

self.W_v = nn.Linear(state_dim, state_dim)

self.norm = nn.LayerNorm(state_dim)

# Plasticity coefficient

self.eta = nn.Parameter(torch.tensor(0.01))

self.decay = nn.Parameter(torch.tensor(0.9))

def forward(self, cell_states: torch.Tensor, steps: int = 5) -> torch.Tensor:

B = cell_states.shape[0]

if cell_states is None:

# Initialize with geometry-aware positions

# In a full version, we'd sample from the manifold

cell_states = torch.zeros(B, self.num_cells, self.state_dim, device=self.geometry.I_bias.device)

current_states = cell_states

# Fast Weight Matrix (Hebbian Trace)

# Initialize as zero or based on metric distance

# For efficiency, we assume it starts empty for each episode

hebbian_trace = torch.zeros(B, self.num_cells, self.num_cells, device=current_states.device)

for t in range(steps):

# 1. Plastic Attention

Q = self.W_q(current_states) # (B, N, D)

K = self.W_k(current_states) # (B, N, D)

V = self.W_v(current_states) # (B, N, D)

# Base Attention (Fixed DNA)

scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.state_dim ** 0.5)

# Plastic Modulation

# Total Attention = Softmax(Base + eta * Trace)

modulated_scores = scores + self.eta * hebbian_trace

attn_weights = F.softmax(modulated_scores, dim=-1)

# Aggregate

neighbor_agg = torch.matmul(attn_weights, V)

neighbor_agg = self.norm(neighbor_agg)

# 2. Hebbian Update

# Fire together (High Q*K), Wire together

# Delta = OuterProduct(Q, K) - simply use current co-activation

# Simplified Oja's rule or accumulation

co_activation = torch.matmul(Q, K.transpose(-2, -1)).detach() # Detach to stop grad through plasticity rule itself (optional)

hebbian_trace = self.decay * hebbian_trace + (1 - self.decay) * co_activation

# 3. Cellular Update

flat_states = current_states.reshape(-1, self.state_dim)

flat_agg = neighbor_agg.reshape(-1, self.state_dim)

new_flat_states, _ = self.cell_rule(flat_states, flat_agg)

current_states = new_flat_states.reshape(B, self.num_cells, self.state_dim)

return current_states