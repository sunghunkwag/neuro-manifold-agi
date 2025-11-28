"""

Energy Dynamics & Equilibrium Solver

The "Mind" of the system.

The system does not just output a value; it settles into an energy minimum.

This implements Energy-Based Models (EBM) logic (Hopfield Network / Equilibrium Propagation style).

Key Concepts:

- Hamiltonian (H): Total energy of the system state.

- Equilibrium: The state where dH/ds = 0.

- Inference as Minimization: Thinking is relaxing to the ground state.

"""

import torch

import torch.nn as nn

import torch.nn.functional as F

class EnergyFunction(nn.Module):

"""

Defines the global energy landscape E(x).

"""

def __init__(self, state_dim: int):

super().__init__()

self.net = nn.Sequential(

nn.Linear(state_dim, state_dim * 2),

nn.SiLU(),

nn.Linear(state_dim * 2, 1)



def forward(self, x: torch.Tensor) -> torch.Tensor:

# x: (B, N, D)

# Sum energy of all cells

energies = self.net(x) # (B, N, 1)

return energies.sum(dim=1).squeeze(-1) # (B,)

class EquilibriumSolver(nn.Module):

"""

Finds the fixed point of the automata dynamics.

Instead of running for fixed steps, we run until convergence or energy minimization.

"""

def __init__(self, automata: nn.Module, energy_fn: EnergyFunction):

super().__init__()

self.automata = automata

self.energy_fn = energy_fn

def forward(self, initial_state: torch.Tensor, max_steps: int = 10, tolerance: float = 1e-3):

state = initial_state

energies = []

for t in range(max_steps):

prev_state = state

# 1. Automata Step (Dynamics)

state = self.automata(state, steps=1)

# 2. Energy Gradient Descent (Optional Refinement)

# "Thinking" -> Moving down the energy gradient

# This makes the dynamics strictly Lyapunov-stable if aligned

# state.requires_grad_(True)

# E = self.energy_fn(state).sum()

# grad = torch.autograd.grad(E, state, create_graph=True)[0]

# state = state - 0.01 * grad # Gradient descent step

# Check convergence

diff = torch.mean((state - prev_state)**2)

E_val = self.energy_fn(state).mean().item()

energies.append(E_val)

if diff < tolerance:

break

return state, energies