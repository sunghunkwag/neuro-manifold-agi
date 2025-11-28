"""
Energy Dynamics & Equilibrium Solver

The "Mind" of the system.
The system does not just output a value; it settles into an energy minimum.
This implements Energy-Based Models (EBM) logic (Hopfield Network / Equilibrium Propagation style).

Project Daedalus Update:
- Soul Injection: Energy function is shaped by Truth and Reject vectors.
- Lock 2: Energy Conservation check (Lyapunov stability).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class EnergyFunction(nn.Module):
    """
    Defines the global energy landscape E(x).
    Now includes Soul-based terms.
    """
    def __init__(self, state_dim: int,
                 v_truth: Optional[torch.Tensor] = None,
                 v_reject: Optional[torch.Tensor] = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.SiLU(),
            nn.Linear(state_dim * 2, 1)
        )

        # Soul Vectors (Buffers)
        if v_truth is not None:
            self.register_buffer('v_truth', v_truth)
        else:
            self.register_buffer('v_truth', torch.zeros(state_dim))

        if v_reject is not None:
            self.register_buffer('v_reject', v_reject)
        else:
            self.register_buffer('v_reject', torch.zeros(state_dim))

        # Weights for Soul Energy
        self.alpha = 1.0 # Attraction to Truth
        self.beta = 10.0 # Repulsion from Rejection (Strong penalty)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute total energy H(x).
        H(x) = H_physics(x) + alpha * Dist(x, Truth) + beta * Sim(x, Reject)
        """
        # x: (B, N, D)

        # 1. Physics Energy (Internal Consistency)
        energies_phys = self.net(x).squeeze(-1) # (B, N)

        # 2. Soul Energy (Alignment)
        # Truth Attraction: Minimize Cosine Distance (Maximize Cosine Sim)
        # We use 1 - CosSim as 'Distance' energy
        # Normalize x per cell
        x_norm = F.normalize(x, p=2, dim=-1)

        # Truth is (D,). Expand to (1, 1, D)
        truth_expanded = self.v_truth.view(1, 1, -1)
        reject_expanded = self.v_reject.view(1, 1, -1)

        # Sim: (B, N)
        sim_truth = torch.sum(x_norm * truth_expanded, dim=-1)
        sim_reject = torch.sum(x_norm * reject_expanded, dim=-1)

        # Energy Terms
        # If sim_truth is 1, energy is 0. If -1, energy is 2*alpha.
        e_truth = self.alpha * (1.0 - sim_truth)

        # Reject Repulsion: If sim_reject is high, energy explodes.
        # We want simple linear penalty for high similarity
        e_reject = self.beta * F.relu(sim_reject)

        total_energy_per_cell = energies_phys + e_truth + e_reject

        return total_energy_per_cell.sum(dim=1) # (B,)


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

        # Initial Energy
        prev_E = self.energy_fn(state).mean().item()
        energies.append(prev_E)

        for t in range(max_steps):
            prev_state = state.clone()

            # 1. Automata Step (Dynamics)
            # Try a candidate update
            candidate_state = self.automata(state, steps=1)

            # 2. Lock 2: Energy Conservation Check (Lyapunov Stability)
            # We check if H(new) <= H(old).
            # In a strict physical simulation, we would reject.
            # But in learning, we might just want to guide it.
            # Here we implement a Soft Lock: Interpolate back if energy rises too much.

            current_E_tens = self.energy_fn(candidate_state)
            current_E = current_E_tens.mean().item()

            # "Rejection of Complacency" Logic:
            # If Energy (Error) increases, we don't just reject, we damped the update.
            if current_E > prev_E:
                # Damping factor
                state = 0.5 * prev_state + 0.5 * candidate_state
            else:
                state = candidate_state

            # 3. Energy Gradient Descent (Refinement)
            # Active inference: slightly nudge state towards lower energy
            # This helps finding the "Truth" ground state
            if state.requires_grad:
                # We need to re-evaluate energy on the accepted state
                # Detach to start a new graph segment for the gradient step?
                # No, we want to optimize the tensor values themselves temporarily.
                # But standard PyTorch autograd doesn't support easy in-place optimization inside forward.
                # We skip this for now to avoid complexity overhead, relying on the automata training.
                pass

            # Check convergence
            diff = torch.mean((state - prev_state)**2)
            energies.append(current_E)

            if diff < tolerance:
                break

            prev_E = current_E

        return state, energies
