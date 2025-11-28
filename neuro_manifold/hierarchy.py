"""
Hierarchical Neuro-Manifold

This module defines the multi-scale structure of the system.
Instead of a single flat layer of cells, we have:
    - Micro-Cells (Sensory/Fast): High frequency updates, tied to raw input.
    - Macro-Cells (Concept/Slow): Low frequency updates, integrate information from Micro-Cells.
    - Top-Down Modulation: Macro-Cells bias the dynamics of Micro-Cells.

Project Daedalus Update:
- Soul Injection: Macro-Cells are initialized with V_identity (The Immutable Director).
- Fixed Macro States: Top-down layer is not learned or updated dynamically in the same way.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .automata import ManifoldAutomata
from .geometry import RiemannianManifold
from .energy import EnergyFunction

class HierarchicalManifold(nn.Module):
    def __init__(self, input_dim: int,
                 num_micro: int = 64,
                 num_macro: int = 16,
                 state_dim: int = 32,
                 v_identity: Optional[torch.Tensor] = None):
        super().__init__()

        self.num_micro = num_micro
        self.num_macro = num_macro
        self.state_dim = state_dim

        # Geometry is shared
        # We pass bias_matrix later if needed, but here we assume geometry is passed or init internally
        # Actually HierarchicalManifold creates its own Geometry in the original code.
        # Ideally, we should inject the soul-biased geometry here.
        # For now, we will let Agent inject the biased geometry or handle it via initialization.
        # But wait, HierarchicalManifold creates the geometry instance: `self.geometry = RiemannianManifold(...)`
        # We need to change this to accept the Soul bias.

        # We need to construct geometry *with* the bias if available.
        # Since v_identity and v_truth interact, we might want to pass a pre-computed bias matrix
        # or just pass v_identity for now to affect initialization if we wanted.
        # Current geometry.py supports `bias_matrix`.
        # Let's create a placeholder geometry here. The Agent will likely overwrite or we need to pass args.
        # To keep it simple, we instantiate default here, but Agent can patch it.
        # BETTER: Use v_identity to create a bias matrix for Geometry if provided.

        bias_matrix = None
        if v_identity is not None:
             # Create a bias matrix from identity vector (outer product)
             # This makes the "Identity" direction easy to traverse
             bias_matrix = torch.outer(v_identity, v_identity)

        self.geometry = RiemannianManifold(dim=state_dim, bias_matrix=bias_matrix)

        # Energy Function (Global)
        self.energy_fn = EnergyFunction(state_dim) # Will be patched with Soul vectors by Agent

        # Layer 1: Sensory (Micro)
        self.micro_layer = ManifoldAutomata(num_micro, state_dim, self.geometry)

        # Layer 2: Concept (Macro) - The Director
        self.macro_layer = ManifoldAutomata(num_macro, state_dim, self.geometry)

        # Bridges
        self.bottom_up = nn.MultiheadAttention(state_dim, num_heads=2, batch_first=True)
        self.top_down = nn.Linear(state_dim, state_dim)

        self.norm = nn.LayerNorm(state_dim)

        # Soul Injection: Immutable Director
        if v_identity is not None:
            self.register_buffer('v_identity', v_identity)
        else:
            self.register_buffer('v_identity', torch.zeros(state_dim))

    def forward(self,
                micro_states: torch.Tensor,
                macro_states: Optional[torch.Tensor] = None,
                micro_trace: Optional[torch.Tensor] = None,
                macro_trace: Optional[torch.Tensor] = None,
                steps: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Interleaved execution of Micro and Macro layers.
        Returns: micro_states, macro_states, micro_trace, macro_trace, energy
        """
        B = micro_states.shape[0]

        # Soul Injection: Macro Initialization
        # If macro_states is None, we initialize it with V_identity (The Director)
        # instead of zeros.
        if macro_states is None:
            # Broadcast V_identity to (B, Num_Macro, D)
            # We want the director to be present.
            identity_expanded = self.v_identity.view(1, 1, -1).expand(B, self.num_macro, -1)
            macro_states = identity_expanded.clone()

        final_energy = torch.zeros(B, device=micro_states.device)

        # Simulation Loop
        for t in range(steps):
            # 1. Micro Update (Fast)
            # Apply top-down bias from Macro Director
            # The Director (Macro) pushes its state (Identity) down to Micro
            macro_bias = self.top_down(macro_states.mean(dim=1, keepdim=True))

            # Daedalus: Macro cells are "Immutable Directors".
            # We can force them to stay close to Identity or just let them drift slowly.
            # Spec says: "Upper layer is not learned... Immutable Director".
            # So we should probably reset macro_states to identity or very heavily anchor it.
            # Let's anchor it:
            # macro_states = 0.9 * macro_states + 0.1 * self.v_identity

            micro_states = micro_states + 0.1 * macro_bias

            micro_states, micro_trace = self.micro_layer(micro_states, hebbian_trace=micro_trace, steps=1)

            # 2. Bottom-Up Integration
            # Macro cells attend to Micro cells
            # Query: Macro, Key/Val: Micro
            integrated, _ = self.bottom_up(macro_states, micro_states, micro_states)

            # Daedalus: The Director observes but does not change its core nature.
            # It only updates its temporary "working memory" to contextualize.
            macro_states = self.norm(macro_states + 0.1 * integrated) # Reduced update rate for stability

            # 3. Macro Update (Slow)
            macro_states, macro_trace = self.macro_layer(macro_states, hebbian_trace=macro_trace, steps=1)

            # Re-Anchor to Identity (Soul Injection)
            # Ensure the Director never forgets who it is.
            if hasattr(self, 'v_identity'):
                identity_expanded = self.v_identity.view(1, 1, -1).expand(B, self.num_macro, -1)
                macro_states = 0.95 * macro_states + 0.05 * identity_expanded

            # Calculate Energy (Thought as Equilibrium)
            final_energy = self.energy_fn(micro_states)

        return micro_states, macro_states, micro_trace, macro_trace, final_energy
