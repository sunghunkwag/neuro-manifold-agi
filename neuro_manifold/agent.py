"""
Neuro-Manifold Agent Interface (V2)

Integrates the Hierarchical Manifold system into an RL agent.
Adds Intrinsic Motivation (Curiosity) based on prediction error on the manifold.

Project Daedalus Update:
- Integrates Soul Vectors (Identity, Truth, Reject).
- Injects vectors into Hierarchy, Geometry, and Energy.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from .hierarchy import HierarchicalManifold
from .soul import get_soul_vectors

class ManifoldAgent(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, num_micro: int = 8, num_macro: int = 4, state_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_cells = num_micro
        self.state_dim = state_dim

        # 0. Soul Injection
        # Retrieve the cognitive DNA
        v_identity, v_truth, v_reject = get_soul_vectors(dim=state_dim)

        # Register them (optional, but good for tracking)
        self.register_buffer('v_identity', v_identity)
        self.register_buffer('v_truth', v_truth)
        self.register_buffer('v_reject', v_reject)

        # 1. Hierarchical Brain
        # Passes v_identity for Director initialization
        self.brain = HierarchicalManifold(input_dim, num_micro, num_macro, state_dim, v_identity=v_identity)

        # Inject Truth/Reject into Energy Function
        # Note: Hierarchy creates the EnergyFunction, we need to update it.
        # Ideally we pass it in init, but we can patch it here since we control the instance.
        self.brain.energy_fn.register_buffer('v_truth', v_truth)
        self.brain.energy_fn.register_buffer('v_reject', v_reject)

        # 2. Interfaces
        self.sensor_map = nn.Linear(input_dim, num_micro * state_dim)

        self.motor_readout = nn.Sequential(
            nn.Linear(num_micro * state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, action_dim * 2) # Mean, LogStd
        )

        self.value_head = nn.Linear(num_micro * state_dim, 1)

        # Intrinsic Motivation: Predictor
        self.predictor = nn.Linear(num_micro * state_dim, num_micro * state_dim)

        # Internal State Container
        self.state: Dict[str, Optional[torch.Tensor]] = {
            'micro': None,
            'macro': None,
            'micro_trace': None,
            'macro_trace': None
        }

    def reset(self):
        self.state = {
            'micro': None,
            'macro': None,
            'micro_trace': None,
            'macro_trace': None
        }

    def get_state(self):
        """Returns the current internal state dict (detached)."""
        return {k: v.detach().clone() if v is not None else None for k, v in self.state.items()}

    def set_state(self, state_dict):
        """Sets the internal state."""
        self.state = state_dict

    def forward(self, obs: torch.Tensor,
                initial_state: Optional[Dict[str, torch.Tensor]] = None,
                mode: str = 'act'):
        """
        Forward pass.
        If initial_state is provided, uses that (stateless mode for training).
        Otherwise uses self.state (stateful mode for inference).
        """
        B = obs.shape[0]

        # Determine starting state
        curr_state = {}
        if initial_state is not None:
             curr_state = initial_state
        else:
             curr_state = self.state

        # Initialize if None or Shape Mismatch (e.g. batch size change)
        micro = curr_state.get('micro')

        # If micro is None or Batch size mismatches, re-init
        if micro is None or micro.shape[0] != B:
            micro = torch.zeros(B, self.brain.num_micro, self.state_dim, device=obs.device)
            macro = None # Hierarchy handles init with Identity
            micro_trace = None
            macro_trace = None
        else:
            macro = curr_state.get('macro')
            micro_trace = curr_state.get('micro_trace')
            macro_trace = curr_state.get('macro_trace')

        # 1. Sensation
        sensory_signal = self.sensor_map(obs).reshape(B, self.brain.num_micro, self.state_dim)
        perturbed_state = micro + sensory_signal

        # 2. Cognition (Hierarchical Simulation)
        new_micro, new_macro, new_micro_trace, new_macro_trace, energy = self.brain(
            perturbed_state, macro, micro_trace, macro_trace, steps=3
        )

        # Update persistent state if in stateful mode (Inference)
        if initial_state is None:
            self.state = {
                'micro': new_micro.detach(),
                'macro': new_macro.detach(),
                'micro_trace': new_micro_trace.detach() if new_micro_trace is not None else None,
                'macro_trace': new_macro_trace.detach() if new_macro_trace is not None else None
            }

        # 3. Action & Value
        flat_state = new_micro.reshape(B, -1)
        action_out = self.motor_readout(flat_state)
        mean, logstd = action_out.chunk(2, dim=-1)

        # Clamp logstd for stability
        logstd = torch.clamp(logstd, -20, 2)

        value = self.value_head(flat_state)

        # Intrinsic Motivation / Auxiliary Task
        # Predictor(State_t) -> State_t (consistency)
        prediction = self.predictor(flat_state)

        if mode == 'train':
            # Return full bundle for loss calculation
            return mean, logstd, value, prediction, flat_state, energy
        else:
            # UNIFIED RETURN SIGNATURE FIX:
            # We return everything even in act mode to simplify the loop or just what's needed.
            # But the user specifically asked to fix the crash.
            # The Training Loop calls: `mean, logstd, val = agent(obs_t, mode='act')`
            # So we MUST return 3 values here to stay compatible with existing loop calls,
            # OR we change the loop.
            # The plan said "Update forward method to return prediction... in mode='act' as well".
            # Let's return the extra info but unpack carefully in the loop.
            return mean, logstd, value, prediction, flat_state, energy
