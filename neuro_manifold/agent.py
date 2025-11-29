"""
Neuro-Manifold Agent Interface (V2)

Integrates the Hierarchical Manifold system into an RL agent.
Adds Intrinsic Motivation (Curiosity) based on prediction error on the manifold.

Project Daedalus Update:
- Integrates Soul Vectors (Identity, Truth, Reject).
- Injects vectors into Hierarchy, Geometry, and Energy.
- Supports Discrete Action Spaces (Atari) and Image Inputs (CNN).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from .hierarchy import HierarchicalManifold
from .soul import get_soul_vectors

class ManifoldAgent(nn.Module):
    def __init__(self, input_dim: int, action_dim: int,
                 num_micro: int = 8, num_macro: int = 4, state_dim: int = 32,
                 is_discrete: bool = False, use_cnn: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_cells = num_micro
        self.state_dim = state_dim
        self.is_discrete = is_discrete
        self.use_cnn = use_cnn

        # 0. Soul Injection
        v_identity, v_truth, v_reject = get_soul_vectors(dim=state_dim)

        self.register_buffer('v_identity', v_identity)
        self.register_buffer('v_truth', v_truth)
        self.register_buffer('v_reject', v_reject)

        # 1. Hierarchical Brain
        self.brain = HierarchicalManifold(input_dim, num_micro, num_macro, state_dim, v_identity=v_identity)

        self.brain.energy_fn.register_buffer('v_truth', v_truth)
        self.brain.energy_fn.register_buffer('v_reject', v_reject)

        # 2. Interfaces

        # Sensation (Visual Cortex vs Linear)
        if self.use_cnn:
            # Simple 3-layer CNN for Atari (84x84x4 assumed standard, or similar)
            # We assume input is (B, C, H, W).
            # Output will be flattened and projected to manifold dimension.
            self.visual_cortex = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            # 64 * 7 * 7 = 3136 for 84x84 input
            # We map this to (num_micro * state_dim)
            self.sensor_map = nn.Linear(3136, num_micro * state_dim)
        else:
            self.visual_cortex = None
            self.sensor_map = nn.Linear(input_dim, num_micro * state_dim)

        # Motor Cortex
        self.motor_hidden = nn.Sequential(
            nn.Linear(num_micro * state_dim, 256),
            nn.SiLU()
        )

        if self.is_discrete:
            # Output Logits for Categorical
            self.motor_head = nn.Linear(256, action_dim)
        else:
            # Output Mean, LogStd for Gaussian
            self.motor_head = nn.Linear(256, action_dim * 2)

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
        obs: (B, Obs_Dim) or (B, C, H, W) if CNN
        """
        B = obs.shape[0]

        # Determine starting state
        curr_state = {}
        if initial_state is not None:
             curr_state = initial_state
        else:
             curr_state = self.state

        # Initialize if None or Shape Mismatch
        micro = curr_state.get('micro')

        if micro is None or micro.shape[0] != B:
            micro = torch.zeros(B, self.brain.num_micro, self.state_dim, device=obs.device)
            macro = None
            micro_trace = None
            macro_trace = None
        else:
            macro = curr_state.get('macro')
            micro_trace = curr_state.get('micro_trace')
            macro_trace = curr_state.get('macro_trace')

        # 1. Sensation
        if self.use_cnn:
            # Expecting (B, C, H, W). If input is (B, H, W, C), permute.
            if obs.ndim == 4 and obs.shape[-1] == 4: # Common Gym (H,W,C)
                 obs = obs.permute(0, 3, 1, 2)

            # Normalize if not already (Atari usually 0-255)
            if obs.max() > 1.0:
                obs = obs / 255.0

            visual_features = self.visual_cortex(obs)
            sensory_signal = self.sensor_map(visual_features).reshape(B, self.brain.num_micro, self.state_dim)
        else:
            sensory_signal = self.sensor_map(obs).reshape(B, self.brain.num_micro, self.state_dim)

        perturbed_state = micro + sensory_signal

        # 2. Cognition (Hierarchical Simulation)
        new_micro, new_macro, new_micro_trace, new_macro_trace, energy = self.brain(
            perturbed_state, macro, micro_trace, macro_trace, steps=3
        )

        if initial_state is None:
            self.state = {
                'micro': new_micro.detach(),
                'macro': new_macro.detach(),
                'micro_trace': new_micro_trace.detach() if new_micro_trace is not None else None,
                'macro_trace': new_macro_trace.detach() if new_macro_trace is not None else None
            }

        # 3. Action & Value
        flat_state = new_micro.reshape(B, -1)

        motor_features = self.motor_hidden(flat_state)

        if self.is_discrete:
            # Return Logits
            action_out = self.motor_head(motor_features)
            # For compatibility with training loop that expects (mean, logstd)
            # We return (logits, None)
            mean = action_out
            logstd = torch.zeros_like(mean) # Dummy
        else:
            action_out = self.motor_head(motor_features)
            mean, logstd = action_out.chunk(2, dim=-1)
            logstd = torch.clamp(logstd, -20, 2)

        value = self.value_head(flat_state)
        prediction = self.predictor(flat_state)

        return mean, logstd, value, prediction, flat_state, energy
