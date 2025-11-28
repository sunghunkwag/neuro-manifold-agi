"""

Neuro-Manifold Agent Interface (V2)

Integrates the Hierarchical Manifold system into an RL agent.

Adds Intrinsic Motivation (Curiosity) based on prediction error on the manifold.

"""

import torch

import torch.nn as nn

from .hierarchy import HierarchicalManifold

from .energy import EnergyFunction, EquilibriumSolver

class ManifoldAgent(nn.Module):

def __init__(self, input_dim: int, action_dim: int, num_micro: int = 64, num_macro: int = 16, state_dim: int = 32):

super().__init__()

self.input_dim = input_dim

self.action_dim = action_dim

self.num_cells = num_micro # Main read-out from micro

self.state_dim = state_dim

# 1. Hierarchical Brain

self.brain = HierarchicalManifold(input_dim, num_micro, num_macro, state_dim)

# 2. Physics (Energy)

# Energy is defined over the Micro states (Sensory reality)

self.energy_fn = EnergyFunction(state_dim)

# 3. Solver (Equilibrium logic is implicit in the Hierarchical forward pass now)

# But we can still iterate the brain

# 4. Interfaces

self.sensor_map = nn.Linear(input_dim, num_micro * state_dim)

# Motor: Reads from BOTH Micro and Macro

readout_dim = (num_micro + num_macro) * state_dim

# Actually, let's just read from Micro for simplicity or define a specific readout

# The hierarchy forward returns Micro, but we want internal access.

# Let's keep it simple: Read from the output of the brain (Micro modulated by Macro)

self.motor_readout = nn.Sequential(

nn.Linear(num_micro * state_dim, 256),

nn.SiLU(),

nn.Linear(256, action_dim * 2) # Mean, LogStd



self.value_head = nn.Linear(num_micro * state_dim, 1)

# Intrinsic Motivation: Predictor

# Predicts the next sensory state on the manifold

self.predictor = nn.Linear(num_micro * state_dim, num_micro * state_dim)

self.internal_state = None

def reset(self):

self.internal_state = None

def forward(self, obs: torch.Tensor, mode: str = 'act'):

B = obs.shape[0]

if self.internal_state is None or self.internal_state.shape[0] != B:

self.internal_state = torch.zeros(B, self.num_cells, self.state_dim, device=obs.device)

# 1. Sensation

sensory_signal = self.sensor_map(obs).reshape(B, self.num_cells, self.state_dim)

perturbed_state = self.internal_state + sensory_signal

# 2. Cognition (Hierarchical Simulation)

# Run the brain for a few steps to settle

final_micro_state = self.brain(perturbed_state, steps=3)

# 3. Intrinsic Motivation Calculation

# Prediction Error = |Predictor(State_t) - State_{t+1}|

# We calculate this conceptually; for training we return the prediction

predicted_next = self.predictor(self.internal_state.reshape(B, -1))

target_next = final_micro_state.reshape(B, -1).detach()

intrinsic_reward = torch.mean((predicted_next - target_next)**2, dim=-1, keepdim=True)

# Update State

self.internal_state = final_micro_state.detach()

# 4. Action

flat_state = final_micro_state.reshape(B, -1)

action_out = self.motor_readout(flat_state)

mean, logstd = action_out.chunk(2, dim=-1)

mean = torch.tanh(mean)

logstd = torch.clamp(logstd, -20, 2)

value = self.value_head(flat_state)

if mode == 'train':

# Return prediction for auxiliary loss

prediction = self.predictor(flat_state)

return mean, logstd, value, prediction

else:

return mean, logstd, value

if __name__ == "__main__":

agent = ManifoldAgent(17, 6)

obs = torch.randn(4, 17)

mean, _, _ = agent(obs, mode='act')

print("Hierarchical Agent forward pass successful.")