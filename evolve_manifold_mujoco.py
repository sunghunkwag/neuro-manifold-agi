"""

Evolve Neuro-Manifold in MuJoCo (Advanced) - Project Daedalus Edition

This script runs the evolutionary/RL loop for the Neuro-Manifold Automata V3 (Daedalus).
It incorporates the "Tri-Lock System" and "Soul Injection".

Daedalus Changes:
1.  **Soul Injection**: Initializes with Daedalus vectors.
2.  **Tri-Lock**: Enforces stability constraints.
3.  **Training Protocol**:
    - Phase 1: Warm-up (Truth Learning).
    - Phase 2: Awakening (Energy Minimization).
4.  **Fixes**: Correct handling of agent state and return values.

"""

import gymnasium as gym

import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import json

from datetime import datetime

from collections import deque

from neuro_manifold.agent import ManifoldAgent

def train_manifold():

    print("=" * 80)
    print("Project Daedalus - Neuro-Manifold V3 Evolution")
    print("=" * 80)

    env_id = "HalfCheetah-v4"
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Organism (Daedalus Agent)
    agent = ManifoldAgent(obs_dim, action_dim, num_micro=16, num_macro=4).to(device)

    # We use a smaller LR for stability with the Energy dynamics
    optimizer = torch.optim.AdamW(agent.parameters(), lr=3e-4, weight_decay=1e-4)

    print(agent)

    # Evolution Loop Configuration
    n_generations = 3 # Adjusted for smoke test
    steps_per_gen = 64 # Reduced for smoke test (verify logic flow)

    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2

    # Daedalus: Truth Alignment Weight
    truth_coef = 0.1

    history = {'returns': [], 'energy': []}

    # Pre-training / Warm-up flag
    # In a full run, we'd loop epochs. Here we treat Generation 1 as Warm-up.

    for gen in range(1, n_generations + 1):

        # Phase Control
        if gen <= 1:
            print(f"Gen {gen} [Phase 1: Warm-up] - Locking Plasticity, Truth Seeking")
            agent.brain.micro_layer.learning_rate_gate = 0.0 # Lock 3
        else:
            print(f"Gen {gen} [Phase 2: Awakening] - Unlocking Plasticity, Energy Optimization")
            agent.brain.micro_layer.learning_rate_gate = 1.0 # Unlock

        # 1. Gather Experience (Life)
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []

        # State Buffer for sequential training
        # We store the state *before* the step to pass as initial_state
        state_buf = []

        obs, _ = env.reset(seed=seed + gen)
        agent.reset()

        ep_ret = 0
        ep_len = 0

        agent.eval()

        for t in range(steps_per_gen):

            # Capture current state for training later
            # We detach to store copies
            current_state = agent.get_state()
            state_buf.append(current_state)

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                # Fix: Unpack 6 values
                mean, logstd, val, pred, flat_state, energy = agent(obs_t, mode='act')

            std = torch.exp(logstd)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(axis=-1)

            act_np = action.cpu().numpy()[0]
            val_np = val.item()
            logp_np = logp.item()
            energy_val = energy.mean().item()

            next_obs, reward, terminated, truncated, _ = env.step(act_np)
            done = terminated or truncated

            # Daedalus: Intrinsic Reward (Energy Minimization)
            # The agent wants to minimize Energy.
            # In RL, Reward = -Energy.
            # We blend External Reward (Physics) and Intrinsic (Truth/Energy).
            # If Energy is high (Bad/Reject), Reward is low.
            intrinsic_reward = -0.1 * energy_val
            total_reward = reward + intrinsic_reward

            # Store
            obs_buf.append(obs)
            act_buf.append(act_np)
            rew_buf.append(total_reward) # Train on total reward
            val_buf.append(val_np)
            logp_buf.append(logp_np)
            done_buf.append(done)

            obs = next_obs
            ep_ret += reward # Log only external reward for user visibility
            ep_len += 1

            if done or (t == steps_per_gen - 1):
                if done:
                    print(f"Gen {gen} | Ep Return: {ep_ret:.2f} | Final Energy: {energy_val:.2f}")
                    history['returns'].append(ep_ret)
                    history['energy'].append(energy_val)

                obs, _ = env.reset()
                agent.reset()
                ep_ret = 0
                ep_len = 0

        # 2. Compute Advantages (GAE)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, last_val, _, _, _ = agent(obs_t, mode='act')
            last_val = last_val.item()

        adv_buf = np.zeros_like(rew_buf, dtype=np.float32)
        last_gae = 0

        for t in reversed(range(len(rew_buf))):
            if t == len(rew_buf) - 1:
                next_non_terminal = 1.0 - float(done_buf[t])
                next_val = last_val
            else:
                next_non_terminal = 1.0 - float(done_buf[t])
                next_val = val_buf[t+1]

            delta = rew_buf[t] + gamma * next_val * next_non_terminal - val_buf[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            adv_buf[t] = last_gae

        ret_buf = adv_buf + np.array(val_buf)

        # 3. Optimize (Mutate/Update)
        obs_tens = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_tens = torch.as_tensor(np.array(act_buf), dtype=torch.float32, device=device)
        logp_tens = torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        adv_tens = torch.as_tensor(adv_buf, dtype=torch.float32, device=device)
        ret_tens = torch.as_tensor(ret_buf, dtype=torch.float32, device=device)

        # Normalize adv
        adv_tens = (adv_tens - adv_tens.mean()) / (adv_tens.std() + 1e-8)

        # Collate State Buffer (Dictionary of lists -> Dictionary of tensors)
        # We need to stack the states to batch process them, BUT this is tricky with recurrent states.
        # Standard PPO trains on shuffled batches, which breaks recurrence.
        # Recurrent PPO usually trains on sequences.
        # For this fix, we will just use the stored states as "Initial State" for each timestep
        # effectively doing "Teacher Forcing" of the internal state.
        # This allows random batch sampling without breaking the state continuity locally.

        # state_buf is list of dicts.
        # keys: micro, macro, micro_trace, macro_trace
        collated_state = {}
        for k in ['micro', 'macro', 'micro_trace', 'macro_trace']:
            # items are tensors of shape (1, ...). Stack to (Batch, ...)
            # handle None
            valid_items = [s[k] for s in state_buf]
            if valid_items[0] is None:
                collated_state[k] = None
            else:
                collated_state[k] = torch.cat(valid_items, dim=0)

        agent.train()

        # Train Loop
        # We process the whole collected batch with shuffled indices
        batch_size = len(obs_buf)
        minibatch_size = 64
        indices = np.arange(batch_size)

        for i in range(10): # Epochs
            np.random.shuffle(indices)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]

                mb_obs = obs_tens[mb_inds]
                mb_act = act_tens[mb_inds]
                mb_logp = logp_tens[mb_inds]
                mb_adv = adv_tens[mb_inds]
                mb_ret = ret_tens[mb_inds]

                # Slice the state dictionary
                mb_state = {}
                for k, v in collated_state.items():
                    if v is not None:
                        mb_state[k] = v[mb_inds]
                    else:
                        mb_state[k] = None

                optimizer.zero_grad()

                # Forward with injected state history (Fixes Logic Bug)
                mean, logstd, values, prediction, flat_s, energy = agent(
                    mb_obs, initial_state=mb_state, mode='train'
                )

                # Losses
                # 1. Prediction Loss (Reconstruction/Consistency)
                loss_pred = torch.mean(prediction**2) * 0.01

                # 2. Energy Loss (Minimize Energy directly?)
                # We already reward low energy. We can also add aux loss to minimize energy.
                loss_energy = energy.mean() * 0.01

                # 3. PPO Loss
                std = torch.exp(logstd)
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(mb_act).sum(axis=-1)

                ratio = torch.exp(logp - mb_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv

                loss_pi = -torch.min(surr1, surr2).mean()
                loss_v = F.mse_loss(values.squeeze(-1), mb_ret)

                loss = loss_pi + 0.5 * loss_v + loss_pred + loss_energy

                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        print(f"Gen {gen} | Loss: {loss.item():.4f}")

    # Save results
    with open("metrics_daedalus.json", "w") as f:
        json.dump(history, f)

    print("Training Complete. Metrics saved to metrics_daedalus.json")

if __name__ == "__main__":
    train_manifold()
