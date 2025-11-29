"""
Evolve Neuro-Manifold in Atari (Visual/Discrete) - Project Daedalus

This script adapts the Daedalus architecture for the Atari Benchmark.
It tests the "Structural Truth" philosophy in a pixel-based, discrete action domain.

Environment: BreakoutNoFrameskip-v4
"""

import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordEpisodeStatistics
from neuro_manifold.agent import ManifoldAgent

# Register Atari Envs
gym.register_envs(ale_py)

def train_atari():
    print("=" * 80)
    print("Project Daedalus - Atari Benchmark Verification")
    print("=" * 80)

    env_id = "ALE/Breakout-v5" # Modern standard ID
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Atari Environment Setup
    # ALE/Breakout-v5 usually has frameskip=4 by default.
    # We want explicit control so we use 'frameskip=1' in make if we use AtariPreprocessing(frame_skip=4)
    # However, AtariPreprocessing expects raw env.
    def make_env():
        # render_mode='rgb_array' for potential visualization or just standard
        env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0)
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, scale_obs=False, frame_skip=4)
        env = FrameStackObservation(env, stack_size=4)
        return env

    env = make_env()

    # Obs shape: (4, 84, 84)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print(f"Observation Shape: {obs_shape}")
    print(f"Action Dimension: {action_dim} (Discrete)")

    # Organism (Daedalus Agent with Visual Cortex)
    # use_cnn=True, is_discrete=True
    agent = ManifoldAgent(
        input_dim=0, # Ignored when use_cnn=True
        action_dim=action_dim,
        num_micro=32, # More neurons for visual processing
        num_macro=8,
        is_discrete=True,
        use_cnn=True
    ).to(device)

    optimizer = torch.optim.AdamW(agent.parameters(), lr=2e-4, weight_decay=1e-4)

    print(agent)

    # Evolution Loop Configuration
    n_generations = 3
    steps_per_gen = 256 # Short horizon for verification
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2

    history = {'returns': [], 'energy': []}

    for gen in range(1, n_generations + 1):

        # Phase Control
        if gen <= 1:
            print(f"Gen {gen} [Phase 1: Warm-up] - Locking Plasticity")
            agent.brain.micro_layer.learning_rate_gate = 0.0
        else:
            print(f"Gen {gen} [Phase 2: Awakening] - Unlocking Plasticity")
            agent.brain.micro_layer.learning_rate_gate = 1.0

        # 1. Gather Experience
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
        state_buf = []

        obs, _ = env.reset(seed=seed + gen)
        agent.reset()

        ep_ret = 0
        ep_len = 0

        agent.eval()

        for t in range(steps_per_gen):

            # Store State
            current_state = agent.get_state()
            state_buf.append(current_state)

            # Obs to Tensor: (1, 4, 84, 84)
            # Gym returns LazyFrames, convert to array first
            obs_np = np.array(obs)
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                # mean here is logits
                logits, _, val, pred, flat_state, energy = agent(obs_t, mode='act')

            # Discrete Action Sampling
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            act_np = action.cpu().numpy()[0]
            val_np = val.item()
            logp_np = logp.item()
            energy_val = energy.mean().item()

            next_obs, reward, terminated, truncated, _ = env.step(act_np)
            done = terminated or truncated

            # Daedalus Intrinsic Reward
            intrinsic_reward = -0.05 * energy_val
            total_reward = reward + intrinsic_reward

            # Store
            obs_buf.append(np.array(obs)) # Store as numpy array
            act_buf.append(act_np)
            rew_buf.append(total_reward)
            val_buf.append(val_np)
            logp_buf.append(logp_np)
            done_buf.append(done)

            obs = next_obs
            ep_ret += reward
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
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32, device=device).unsqueeze(0)
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

        # 3. Optimize
        obs_tens = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_tens = torch.as_tensor(np.array(act_buf), dtype=torch.long, device=device) # Long for Discrete
        logp_tens = torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        adv_tens = torch.as_tensor(adv_buf, dtype=torch.float32, device=device)
        ret_tens = torch.as_tensor(ret_buf, dtype=torch.float32, device=device)

        adv_tens = (adv_tens - adv_tens.mean()) / (adv_tens.std() + 1e-8)

        # Collate State Buffer (Fix from previous step)
        collated_state = {}
        for k in ['micro', 'macro', 'micro_trace', 'macro_trace']:
            sample_tensor = next((item for item in [s[k] for s in state_buf] if item is not None), None)
            if sample_tensor is None:
                collated_state[k] = None
            else:
                processed_items = []
                for item in [s[k] for s in state_buf]:
                    if item is None:
                         processed_items.append(torch.zeros_like(sample_tensor))
                    else:
                         processed_items.append(item)
                collated_state[k] = torch.cat(processed_items, dim=0)

        agent.train()

        batch_size = len(obs_buf)
        minibatch_size = 32 # Smaller batch for images
        indices = np.arange(batch_size)

        for i in range(5): # Fewer epochs for verification speed
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]

                mb_obs = obs_tens[mb_inds]
                mb_act = act_tens[mb_inds]
                mb_logp = logp_tens[mb_inds]
                mb_adv = adv_tens[mb_inds]
                mb_ret = ret_tens[mb_inds]

                mb_state = {}
                for k, v in collated_state.items():
                    if v is not None:
                        mb_state[k] = v[mb_inds]
                    else:
                        mb_state[k] = None

                optimizer.zero_grad()

                # Forward
                logits, _, values, prediction, flat_s, energy = agent(
                    mb_obs, initial_state=mb_state, mode='train'
                )

                # Losses
                loss_pred = torch.mean(prediction**2) * 0.01
                loss_energy = energy.mean() * 0.01

                # PPO (Categorical)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(mb_act)

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

    with open("metrics_atari.json", "w") as f:
        json.dump(history, f)
    print("Atari Benchmark Verification Complete.")

if __name__ == "__main__":
    train_atari()
