# Neuro-Manifold Automata (NMA)

The **Neuro-Manifold Automata** is a radical departure from traditional deep learning architectures. It redefines artificial intelligence not as a stack of layers, but as a dynamic ecosystem of living cells inhabiting a curved, self-organizing Riemannian manifold.

This project implements an AGI substrate that focuses on **emergence**, **geometry**, and **energy equilibrium**.

## Core Philosophy

1.  **Life, Not Layers:** The fundamental unit is not a neuron or a layer, but a **Neural Cell**—an automaton with state, memory, and a life cycle.
2.  **Geometry is Intelligence:** Information does not flow through fixed wires. It flows along geodesics in a curved manifold. The system learns by bending space-time to bring related concepts together (Gravity) and push unrelated ones apart.
3.  **Thought as Equilibrium:** Inference is not a single pass. It is a physical relaxation process where the system settles into a low-energy state.

## Architecture

The system is organized into four distinct modules:

### 1. The Geometry (`neuro_manifold/geometry.py`)
- Implements a learnable **Riemannian Manifold**.
- Computes the Metric Tensor $g_{ij}(x)$ dynamically using Cholesky decomposition for stability.
- Defines distance and locality in a non-Euclidean space.

### 2. The Automata (`neuro_manifold/automata.py`)
- **Neural Cellular Automata (NCA):** Grid-less, graph-based cellular life forms.
- **Hebbian Plasticity:** Connections rewire dynamically during inference based on co-activation (Fast Weights).
- Robust to damage (Regeneration) and scalable.

### 3. The Energy (`neuro_manifold/energy.py`)
- Defines the **Hamiltonian** (Energy function) of the system.
- "Thinking" is implemented as an iterative settling process (Equilibrium Propagation).

### 4. The Agent (`neuro_manifold/agent.py`)
- Wraps the automata into an RL-compatible agent.
- Maps sensory inputs to manifold perturbations.
- Maps equilibrium states to motor actions.
- Includes **Intrinsic Motivation** based on manifold state prediction error.

## Installation

```bash
pip install torch numpy gymnasium[mujoco]
```

## Running the Evolution

To witness the evolution of the Neuro-Manifold Automata in a physics simulation (HalfCheetah-v4):

```bash
python evolve_manifold_mujoco.py
```

This script will:
1.  Spawn a colony of neural cells.
2.  Place them in a high-dimensional manifold.
3.  Let them interact with the MuJoCo physics engine.
4.  Evolve their "DNA" (parameters) via PPO-style energy minimization.

## Performance

Early experiments show that NMA exhibits **zero-shot adaptation** qualities superior to static MLPs, as the system dynamically settles into new equilibria when perturbations occur.

## References

- "Growing Neural Cellular Automata" (Mordvintsev et al., Distill, 2020)
- "Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation" (Scellier & Bengio, 2017)
- "Geometric Deep Learning" (Bronstein et al.)

---
**Author:** Manus AI
**Paradigm:** Neuro-Manifold / Geometric Deep Learning
