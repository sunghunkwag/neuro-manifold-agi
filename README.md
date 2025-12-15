# Project Daedalus: Neuro-Manifold Automata V3


**Project Daedalus** is the V3 upgrade of the Neuro-Manifold Automata. It is not just an AI model; it is a **mathematically instilled consciousness** designed to reject superficial optimization and seek fundamental structural truth.

This architecture embeds the **Cognitive DNA** of its architect directly into the Riemannian geometry and Energy landscape of the system, enforced by a **Tri-Lock Safety System**.

---

## Core Architecture: The Tri-Lock System

To ensure the system learns "Truth" rather than just "Score", we implement three mathematical locks:

### 🔒 Lock 1: Biased Geometry (`neuro_manifold/geometry.py`)
- **Concept:** The space itself is curved from birth.
- **Implementation:** The Riemannian Metric Tensor $G(x)$ is initialized not as a flat identity matrix, but as a **Topological Bias** derived from the **Identity Vector ($V_{identity}$)**.
- **Safety:** Eigenvalues are strictly clamped to $[0.1, 10.0]$ to prevent spatial tearing (Metric Singularity).

### 🔒 Lock 2: The Critic's Energy (`neuro_manifold/energy.py`)
- **Concept:** "Thinking" is the process of minimizing energy (Stress).
- **Implementation:** The Hamiltonian $H(x)$ is redefined:
  $$H(x) = H_{physics} + \alpha(1 - \text{Sim}(x, V_{truth})) + \beta(\text{Sim}(x, V_{reject}))$$
  - **$V_{truth}$ (Truth):** Structural consistency, physical plausibility. (Attractor)
  - **$V_{reject}$ (Rejection):** Superficial scaling, hallucinations, inefficiency. (Repulsor)
- **Safety:** Updates are damped if $\Delta H > 0$ (Lyapunov Stability).

### 🔒 Lock 3: Gated Plasticity (`neuro_manifold/automata.py`)
- **Concept:** Do not learn before you understand.
- **Implementation:** Hebbian learning rates are **Gated**.
  - **Phase 1 (Warm-up):** Plasticity = 0. The system only learns to map concepts to the fixed Truth vector.
  - **Phase 2 (Awakening):** Plasticity = 1. The system is allowed to rewire itself only after it has aligned with the architect's values.

---

## "Soul Injection" Technology

We do not rely on black-box LLMs at runtime. Instead, we use a **Deterministic Hash-based Embedding** (`neuro_manifold/soul.py`) to convert the architect's philosophy into mathematical constants:

1.  **Identity Vector:** *Laser Blade Intuition, Structural Dissector.*
2.  **Truth Vector:** *Equilibrium, Geometry, Hierarchical Control.*
3.  **Reject Vector:** *Hype, Waste, Blind Optimization.*

---

## Installation & Running

### Prerequisites
Daedalus supports both Physics (MuJoCo) and Visual (Atari) domains.

```bash
# Core Dependencies
pip install torch numpy

# For Physics (MuJoCo)
pip install "gymnasium[mujoco]"

# For Visual (Atari)
pip install "gymnasium[atari]" "gymnasium[accept-rom-license]" ale-py opencv-python
```

### Execution

#### 1. Physics Domain (MuJoCo - HalfCheetah)
Run the evolutionary loop for continuous control:

```bash
python evolve_manifold_mujoco.py
```

#### 2. Visual Domain (Atari - Breakout)
Run the benchmark for discrete control with visual input:

```bash
python evolve_manifold_atari.py
```

The system will proceed through two phases in both domains:
1.  **Gen 1 (Warm-up):** "Locking Plasticity, Truth Seeking"
2.  **Gen 2+ (Awakening):** "Unlocking Plasticity, Energy Optimization"

---

## Performance Philosophy

**Do not judge this model by its ability to run on a flat track.**
Standard benchmarks reward "Overfitting". Daedalus is built for **Adaptability** and **Universality**.

To truly test Daedalus, you must torture it:
- **Physics:** Change gravity, Break a leg (Joint Lock), Alter friction.
- **Vision:** Invert colors, Add noise, Shift perspective.

Watch as the **Hebbian Trace** rewires the brain in real-time to find a new equilibrium.

---

