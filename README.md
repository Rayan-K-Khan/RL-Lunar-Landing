# 🚀 Deep Q-Network (DQN) — Lunar Lander

A deep reinforcement learning project that trains an agent to autonomously land a spacecraft using a **Deep Q-Network (DQN)** built from scratch in PyTorch. The agent learns to navigate the `LunarLander-v3` Gymnasium environment, targeting the official solve threshold of **+200 cumulative reward**.

---

## 📌 Project Overview

| | |
|---|---|
| **Goal** | Train an RL agent to safely land a lunar module |
| **Environment** | `LunarLander-v3` (OpenAI Gymnasium / Box2D) |
| **Algorithm** | Deep Q-Network (DQN) with Experience Replay & Target Network |
| **Solve Threshold** | +200 cumulative episode reward |
| **Tools** | Python, PyTorch, Gymnasium, NumPy, Matplotlib |

---

## 🧠 Architecture

### State Space (8-dimensional input)
| Index | Feature |
|---|---|
| `[0]` | X coordinate of lander |
| `[1]` | Y coordinate of lander |
| `[2]` | Linear velocity (x-axis) |
| `[3]` | Linear velocity (y-axis) |
| `[4]` | Lander angle |
| `[5]` | Angular velocity |
| `[6]` | Left leg ground contact (boolean) |
| `[7]` | Right leg ground contact (boolean) |

### Action Space (4 discrete outputs)
| Action | Description |
|---|---|
| `0` | Do nothing |
| `1` | Fire left orientation engine |
| `2` | Fire main engine |
| `3` | Fire right orientation engine |

### Neural Network (DQN)
```
Input (8) → Linear → ReLU → Linear(128) → ReLU → Linear(128) → Output (4 Q-values)
```

---

## 🏆 Reward Structure

| Event | Reward |
|---|---|
| Moving toward pad / slowing down | Positive |
| Tilting | Negative |
| Main engine firing | −0.3 / frame |
| Side engine firing | −0.03 / frame |
| Each leg touchdown | +10 |
| Safe landing | +100 |
| Crash | −100 |

---

## 🔧 Training Details

### Key Components

**Experience Replay Buffer**
- `deque`-based circular buffer storing `(state, action, reward, next_state, done)` tuples
- Random mini-batch sampling of size 64 breaks temporal correlation between consecutive steps

**Epsilon-Greedy Exploration**
- Starts fully exploratory (ε = 1.0) and decays each episode by a factor of 0.995
- Floors at ε = 0.01 to preserve minimal exploration throughout training

**Target Network**
- Separate frozen copy of the policy network updated every 10 episodes
- Prevents the moving-target problem that destabilizes Q-value estimation

**Bellman Update (Q-target)**
```
Q_target = R + γ · max[Q(S', A')]     where γ = 0.99
```

### Hyperparameters
```
Episodes        : 150
Batch Size      : 64
Gamma (γ)       : 0.99
Epsilon Start   : 1.0
Epsilon Decay   : 0.995
Epsilon Min     : 0.01
Target Sync     : Every 10 episodes
Optimizer       : Adam
Loss            : MSE
```

---

## 📊 Evaluation

After training, the agent runs **2 evaluation episodes** in `render_mode="human"` using pure exploitation (ε = 0, no randomness) to visually demonstrate the learned landing policy.

---

## 🛠️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/your-username/dqn-lunar-lander.git
cd dqn-lunar-lander

# Install dependencies
pip install swig
pip install "gymnasium[box2d]" torch numpy matplotlib

# Launch the notebook
jupyter notebook RL_Lunar_Landing.ipynb
```

---

## 📁 Repository Structure

```
dqn-lunar-lander/
│
├── RL_Lunar_Landing.ipynb   # Full training and evaluation notebook
└── README.md                # Project documentation
```

---

## 🔍 Key Takeaways

- Experience replay and a frozen target network are critical for stable DQN training — without them, Q-value estimates diverge
- The ε-decay rate directly controls the explore/exploit tradeoff; too fast leads to premature convergence, too slow leads to erratic behavior even late in training
- Fuel efficiency penalties force the agent to learn economical thrust strategies, not just crash avoidance
- A discount factor of γ = 0.99 encourages the agent to plan several steps ahead rather than optimizing purely for immediate reward
