# Reinforcement Learning for Chess 

This project explores the design, training, and evaluation of intelligent chess-playing agents using **Deep Q-Learning**, **Monte Carlo Tree Search (MCTS)**, and **Minimax with Alpha-Beta pruning**. 
It was developed as part of a university research project to investigate how learning-based agents compare to classical approaches in terms of gameplay performance, decision-making efficiency, and adaptability.

---

##  Project Overview

The core objective was to answer:

> **"How does a reinforcement learning-based chess AI compare to a Minimax-based AI in terms of decision-making efficiency, learning adaptability, and gameplay performance?"**

To explore this, we implemented and evaluated:
-  A **DQN agent** trained through curriculum-based self-play
-  A **Minimax agent** with stochastic tie-breaking and a handcrafted evaluation function
-  An **MCTS agent** guided by a shared neural network

Training was performed using PyTorch and custom chess environment wrappers built on [python-chess](https://pypi.org/project/python-chess/).

---

##  Key Features

- **Curriculum-based self-play**: DQN trains against random, self, past, and minimax opponents.
- **Custom reward shaping**: Encourages captures, center control, castling, and penalizes repetition.
- **Efficient board encoding**: 18-channel tensor representation captures piece layout, castling rights, en passant, and turn.
- **Evaluation engine**: Logs game outcomes, move times, Q-values, and win rates across agents.

---

##  Reproducibility

To run training or experiments:

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Train your own model from scratch
python train_dqn_updated.py

# 4. Or use the provided pretrained models in saved_models/
# and run evaluation experiments directly:
python run_experiments.py
python run_targeted.py


