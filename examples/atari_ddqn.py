"""A basic example showing how to train a DQN agent to play Breakout from pixel information"""

import numpy as np
import os
import torch

from src.algorithms.double_deep_q_learning import DoubleDQNAtariAgent
from src.models import DDQN
from src.utils.assessment import AtariEvaluator
from src.utils.env import make_atari, wrap_deepmind
from src.utils.logger import Logger
from src.utils.replay_memory import ReplayMemory


def _moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return list(np.convolve(interval, window, "same"))


# -------Parameters----------
CAPACITY = 200_000
SKIP_N = 4

frames = 500_000
TARGET_UPDATE_FREQUENCY = 5_000

EPSILON_METHOD = "linear"
EPSILON_FRAMES = int(0.2 * frames)
EPSILON_ARGS = [EPSILON_METHOD, EPSILON_FRAMES]
EPSILON_KWARGS = {"epsilon_min": 0.1}


# ------Env------------------
env_name = "SeaquestNoFrameskip-v4"
env = make_atari(env_name)
env = wrap_deepmind(env)
n_actions = env.action_space.n
print(n_actions)

# -------Models--------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDQN(SKIP_N, 84, n_actions).to(device)
target_model = DDQN(SKIP_N, 84, n_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

memory = ReplayMemory(CAPACITY)

# ------Saving and Logging---
name = (
    env_name
    + "_{}capacity".format(CAPACITY)
    + "_{}{}EPSILON".format(EPSILON_FRAMES, EPSILON_METHOD)
    + "_{}C".format(TARGET_UPDATE_FREQUENCY)
)
save_path = os.path.join("results", "models", name)

logger = Logger(
    save_path,
    save_best=True,
    save_every=100,
    log_every=50,
    C=TARGET_UPDATE_FREQUENCY,
    capacity=CAPACITY,
)

# ------Training------------

agent = DoubleDQNAtariAgent(
    model, target_model, env, memory, logger, *EPSILON_ARGS, **EPSILON_KWARGS
)
agent.train(
    n_frames=frames,
    C=TARGET_UPDATE_FREQUENCY,
    render=False,
    frames_before_train=int(0.05 * frames),
)
# This saves a model to results/models/Breakout.....


# ------Evaluating----------
evaluator = AtariEvaluator(model, os.path.join(save_path, "best_model.pth"), device)
# Play once
evaluator.record(env, os.path.join("results", "videos", name))
# Get average score
# scores = evaluator.play(100, env, render=False)
# print("{:.3f} +/- {:.1f}".format(np.mean(scores), np.std(scores) / np.sqrt(len(scores))))
