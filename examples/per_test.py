"""Demonstration of the power of Prioritised Experience Replay"""
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.algorithms.double_deep_q_learning import DoubleDQNAtariAgent
from src.models import DDQN
from src.utils.env import DiscreteCarRacing, wrap_deepmind
from src.utils.logger import Logger
from src.utils.replay_memory import ReplayMemory, PrioritisedMemory


def _moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return list(np.convolve(interval, window, "same"))


CAPACITY = 50_000
SKIP_N = 4

frames = 250_000
TARGET_UPDATE_FREQUENCY = 1_000

EPSILON_METHOD = "linear"
EPSILON_FRAMES = int(0.1 * frames)
EPSILON_ARGS = [EPSILON_METHOD, EPSILON_FRAMES]
EPSILON_KWARGS = {"epsilon_min": 0.1}


# ------Env------------------
name = "CarRacing-v0"
env = gym.make(
    name, verbose=0
)  # Verbosity off for CarRacing - track generation info can get annoying!

env = wrap_deepmind(env, episode_life=False)

if "CarRacing" in name:
    # DQN needs discrete inputs
    env = DiscreteCarRacing(env)
n_actions = env.action_space.n

# -------------------------------------------------Random memory---------------------------------------------------------
# -------Models--------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDQN(SKIP_N, 84, n_actions).to(device)
target_model = DDQN(SKIP_N, 84, n_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

memory = ReplayMemory(CAPACITY)

# ------Saving and Logging----
save_path = os.path.join("results", "models", name + "_per_test")

logger = Logger(
    save_path,
    save_best=False,
    save_every=np.nan,
    log_every=25,
    C=TARGET_UPDATE_FREQUENCY,
    capacity=CAPACITY,
)

# ------Training--------------
agent = DoubleDQNAtariAgent(
    model, target_model, env, memory, logger, *EPSILON_ARGS, **EPSILON_KWARGS
)
agent.train(n_frames=frames, C=TARGET_UPDATE_FREQUENCY, render=False)

random_rewards = agent.logger._rewards
random_q = agent.logger._q_values


# -------------------------------------------------Prioritised memory----------------------------------------------------
# -------Models--------------
model = DDQN(SKIP_N, 84, n_actions).to(device)
target_model = DDQN(SKIP_N, 84, n_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

memory = PrioritisedMemory(CAPACITY)

# ------Saving and Logging----
env.reset()
logger.clear()

# ------Training--------------
agent = DoubleDQNAtariAgent(
    model, target_model, env, memory, logger, *EPSILON_ARGS, **EPSILON_KWARGS
)
agent.train(n_frames=frames, C=TARGET_UPDATE_FREQUENCY, render=False)

prioritised_rewards = agent.logger._rewards
prioritised_q = agent.logger._q_values


# ---------------------------------------------------Plots---------------------------------------------------------------
q = _moving_average(random_q, 50)
q_per = _moving_average(prioritised_q, 50)

plt.plot(range(len(q)), q, label="Random memory")
plt.plot(range(len(q_per)), q_per, label="Prioritised memory")

plt.xlabel("Episode")
plt.ylabel("Total episode Q value predictions")
plt.legend()

plt.savefig(os.path.join(save_path, "q_values.png"))


# Rewards
r = _moving_average(random_rewards, 50)
r_per = _moving_average(prioritised_rewards, 50)

plt.plot(range(len(r)), r, label="Random memory")
plt.plot(range(len(r_per)), r_per, label="Prioritised memory")

plt.xlabel("Episode")
plt.ylabel("Total episode reward")
plt.legend()

plt.savefig(os.path.join(save_path, "rewards.png"))
