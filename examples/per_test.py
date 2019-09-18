"""Demonstration of the power of Prioritised Experience Replay"""
import time
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.algorithms.double_deep_q_learning import DoubleDQNAgent
from src.models import DQNLinear
from src.utils.env import wrap_no_image
from src.utils.logger import Logger
from src.utils.replay_memory import ReplayMemory, PrioritisedMemory


def _moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return list(np.convolve(interval, window, "same"))


CAPACITY = 50_000
SKIP_N = 4

frames = 100_000
TARGET_UPDATE_FREQUENCY = 1000

EPSILON_METHOD = "linear"
EPSILON_FRAMES = int(0.1 * frames)
EPSILON_ARGS = [EPSILON_METHOD, EPSILON_FRAMES]
EPSILON_KWARGS = {"epsilon_min": 0.1}


# ------Env------------------
name = "CartPole-v1"
env = gym.make(name)
env = wrap_no_image(env, skip_n=SKIP_N)

n_inputs = env.observation_space.shape[0]
n_actions = env.action_space.n

# -------------------------------------------------Random memory---------------------------------------------------------
# -------Models--------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQNLinear(n_inputs, n_actions).to(device)
target_model = DQNLinear(n_inputs, n_actions).to(device)
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
agent = DoubleDQNAgent(
    model, target_model, env, memory, logger, False, *EPSILON_ARGS, **EPSILON_KWARGS
)
agent.train(n_frames=frames, C=TARGET_UPDATE_FREQUENCY, render=False, plot_results=False)

replay_rewards = agent.logger._rewards
replay_q = agent.logger._q_values


# -------------------------------------------------Prioritised memory----------------------------------------------------
# -------Models--------------
model = DQNLinear(n_inputs, n_actions).to(device)
target_model = DQNLinear(n_inputs, n_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

memory = PrioritisedMemory(CAPACITY, EPSILON_FRAMES)

# ------Saving and Logging----
env.reset()
logger.clear()

# ------Training--------------
agent = DoubleDQNAgent(
    model, target_model, env, memory, logger, True, *EPSILON_ARGS, **EPSILON_KWARGS
)
agent.train(n_frames=frames, C=TARGET_UPDATE_FREQUENCY, render=False, plot_results=False)

prioritised_rewards = agent.logger._rewards
prioritised_q = agent.logger._q_values


# ---------------------------------------------------Plots---------------------------------------------------------------
q = _moving_average(replay_q, 50)
q_per = _moving_average(prioritised_q, 50)

plt.plot(range(len(q)), q, label="Random memory")
plt.plot(range(len(q_per)), q_per, label="Prioritised memory")

plt.xlabel("Episode")
plt.ylabel("Total episode Q value predictions")
plt.legend()

plt.savefig(os.path.join(save_path, "q_values.png"))
plt.clf()


# Rewards
r = _moving_average(replay_rewards, 50)
r_per = _moving_average(prioritised_rewards, 50)

plt.plot(range(len(r)), r, label="Random memory")
plt.plot(range(len(r_per)), r_per, label="Prioritised memory")

plt.xlabel("Episode")
plt.ylabel("Total episode reward")
plt.legend()

plt.savefig(os.path.join(save_path, "rewards.png"))
