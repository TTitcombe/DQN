"""A basic example showing how to train a DQN agent to play Breakout from pixel information"""
import gym
import numpy as np
import os
import torch

from src.algorithms.double_deep_q_learning import DoubleDQNAtariAgent
from src.models import DDQN
from src.utils.assessment import AtariEvaluator
from src.utils.env import DiscreteCarRacing, wrap_deepmind
from src.utils.logger import Logger
from src.utils.replay_memory import ReplayMemory


def _moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return list(np.convolve(interval, window, "same"))


# -------Parameters----------
CAPACITY = 50_000
SKIP_N = 4

frames = 100_000
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

# -------Models--------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDQN(SKIP_N, 84, n_actions).to(device)
target_model = DDQN(SKIP_N, 84, n_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

memory = ReplayMemory(CAPACITY)

# ------Saving and Logging----
save_path = os.path.join("results", "models", name)

logger = Logger(
    save_path,
    save_best=True,
    save_every=50,
    log_every=25,
    C=TARGET_UPDATE_FREQUENCY,
    capacity=CAPACITY,
)

# ------Training--------------
agent = DoubleDQNAtariAgent(
    model, target_model, env, memory, logger, *EPSILON_ARGS, **EPSILON_KWARGS
)
agent.train(n_frames=frames, C=TARGET_UPDATE_FREQUENCY, render=False)
# This saves a model to results/models/CarRacing-v0


# ------Evaluating------------
evaluator = AtariEvaluator(model, os.path.join(save_path, "best_model.pth"), device)

# Play once
evaluator.record(env, os.path.join("results", "videos", name))

# Get average score
scores = evaluator.play(10, env, render=False)
print(
    "{:.3f} +/- {:.1f}".format(np.mean(scores), np.std(scores) / np.sqrt(len(scores)))
)
