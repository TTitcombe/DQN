"""A basic example showing how to train a DoubleDQNAgent to play CartPoleSwingUp"""
import os

from cartpoleswingup import CartPoleSwingUpEnv
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.algorithms.double_deep_q_learning import DoubleDQNAgent
from src.models import DQNLinear
from src.utils.assessment import AgentEvaluation
from src.utils.logger import Logger
from src.utils.replay_memory import ReplayMemory


def _moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return list(np.convolve(interval, window, 'same'))


# -------Parameters----------
CAPACITY = 10000
SKIP_N = 4

EPSILON_METHOD = "linear"
EPSILON_FRAMES = 10000
EPSILON_ARGS = [EPSILON_METHOD, EPSILON_FRAMES]
C = 1000
frames = 50000

# -------Models--------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQNLinear(5 * SKIP_N, 2).to(device)
target_model = DQNLinear(5 * SKIP_N, 2).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

env = CartPoleSwingUpEnv()
memory = ReplayMemory(CAPACITY)

name = ("CartPoleSwingUp" + "_{}capacity".format(CAPACITY) +
        "_{}{}EPSILON".format(EPSILON_FRAMES, EPSILON_METHOD) + "_{}C".format(C))
save_path = os.path.join("results", "models", name)

logger = Logger(save_path, save_best=False,
                save_every=np.inf, C=C, capacity=CAPACITY, frames=frames, anneal=EPSILON_FRAMES)

# ------Training------------
agent = DoubleDQNAgent(model, target_model, env, memory, logger, *EPSILON_ARGS)
agent.train(n_frames=frames, C=C, render=False)
# This saves a model to results/models/CartPoleSwingUp.....

good_results = logger._rewards
good_results = _moving_average(good_results, 50)
logger.clear()

model = DQNLinear(5*SKIP_N, 2).to(device)
target_model = DQNLinear(5 * SKIP_N, 2).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

env = CartPoleSwingUpEnv()
agent = DoubleDQNAgent(model, target_model, env, ReplayMemory(1000), logger, *EPSILON_ARGS)
agent.train(n_frames=frames, C=C, render=False)
short_memory = logger._rewards
short_memory = _moving_average(short_memory, 50)
logger.clear()

model = DQNLinear(5*SKIP_N, 2).to(device)
target_model = DQNLinear(5 * SKIP_N, 2).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

env = CartPoleSwingUpEnv()
agent = DoubleDQNAgent(model, target_model, env, ReplayMemory(10000), logger, *EPSILON_ARGS)
agent.train(n_frames=frames, C=1, render=False)
quick_update = logger._rewards
quick_update = _moving_average(quick_update, 50)
logger.clear()

model = DQNLinear(5*SKIP_N, 2).to(device)
target_model = DQNLinear(5 * SKIP_N, 2).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

env = CartPoleSwingUpEnv()
agent = DoubleDQNAgent(model, target_model, env, ReplayMemory(10000), logger, *["linear", 1000])
agent.train(n_frames=frames, C=C, render=False)
quick_anneal = logger._rewards
quick_anneal = _moving_average(quick_anneal, 50)
logger.clear()

model = DQNLinear(5, 2).to(device)
target_model = DQNLinear(5, 2).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

env = CartPoleSwingUpEnv()
agent = DoubleDQNAgent(model, target_model, env, ReplayMemory(10000), logger, *EPSILON_ARGS)
agent.train(n_frames=frames, C=C, render=False, skip_n=1)
no_history = logger._rewards
no_history = _moving_average(no_history, 50)
logger.clear()

model = DQNLinear(5*SKIP_N, 2).to(device)
target_model = DQNLinear(5 * SKIP_N, 2).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

env = CartPoleSwingUpEnv()
agent = DoubleDQNAgent(model, target_model, env, ReplayMemory(10000), logger, *EPSILON_ARGS)
agent.train(n_frames=frames, C=C, render=False, pre_fill_memory=False)
no_init_memory = logger._rewards
no_init_memory = _moving_average(no_init_memory, 50)
logger.clear()


good_episodes = len(good_results)

plt.plot(range(len(good_results)), good_results, label="Double DQN")

if len(short_memory) > good_episodes:
    short_memory = short_memory[:good_episodes]
plt.plot(range(len(short_memory)), short_memory, label="Small memory")

if len(quick_update) > good_episodes:
    quick_update = quick_update[:good_episodes]
plt.plot(range(len(quick_update)), quick_update, label="Frequent updates")

if len(quick_anneal) > good_episodes:
    quick_anneal = quick_anneal[:good_episodes]
plt.plot(range(len(quick_anneal)), quick_anneal, label="Quick epsilon anneal")

if len(no_history) > good_episodes:
    no_history = no_history[:good_episodes]
plt.plot(range(len(no_history)), no_history, label="Single frame")

if len(no_init_memory) > good_episodes:
    no_init_memory = no_init_memory[:good_episodes]
plt.plot(range(len(no_init_memory)), no_init_memory, label="No memory initialisation")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Episode reward (smoothed 50)")
plt.savefig("all_rewards.png")
plt.show()

""""# ------Evaluating----------
evaluator = AgentEvaluation(model, save_path, device)
# Play once
evaluator.play(1, env)
# Get average score
scores = evaluator.play(100, env, render=False)
print("{:.3f} +/- {:.1f}".format(np.mean(scores), np.std(scores) / np.sqrt(len(scores))))"""
