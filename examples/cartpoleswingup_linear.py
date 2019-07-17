"""A basic example showing how to train a DQNAgent to play CartPoleSwingUp"""

from CartPoleSwingUp import CartPoleSwingUpEnv
import numpy as np
import os
import torch

from src.algorithms.deep_q_learning import DQNAgent
from src.models.dqn_linear import DQNLinear
from src.utils.assessment import AgentEvaluation
from src.utils.logger import Logger
from src.utils.replay_memory import ReplayMemory


# -------Parameters----------
CAPACITY = 10000
SKIP_N = 4

EPSILON_METHOD = "linear"
EPSILON_FRAMES = 10000
EPSILON_ARGS = [EPSILON_METHOD, EPSILON_FRAMES]
C = 1000
frames = 100000

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

logger = Logger(save_path, save_best=True,
                save_every=np.inf, C=C, capacity=CAPACITY, frames=frames, anneal=EPSILON_FRAMES)

# ------Training------------
agent = DQNAgent(model, target_model, env, memory, logger, *EPSILON_ARGS)
agent.train(n_frames=frames, C=C, render=False)
# This saves a model to results/models/CartPoleSwingUp.....

# ------Evaluating----------
evaluator = AgentEvaluation(model, save_path, device)
# Play once
evaluator.play(1, env)
# Get average score
scores = evaluator.play(100, env, render=False)
print("{:.3f} +/- {:.1f}".format(np.mean(scores), np.std(scores) / np.sqrt(len(scores))))
