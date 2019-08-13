"""A basic example showing how to train a DQN agent to play Breakout from pixel information"""

import gym
import numpy as np
import os
import torch

from src.algorithms.double_deep_q_learning import DoubleDQNAtariAgent
from src.models import DDQN
from src.utils.assessment import AtariEvaluator
from src.utils.logger import Logger
from src.utils.replay_memory import ReplayMemory


def _moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return list(np.convolve(interval, window, 'same'))


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


# -------Parameters----------
CAPACITY = 100_000
SKIP_N = 4

frames = 1_000_000
TARGET_UPDATE_FREQUENCY = 10_000

EPSILON_METHOD = "linear"
EPSILON_FRAMES = int(0.1 * frames)
EPSILON_ARGS = [EPSILON_METHOD, EPSILON_FRAMES]
EPSILON_KWARGS = {"epsilon_min": 0.05}


# ------Env------------------
env = gym.make("Breakout-v4")
env = NoopResetEnv(env)
env = EpisodicLifeEnv(env)
if 'FIRE' in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)
n_actions = env.action_space.n

# -------Models--------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDQN(SKIP_N, 84, n_actions).to(device)
target_model = DDQN(SKIP_N, 84, n_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

memory = ReplayMemory(CAPACITY)

# ------Saving and Logging---
name = ("Breakout" + "_{}capacity".format(CAPACITY) +
        "_{}{}EPSILON".format(EPSILON_FRAMES, EPSILON_METHOD) + "_{}C".format(TARGET_UPDATE_FREQUENCY))
save_path = os.path.join("results", "models", name)

logger = Logger(save_path, save_best=True, save_every=10,
                log_every=5, C=TARGET_UPDATE_FREQUENCY, capacity=CAPACITY)

# ------Training------------

"""agent = DoubleDQNAtariAgent(model, target_model, env, memory, logger, *EPSILON_ARGS, **EPSILON_KWARGS)
agent.train(n_frames=frames, C=TARGET_UPDATE_FREQUENCY, render=False)"""
# This saves a model to results/models/Breakout.....


# ------Evaluating----------
evaluator = AtariEvaluator(model, os.path.join(save_path, "best_model.pth"), device)
# Play once
evaluator.record(env, os.path.join("results", "videos", name))
# Get average score
#scores = evaluator.play(100, env, render=False)
#print("{:.3f} +/- {:.1f}".format(np.mean(scores), np.std(scores) / np.sqrt(len(scores))))
