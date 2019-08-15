import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from torch import save as tsave

from .general_functions import create_dir


class Logger:
    def __init__(
        self,
        save_path="",
        save_every=100,
        save_best=False,
        log_every=50,
        log_style="block",
        **training_kwargs
    ):
        self.save_path = save_path
        create_dir(save_path)

        self._save_every = save_every

        self._save_best = save_best
        self.best_reward = -np.inf

        self._rewards = []
        self._losses = []

        self._block_rewards = []
        self._block_losses = []

        self.random_rewards = []

        self.episode = 0
        self.log_every = log_every
        self.log_style = log_style

        self.training_kwargs = training_kwargs

    def clear(self):
        self._rewards = []
        self._losses = []
        self._block_rewards = []
        self._block_losses = []
        self.random_rewards = []
        self.episode = 0

    def update(self, reward, loss, model):
        self.episode += 1
        self._block_rewards.append(reward)
        self._block_losses.append(loss)

        if self.episode % self.log_every == 0:
            self.report()

        if self._save_best and reward > self.best_reward:
            self.best_reward = reward
            self.training_kwargs.update({"model_state_dict": model.state_dict()})
            tsave(self.training_kwargs, os.path.join(self.save_path, "best_model.pth"))

        if self.episode % self._save_every == 0:
            tsave(model, os.path.join(self.save_path, "{}.pth").format(self.episode))

    def report(self):
        if self.log_style == "continuous":
            losses = self._losses
            rewards = self._rewards
        elif self.log_style == "block":
            losses = self._block_losses
            rewards = self._block_rewards

            self._losses.extend(self._block_losses)
            self._block_losses = []
            self._rewards.extend(self._block_rewards)
            self._block_rewards = []
        else:
            raise RuntimeError("Log style must be 'continuous' or 'block'")

        mean_loss = np.mean(losses)
        se_loss = np.std(losses) / np.sqrt(len(losses))

        mean_reward = np.mean(rewards)
        se_reward = np.std(rewards) / np.sqrt(len(rewards))

        print("\nEpisode {}".format(self.episode))
        print("Loss: {:.3f} +/- {:.1f}".format(mean_loss, se_loss))
        print("Reward: {:.3f} +/- {:.1f}".format(mean_reward, se_reward))

    def save_data(self):
        with open(os.path.join(self.save_path, "temp_rewards.pkl"), "wb") as f:
            pickle.dump(self._rewards, f)
        with open(os.path.join(self.save_path, "temp_losses.pkl"), "wb") as f:
            pickle.dump(self._losses, f)

    def save_model(self, model, name):
        if not name.endswith(".pth"):
            name += ".pth"
        tsave(model, os.path.join(self.save_path, name))

    def plot_reward(self, sliding_window=50, show=False, save=False):
        if self.random_rewards:
            random_rewards = self._moving_average(self.random_rewards, sliding_window)
            plt.plot(range(len(random_rewards)), random_rewards, label="Random actions")

        rewards = self._moving_average(self._rewards, sliding_window)
        plt.plot(range(len(rewards)), rewards, label="DQN")

        plt.xlabel("Episode")
        plt.ylabel("Total episode reward")
        plt.legend()

        if save:
            plt.savefig(os.path.join(self.save_path, "rewards.png"))

        if show:
            plt.show()

    def plot_reward_continuous(self, show=False, save=False):
        if self.random_rewards:
            plt.plot(
                range(len(self.random_rewards)),
                np.cumsum(self.random_rewards),
                label="Random actions",
            )
        plt.plot(range(len(self._rewards)), np.cumsum(self._rewards), label="DQN")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward")
        plt.legend()

        if save:
            plt.savefig(os.path.join(self.save_path, "cumulative_rewards.png"))

        if show:
            plt.show()

    @staticmethod
    def _moving_average(interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, "same")
