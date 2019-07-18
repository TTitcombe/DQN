import matplotlib.pyplot as plt
import numpy as np


class Epsilon:
    def __init__(self, anneal_method, n_frames, epsilon_max=1.0, epsilon_min=0.1):
        if anneal_method.lower() == "linear":
            self._get_epsilon = self.linear
        elif anneal_method.lower() == "exp":
            self._get_epsilon = self.exp
        else:
            raise NotImplementedError("anneal_method must be linear or exp")

        self.n_frames = n_frames
        self.epsilon_min = max(0.0, epsilon_min)
        self.epsilon_max = min(1.0, epsilon_max)

    def __call__(self, frame):
        return self._get_epsilon(frame)

    def linear(self, frame):
        return max(self.epsilon_min, self.epsilon_max - (self.epsilon_max - self.epsilon_min) * frame / self.n_frames)

    def exp(self, frame):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-1 * frame / self.n_frames)

    def plot_epsilon(self):
        x = np.linspace(self.epsilon_min, self.epsilon_max, self.n_frames)
        y = [self(_x) for _x in x]
        plt.plot(x, y)
        plt.xlabel("Frame")
        plt.ylabel("Epsilon")
        plt.show()