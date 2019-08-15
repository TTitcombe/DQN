import matplotlib.pyplot as plt
import numpy as np


class Epsilon:
    def __init__(
        self, anneal_method, n_frames, epsilon_max=1.0, epsilon_min=0.1, anneal_delay=0
    ):
        anneal_method = anneal_method.lower()
        if anneal_method == "linear":
            self._get_epsilon = self.linear
        elif anneal_method == "exp":
            self._get_epsilon = self.exp
        elif anneal_method == "inverse_sigmoid":
            self._get_epsilon = self.inverse_sigmoid
        else:
            raise NotImplementedError(
                "anneal_method must be linear, exp, or inverse_sigmoid"
            )

        self.n_frames = n_frames + anneal_delay
        self.anneal_delay = anneal_delay
        self.epsilon_min = max(0.0, epsilon_min)
        self.epsilon_max = min(1.0, epsilon_max)

    def __call__(self, frame):
        frame = max(0, frame - self.anneal_delay)
        return self._get_epsilon(frame)

    def linear(self, frame):
        return max(
            self.epsilon_min,
            self.epsilon_max
            - (self.epsilon_max - self.epsilon_min) * frame / self.n_frames,
        )

    def exp(self, frame):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
            -1 * frame / self.n_frames
        )

    def inverse_sigmoid(self, frame):
        raise NotImplementedError("Inverse sigmoid has not yet been implemented")

    def plot_epsilon(self):
        x = range(self.n_frames)
        y = [self(_x) for _x in x]
        plt.plot(x, y)
        plt.xlabel("Frame")
        plt.ylabel("Epsilon")
        plt.show()
