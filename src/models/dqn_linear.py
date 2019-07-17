import torch.nn as nn
import torch.nn.functional as F


class DQNLinear(nn.Module):
    """
    A feedforward, non-convolutional network.
    There is nothing about this architecture which is specific to Deep-q-learning - in fact,
    the algorithm's performance should be fairly robust to the number and sizes of layers.
    """
    def __init__(self, num_inputs, num_actions):
        """
        :param num_inputs: Number of inputs in the openai gym env's state
        :param num_actions: Number of actions in the env
        """
        super(DQNLinear, self).__init__()
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)
