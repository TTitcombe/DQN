import torch.nn as nn
import torch.nn.functional as F


class DQNLinear(nn.Module):
    def __init__(self, num_inputs, num_actions):
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
