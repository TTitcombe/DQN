import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    A convolutional network.
    Architecture as outlined in the methods section of
    "Human-level control through deep reinforcement learning" - Mnih et. al
    There is nothing about this architecture which is specific to Deep-q-learning - in fact,
    the algorithm's performance should be fairly robust to the number and sizes of layers.
    """

    def __init__(self, input_channels, input_size, output_size):
        """
        Initialise the layers of the DQN
        :param input_channels: number of input channels (usually 4)
        :param input_size: width/height of the input image (we assume it's square)
        :param output_size: number out elements in the output vector
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the image when squashed to a linear vector
        # We assume here that the input image is square
        conv_length = self._conv_shape(
            self._conv_shape(self._conv_shape(input_size, 4, 4), 4, 2), 3, 1
        )
        conv_shape = conv_length ** 2 * 64
        self.linear1 = nn.Linear(conv_shape, 512)
        self.linear2 = nn.Linear(512, output_size)

    @staticmethod
    def _conv_shape(input_size, filter_size, stride, padding=0):
        return 1 + (input_size - filter_size + 2 * padding) // stride

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear1(x.view(x.size(0), -1)))
        return self.linear2(x)


class DDQN(DQN):
    """Dueling DQN. We inherit from DQN to get _conv_shape, but overwrite init and forward."""

    def __init__(self, input_channels, input_size, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the image when squashed to a linear vector
        # We assume here that the input image is square
        conv_length = self._conv_shape(
            self._conv_shape(self._conv_shape(input_size, 4, 4), 4, 2), 3, 1
        )
        conv_shape = conv_length ** 2 * 64

        self.linear1 = nn.Linear(conv_shape, 512)

        # The linear layers for the action stream
        self.action1 = nn.Linear(512, 256)
        self.action2 = nn.Linear(256, output_size)

        # The linear layers for the state stream
        self.state1 = nn.Linear(512, 256)
        self.state2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))

        # Action stream
        x_action = F.relu(self.action1(x))
        x_action = self.action2(x_action)
        x_action = x_action - torch.mean(x_action)

        # State stream
        x_state = F.relu(self.state1(x))
        x_state = self.state2(x_state)

        return x_action + x_state
