from collections import namedtuple
import random

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class ReplayMemory:
    def __init__(self, capacity):
        """
        Initialise replay memory
        :param capacity: Maximum number of memories to store
        :type capacity: int
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def update(self, *args):
        """
        Add a memory
        :param args: state, action, reward, next_state
        """
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)

        self.position = (self.position + 1) % self.capacity

    def sample(self, n):
        """
        Take a random selection of memories
        :param n: Number of memories to take
        :type n: int
        :return: a random selection of n memories
        """
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)
