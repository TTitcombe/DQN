"""
Replay memory, adapted from
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

from collections import deque, namedtuple
import random

import numpy as np

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class ReplayMemory:
    def __init__(self, capacity):
        """
        Initialise replay memory
        :param capacity: Maximum number of memories to store
        :type capacity: int
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add_memory(self, *args):
        """
        Add a memory
        :param args: state, action, reward, next_state
        """
        self.memory.append(Transition(*args))

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


class PrioritisedMemory(ReplayMemory):
    def __init__(self, capacity, beta_anneal_frames=None, alpha=0.6, beta=0.4):
        super(PrioritisedMemory, self).__init__(capacity)

        self.alpha = alpha
        self.beta = beta

        if not beta_anneal_frames:
            beta_anneal_frames = int(capacity * 0.1)
        self.beta_anneal = (1.0 - self.beta) / beta_anneal_frames

        self.current_node = 0
        self.capacity = capacity

        self.p_tree = np.zeros(2 * capacity - 1)
        self.memory = []

        self.full_memory = False

    def add_memory(self, *args):
        # Td-error defaults to 1 for new memories

        self._update_p(self.capacity + self.current_node - 1, 1)
        if self.full_memory:
            self.memory[self.current_node] = Transition(*args)
        else:
            self.memory.append(Transition(*args))

        self.current_node += 1

        # If we've got to the end of the memory, start from the beginning
        if self.current_node == self.capacity:
            self.full_memory = True
            self.current_node = 0

    def _update_p(self, index, p):
        """Update the td error of a node and propagate it through the tree"""
        difference = p - self.p_tree[index]
        self._update_node(index, difference)

    def _update_node(self, index, difference):
        self.p_tree[index] = self.p_tree[index] + difference

        if index > 0:
            self._update_node((index - 1) // 2, difference)

    def update(self, indices, td_errors):
        for index, td_error in zip(indices, td_errors):
            self.update_memory(index, float(td_error))

    def update_memory(self, index, td_error):
        td_error = abs(td_error) ** self.alpha
        self._update_p(index, td_error ** self.alpha)

    def sample(self, n):
        indices = []
        w_values = []
        memories = []
        p_range = self.p_tree[0] / n

        if len(self.memory) == 0:
            p_min = 0.
        else:
            up_to_point = min(-1, -self.capacity + len(self.memory))
            p_min = np.min(self.p_tree[-self.capacity:up_to_point])
        w_max = p_min**self.beta

        p_start = 0
        for _ in range(n):
            p_end = p_start + p_range

            p_selection = random.uniform(p_start, p_end)
            index, p, memory = self._sample_p(p_selection)
            indices.append(index)
            w_values.append(w_max / p**self.beta)
            memories.append(memory)

            p_start = p_end

        self.beta += self.beta_anneal

        return indices, w_values, memories

    def _sample_p(self, total_p):
        index = self._retrieve_index(0, total_p)
        p = self.p_tree[index]
        memory = self.memory[index + 1 - self.capacity]
        return index, p, memory

    def _retrieve_index(self, index, total_p):
        left_index = 2 * index + 1

        if left_index >= (2 * self.capacity - 1):
            # If there is no leaf node, we're at the leaf node!
            return index

        left_p = self.p_tree[left_index]

        if left_p >= total_p:
            return self._retrieve_index(left_index, total_p)
        else:
            return self._retrieve_index(left_index + 1, total_p - left_p)


if __name__ == "__main__":
    # Test the PER
    per = PrioritisedMemory(4)
    for i in range(4):
        per.add_memory(1, 2, 3, 4)
    td_errors = [1, 2, 3, 4]
    per.update([3, 4, 5, 6], td_errors)

    # The nodes of the memory i.e. from 0 to capacity-1
    counts = {3: 0, 4: 0, 5: 0, 6: 0}
    samples = 100000
    for _ in range(samples):
        memories = per.sample(1)[0]
        counts[memories[0]] += 1
    print("Node index: actual calls | expected calls")
    for k, v in counts.items():
        print("{}: {} | {}".format(k, v, (k - 2) * samples / sum(td_errors)))
