import torch
import torch.nn.functional as F

from .deep_q_learning import DQNAgent, DQNAtariAgent
from src.utils.replay_memory import Transition


class DoubleDQNAgent(DQNAgent):
    """
    This class implements a simplified double-deep-q-learning algorithm.
    State information extracted from the environment is used as model input,
    rather than raw pixels.

    The only change from the DQNAgent is that Q estimates in the loss function are
    calculated by taking the best action as predicted by the model; normal DQN takes the
    greatest Q value calculated by the target network.
    """
    def _update_model(self, gamma, batch_size):
        if len(self.memory) > batch_size:
            samples = self.memory.sample(batch_size)
            samples = Transition(*zip(*samples))

            # Get y
            rewards = torch.cat(samples.reward)
            non_terminal_indices = torch.tensor(tuple(map(lambda s: s is not None, samples.next_state)),
                                                device=self.device, dtype=torch.uint8)
            next_states = torch.cat([ns for ns in samples.next_state if ns is not None])

            # In Double Deep Q Learning, we use the Q value attached to the best value from the model
            selected_actions = self.model(next_states).max(1)[1].detach().unsqueeze(1)
            max_q = self.target_model(next_states).gather(1, selected_actions).squeeze(1)

            additional_qs = torch.zeros(batch_size, device=self.device)
            additional_qs[non_terminal_indices] = max_q
            y = rewards + additional_qs.unsqueeze(1) * gamma

            # get Q for each action we took in states
            actions = torch.cat(samples.action)
            q = self.model(torch.cat(samples.state)).gather(1, actions)

            # Update the model
            loss = F.mse_loss(y, q)
            self._losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            return loss.item()
        else:
            return 0.


class DoubleDQNAtariAgent(DQNAtariAgent, DoubleDQNAgent):
    """
    This class is used to perform double deep q learning on an Atari environment
    (using pixel, not state, information to make decisions).
    It inherits the atari processing methods from DQNAtariAgent and
    Double deep q learning methods from DoubleDQNAgent
    """
    pass
