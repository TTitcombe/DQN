import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.utils.epsilon_greedy import Epsilon
from src.utils.general_functions import torch_from_frame
from src.utils.replay_memory import Transition


class DQNAgent:
    """
    This class implements a simplified deep-q-learning algorithm.
    State information extracted from the environment is used as model input,
    rather than raw pixels.
    """
    def __init__(self, model, target_model, env, memory, logger, *epsilon_args, **epsilon_kwargs):
        """
        :param model: The model which decides actions and is trained
        :param target_model: The model used in calculating the loss function. is updated to match model every C steps
        :param env: An OpenAI gym environment
        :param memory: A store for the agent's memories
        :type memory: utils.replay_memory.ReplayMemory
        :param logger: An object which stores episode rewards and losses, and can generate charts
        :type logger: utils.logger.Logger
        :param epsilon_args: args to construct our epsilon-greedy policy:
                                    anneal method and number of frames over which to anneal
        :param epsilon_kwargs: kwargs to construct our epsilon-greedy policy:
                                    maximum and minimum values for epsilon
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.target_model = target_model  # Model to use in calculating rewards

        self.env = env
        self.memory = memory
        self.epsilon = Epsilon(*epsilon_args, **epsilon_kwargs)

        self._rewards = []
        self._losses = []

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=2.5e-4)

        self.logger = logger

    def train(self, n_frames=10000, C=100, gamma=0.999, batch_size=32, render=True, clip_rewards=True, skip_n=4):
        # Pre-fill memory up to 5% capacity with random play
        pre_fill_frames = int(self.memory.capacity * 0.5)
        print("Getting {} random memories...".format(pre_fill_frames))
        self._fill_memory_with_random(pre_fill_frames, False, clip_rewards, skip_n)

        print("Starting training...\n")
        frame = 0
        is_done = True
        episode_count = 0

        while frame < n_frames or not is_done:

            # If episode has finished, start a new one
            if is_done:
                episode_count += 1
                """if episode_count % 250 == 0:
                    render = True
                else:
                    render = False"""
                reset_state = torch_from_frame(self.env.reset(), self.device)
                state = torch.cat([reset_state] * skip_n, dim=1)
                if render:
                    self.env.render()

                episode_reward = 0.
                episode_loss = 0.
                is_done = False

            action, reward, is_done, next_state, game_reward = self._act(state, frame, is_done, render,
                                                                         clip_rewards, skip_n)
            self.memory.update(state, action, reward, next_state)
            state = next_state

            # Update
            loss = self._update_model(gamma, batch_size)
            episode_loss += loss
            episode_reward += game_reward
            frame += 1

            if frame % C == 0:
                self._update_target_model()

            if is_done:
                self.logger.update(episode_reward, episode_loss, self.model)

        random_rewards = []
        for episode in range(episode_count):
            random_rewards.append(self._play_random_episode(False, clip_rewards, skip_n))

        print("\nBest reward: {}".format(self.logger.best_reward))

        self.logger.random_rewards = random_rewards

        self.logger.plot_reward(save=True)

    def _fill_memory_with_random(self, n_frames, render, clip_rewards, skip_n):
        frame = 0
        is_done = True
        episode_count = 0

        while frame < n_frames or not is_done:
            # If episode has finished, start a new one
            if is_done:
                episode_count += 1
                reset_state = torch_from_frame(self.env.reset(), self.device)
                state = torch.cat([reset_state] * skip_n, dim=1)
                if render:
                    self.env.render()
                is_done = False

            action, reward, is_done, next_state, game_reward = self._act(state, 0, is_done, render,
                                                                         clip_rewards, skip_n)
            self.memory.update(state, action, reward, next_state)
            state = next_state
            frame += 1

    def _play_random_episode(self, render, clip_rewards, skip_n, update=False):
        reset_state = torch_from_frame(self.env.reset(), self.device)
        state = torch.cat([reset_state] * skip_n, dim=1)
        if render:
            self.env.render()
        is_done = False
        episode_reward = 0.

        while not is_done:
            action, reward, is_done, next_state, game_reward = self._act(state, 0, False, render, clip_rewards, skip_n)

            if update:
                self.memory.update(state, action, reward, next_state)

            state = next_state
            episode_reward += game_reward
        return episode_reward

    def _act(self, state, frame, is_done, render, clip_rewards, skip_n):
        # Select an action
        action = self._select_action(state, self.epsilon(frame))

        next_state = None
        game_reward = 0.
        for _ in range(skip_n):
            if is_done:
                next_state = None
                break
            next_frame, reward, is_done, _ = self.env.step(action.item())
            if render:
                self.env.render()
            game_reward += reward
            next_frame = torch_from_frame(next_frame, self.device)
            next_state = next_frame if next_state is None else torch.cat((next_state, next_frame), dim=1)

        reward = np.sign(game_reward) if clip_rewards else game_reward
        reward = torch.tensor([[reward]], device=self.device)

        return action, reward, is_done, next_state, game_reward

    def _select_action(self, state, epsilon):
        if random.random() < epsilon:
            # Select random action
            action = self.env.action_space.sample()
            action = torch.tensor([[action]], device=self.device)
        else:
            # Select model's best action
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.model(state).max(1)[1].view(1, 1)
        return action

    def _update_model(self, gamma, batch_size):
        if len(self.memory) > batch_size:
            samples = self.memory.sample(batch_size)
            samples = Transition(*zip(*samples))

            # Get y
            rewards = torch.cat(samples.reward)
            non_terminal_indices = torch.tensor(tuple(map(lambda s: s is not None, samples.next_state)),
                                                device=self.device, dtype=torch.uint8)
            next_states = torch.cat([ns for ns in samples.next_state if ns is not None])
            max_q = self.target_model(next_states).max(1)[0].detach()
            # print(max_q.shape)
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

    def _update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
