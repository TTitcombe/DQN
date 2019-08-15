import cv2
import gym
import torch

from src.utils.general_functions import torch_from_frame


class AgentEvaluation:
    def __init__(self, model, path, device):
        self.checkpoint = torch.load(path)
        self.device = device

        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.eval()
        self.model = model

        self.env = None

    def play(self, n, env, skip_n=4, render=True):
        episodes_played = 0
        rewards = []
        is_done = True
        while episodes_played < n:
            if is_done:
                state = torch_from_frame(env.reset(), self.device)
                if render:
                    env.render()
                state = torch.cat([state] * 4, dim=1)
                is_done = False
                episode_reward = 0.0

            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            next_state = None
            for _ in range(skip_n):
                new_state, reward, is_done, _ = env.step(action.item())
                if render:
                    env.render()
                new_state = torch_from_frame(new_state, self.device)
                episode_reward += reward
                if is_done:
                    break
                next_state = (
                    new_state
                    if next_state is None
                    else torch.cat((next_state, new_state), dim=1)
                )
            state = next_state

            if is_done:
                rewards.append(episode_reward)
                episodes_played += 1
        return rewards

    def record(self, env, path):
        wrapped_env = gym.wrappers.Monitor(env, path, force=True)
        self.play(1, wrapped_env, render=False)
        wrapped_env.close()
        env.close()


class AtariEvaluator(AgentEvaluation):
    def play(self, n, env, skip_n=4, render=True):
        episodes_played = 0
        rewards = []
        is_done = True
        self.env = env
        while episodes_played < n:
            if is_done:
                self.env.reset()
                state = self._get_state_from_frame(None)

                if render:
                    self.env.render()
                state = torch.cat([state] * 4, dim=1)
                is_done = False
                episode_reward = 0.0

            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            next_state = None
            for _ in range(skip_n):
                new_state, reward, is_done, _ = env.step(action.item())
                if render:
                    env.render()
                new_state = self._get_state_from_frame(None)
                episode_reward += reward
                if is_done:
                    break
                next_state = (
                    new_state
                    if next_state is None
                    else torch.cat((next_state, new_state), dim=1)
                )
            state = next_state

            if is_done:
                rewards.append(episode_reward)
                episodes_played += 1
        return rewards

    def _get_state_from_frame(self, _):
        # Get the frame
        frame = self.env.render(mode="rgb_array")

        # Resize
        frame = cv2.resize(frame, (84, 84))

        # Transpose it to channels x height x width
        frame = frame.transpose((2, 0, 1))

        # Convert to greyscale
        # Taken from https://github.com/ttaoREtw/Flappy-Bird-Double-DQN-Pytorch/blob/master/env.py
        luma = [0.2989, 0.5870, 0.1140]
        frame = (
            luma[0] * frame[0, :, :]
            + luma[1] * frame[1, :, :]
            + luma[2] * frame[2, :, :]
        )

        # Scale
        frame = frame.astype("float32") / 255.0

        # Convert to torch tensor and add a batch dimension
        return torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(self.device)
