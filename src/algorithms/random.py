class RandomAgent:
    def __init__(self, env):
        self.env = env

    def play(self, n_episodes, render=False):
        episode = 0
        while episode < n_episodes:
            is_done = False
            self.env.reset()
            while not is_done:
                _, _, is_done, _ = self.env.step(self.env.action_space.sample())
                print("playing")
                if render:
                    self.env.render()
            episode += 1
