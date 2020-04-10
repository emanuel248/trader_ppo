import numpy as np

class MultiEnv:
    def __init__(self, envs):
        self.envs = [e() for e in envs]

    def reset(self):
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions):
        results = [env.step(ac) for env, ac in zip(self.envs, actions)]
	
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos