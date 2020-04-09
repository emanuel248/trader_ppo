
import gym
import random
from kaggle_environments import evaluate, make

class ConnectX(gym.Env):
    def __init__(self, switch_prob=0.5, use_random_training=True, random_agent=False, test_mode=False):
        self.env = make('connectx', debug=True)
        if use_random_training:
            if random.uniform(0, 1) < 0.6:
                self.pair = [None, 'negamax']
                print('create negamax agent')
            else:
                self.pair = [None, 'random']
                print('create random agent')
        else:
            self.pair = [None, 'negamax']

        #test setup
        if random_agent and test_mode:
            self.pair = [None, 'random']
        elif test_mode:
            self.pair = [None, 'negamax']

        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob
        
        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)

    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)

    def step(self, action):
        state, reward, done, info = self.trainer.step(int(action))
        reward = reward if reward==1 or reward==-1 else 0.5
        return state, reward, done, info
    
    def reset(self):
        if random.uniform(0, 1) < self.switch_prob:
            self.switch_trainer()
        return self.trainer.reset()
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
