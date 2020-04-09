import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=256, std=0.0, categoric=False):
        super(ActorCritic, self).__init__()

        self.categoric = categoric
        self.num_outputs = num_outputs

        self.critic = nn.Sequential(
            nn.Linear(num_inputs[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs)
        )
        
        #random factor
        self.log_std = nn.Parameter(T.ones(1, num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)

        #mask out invalid actions, lower 
        #neg_mask = (x[:,:self.num_outputs]>0)
        #mu[neg_mask] = float('-inf')

        mu = F.softmax(mu, dim=-1)
        if not self.categoric:
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
        else:
            dist = Categorical(mu)
        
        return dist, value