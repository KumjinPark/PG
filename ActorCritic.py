import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder


class Actor:
    def __init__(self, model, dim_obs, n_action, lr):
        self.dim_obs = dim_obs
        self.n_action = n_action
        self.model = model
        self.lr = lr

    def get_policy(self, obs):
        logits = torch.zeros(self.n_action)
        for action in range(self.n_action):
            logits[action] = torch.matmul(torch.tensor(np.append(obs, action)), self.theta)
        # logits = logits + self.bias

        return self.policy(logits)

    def get_action(self, obs):
        pi = self.get_policy(obs)
        action = np.random.choice(self.n_action, p=pi)

        return action

    def update(self, obs, action, dtheta=None):
        if dtheta is None:
            dtheta = self.calculate_dtheta(obs, action)
        self.theta = 0

    def calculate_mean_feat(self, obs, policy=None):
        mean_feat = 0
        if policy is None:
            policy = self.get_policy(obs)

        for action in range(self.n_action):
            phi = np.append(obs, action)
            mean_feat = mean_feat + policy[action]*phi

        return mean_feat

    def calculate_dtheta(self, obs, action, mean_feat=None, policy=None):
        if mean_feat is None:
            mean_feat = self.calculate_mean_feat(obs, policy)
        phi = np.append(obs, action)
        phi = phi - mean_feat

        return phi


class Critic:
    def __init__(self, dim_obs, n_action, lr):
        self.dim_obs = dim_obs
        self.n_action = n_action
        self.w = torch.tensor(np.random.randn(dim_obs+1), requires_grad=True)
        self.lr = lr

    def get_values(self, obs, policy):
        mean_feat = self.calculate_mean_feat(obs, policy)
        dws = self.calculate_dw(obs, 0, mean_feat)
        for b in range(1, self.n_action):
            phi = self.calculate_dw(obs, b, mean_feat)
            dws = torch.cat([dws, phi], dim=1)
        q_values = torch.matmul(dws, self.w)

        return q_values

    def get_value(self, obs, action, policy):
        dw = self.calculate_dw(obs, action, policy=policy)
        q_value = torch.matmul(dw, self.w)

        return q_value

    def calculate_mean_feat(self, obs, policy=None):
        mean_feat = 0
        if policy is None:
            a = np.mean(np.arange(self.n_action))

            return np.append(obs, a)

        for action in range(self.n_action):
            phi = np.append(obs, action)
            mean_feat = mean_feat + policy[action]*phi

        return mean_feat

    def calculate_dw(self, obs, action, mean_feat=None, policy=None):
        if mean_feat is None:
            mean_feat = self.calculate_mean_feat(obs, policy)
        phi = np.append(obs, action)
        phi = phi - mean_feat

        return phi

    def update(self):
        pass


class AC:
    def __init__(self, env, actor, critic, gamma, lr_actor, lr_critic,
                 criterion_actor, criterion_critic, optimizer_actor, optimizer_critic):
        self.env = env
        self.action = env.action_space
        self.phi_dim = env.observation_space.shape[0]
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.criterion_actor = criterion_actor
        self.criterion_critic = criterion_critic
        self.optimizer_actor = optimizer_actor(actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optimizer_critic(critic.parameters(), lr=lr_critic)

    def update(self, obs, action, reward, next_state):
        next_q = self.actor.get_values(obs)
        q_target = reward + self.gamma*torch.max(next_q)
        q_predict = self.actor.get_value(obs, action)
        critic_loss = self.criterion_critic(q_target, q_predict)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()




    def get_action(self, obs):
        action = self.actor.get_action(obs)

        return action


if __name__ == "__main__":
    a = torch.randn((5, 4))
    w = torch.tensor([1., 1., 1., 1.])
    print(torch.matmul(a, w))
