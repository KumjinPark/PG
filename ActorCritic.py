import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from sklearn.preprocessing import OneHotEncoder


class Actor:
    def __init__(self, model, dim_obs, dim_action, lr):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.policy = model
        self.lr = lr

    def get_action(self, obs):
        pi = self.policy(obs)
        pi = pi.detach().numpy()
        if self.policy.type == 'softmax':
            return np.random.choice(self.dim_action, p=pi)
        elif self.policy.type == 'Gaussian':
            return pi

    def step(self, obs, action):
        action_target = action
        if self.policy.type == 'softmax':
            action_target = [0.] * self.dim_obs
            action_target[action] = 1
        action_predict = self.policy(obs)

        return action_target, action_predict

#####################################################################
# Work to do :  Differentiate the value function approximation
#               according to the type of the policy of Critic.
#####################################################################
class Critic:
    def __init__(self, model, dim_obs, dim_action, lr):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.model = model
        self.lr = lr

    def get_value(self, obs, action, policy):
        values = self.model(obs)
        value = values[action]

        return value

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
    def __init__(self, env, actor, critic, architecture, gamma, lr_actor, lr_critic,
                 criterion_actor, criterion_critic, optimizer_actor, optimizer_critic):
        self.env = env
        self.action = env.action_space
        self.phi_dim = env.observation_space.shape[0]
        self.actor = actor
        self.critic = critic
        self.architecture = architecture
        self.gamma = gamma
        self.criterion_actor = criterion_actor
        self.criterion_critic = criterion_critic
        self.optimizer_actor = optimizer_actor(actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optimizer_critic(critic.parameters(), lr=lr_critic)

    def update(self, obs, action, reward, next_obs):
        value_target, value_predict, baseline = self.critic.step(obs, action, reward, next_obs, 
                                                                 self.gamma, self.architecture)
        action_target, action_predict = self.actor.step(obs, action)
        critic_loss = self.criterion_critic(value_predict, value_target)
        actor_loss = self.criterion_actor(action_predict, action_target)*baseline

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

    def get_action(self, obs):
        action = self.actor.get_action(obs)

        return action


if __name__ == "__main__":
    a = torch.randn((5, 4))
    w = torch.tensor([1., 1., 1., 1.])
    print(torch.matmul(a, w))
