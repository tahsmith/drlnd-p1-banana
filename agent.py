import random

import torch
import torch.optim
import numpy as np

from q_network import QNetwork
from replay_buffer import ReplayBuffer


class Agent:
    def __init__(self, device, state_size, action_size, buffer_size=10,
                 batch_size=10,
                 learning_rate=0.1,
                 discount_rate=0.99,
                 eps_decay=0.9,
                 tau=0.1,
                 steps_per_update=4
                 ):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size

        self.q_network_control = QNetwork(state_size, action_size).to(device)
        self.q_network_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.q_network_control.parameters(),
                                          lr=learning_rate)

        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(device, state_size, action_size,
                                          buffer_size)

        self.discount_rate = discount_rate

        self.eps = 1.0
        self.eps_decay = eps_decay

        self.tau = tau

        self.step_count = 0
        self.steps_per_update = steps_per_update

    def policy(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return self.epsilon_greedy_policy(self.eps, state)

    def epsilon_greedy_policy(self, eps, state):
        self.q_network_control.eval()
        with torch.no_grad():
            action_values = self.q_network_control(state)
        self.q_network_control.train()

        if random.random() > eps:
            greedy_choice = np.argmax(action_values.cpu().data.numpy())
            return greedy_choice
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        p = self.calculate_p(state, action, reward, next_state, done)
        self.replay_buffer.add(state, action, reward, next_state, done, p)
        if self.step_count % self.steps_per_update == 0:
            self.learn()
        self.step_count += 1

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, p = \
            self.replay_buffer.sample(self.batch_size)

        error = self.bellman_eqn_error(
            states, actions, rewards, next_states, dones)
        importance_scaling = (self.replay_buffer.buffer_size * p) ** -1
        loss = (importance_scaling * (error ** 2)).sum() / self.batch_size
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target()

    def bellman_eqn_error(self, states, actions, rewards, next_states, dones):
        """Double DQN error - use the control network to get the best action
        and apply the target network to it to get the target reward which is
        used for the bellman eqn error.
        """
        self.q_network_control.eval()
        with torch.no_grad():
            a_max = self.q_network_control(next_states).argmax(1).unsqueeze(1)

        target_action_values = self.q_network_target(next_states).gather(1,
                                                                         a_max)
        target_rewards = rewards + self.discount_rate * (1 - dones) \
                         * target_action_values

        self.q_network_control.train()
        current_rewards = self.q_network_control(states).gather(1, actions)
        error = current_rewards - target_rewards
        return error

    def calculate_p(self, state, action, reward, next_state, done):
        next_state = torch.from_numpy(next_state[np.newaxis, :]).float().to(
            self.device)
        state = torch.from_numpy(state[np.newaxis, :]).float().to(self.device)
        action = torch.from_numpy(np.array([[action]])).long().to(self.device)
        reward = torch.from_numpy(np.array([reward])).float().to(self.device)
        done = torch.from_numpy(np.array([[done]], dtype=np.uint8)).float().to(
            self.device)

        return abs(self.bellman_eqn_error(state, action, reward, next_state,
                                          done)) + 1e-3

    def update_target(self):
        for target_param, control_param in zip(
                self.q_network_target.parameters(),
                self.q_network_control.parameters()):
            target_param.data.copy_(
                self.tau * control_param.data + (1.0 - self.tau) *
                target_param.data)

    def end_of_episode(self):
        self.eps *= self.eps_decay
        self.step_count = 0

    def save(self, path):
        torch.save(self.q_network_control.state_dict(), path)

    def restore(self, path):
        self.q_network_control.load_state_dict(torch.load(path))
