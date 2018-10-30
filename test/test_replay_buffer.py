import numpy as np
import torch

from replay_buffer import ReplayBuffer


def test_replay_buffer(device):
    replay_buffer = ReplayBuffer(device, 10, 5, 10)

    for i in range(10):
        replay_buffer.add(np.zeros(10), 0, 0.0, np.zeros(10), 0)

    states, actions, rewards, next_states, dones = replay_buffer.sample(5)
    assert (5, 10) == states.shape
    assert (5, 5) == actions.shape
    assert (5, 1) == rewards.shape
    assert (5, 10) == next_states.shape
    assert (5, 1) == dones.shape

    for i in range(10):
        replay_buffer.add(np.zeros(10), 0, 0.0, np.zeros(10), 0)

    states, actions, rewards, next_states, dones = replay_buffer.sample(5)
    assert (5, 10) == states.shape
    assert (5, 5) == actions.shape
    assert (5, 1) == rewards.shape
    assert (5, 10) == next_states.shape
    assert (5, 1) == dones.shape

