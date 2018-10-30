from unittest.mock import MagicMock

import pytest

from banana_env import banana_episode


@pytest.fixture
def env():
    mock_env = MagicMock()
    mock_env.step.return_value = MagicMock()

    return mock_env


@pytest.fixture
def agent():
    mock_agent = MagicMock()
    mock_agent.policy.return_value = 1
    return mock_agent


def test_banana_episode(env, agent):
    banana_episode(env, agent, 'brain')
