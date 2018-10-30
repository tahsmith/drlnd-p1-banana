import platform
from contextlib import contextmanager

from unityagents import UnityEnvironment

from agent import Agent

if platform.system() == 'Darwin':
    BANANA_APP = './Banana.app'
else:
    BANANA_APP = './Banana_Linux_NoVis/Banana.x86_64'


@contextmanager
def make_banana_env():
    env = UnityEnvironment(file_name=BANANA_APP)
    yield env
    env.close()


def banana_episode(env, agent: Agent, brain_name, max_t=1000):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    for t in range(max_t):
        action = agent.policy(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break

    agent.end_of_episode()
    return score
