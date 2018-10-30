from banana_env import banana_episode, make_banana_env

import torch
from agent import Agent


def main():
    with make_banana_env() as env:
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]

        action_size = brain.vector_action_space_size
        state = env_info.vector_observations[0]
        state_size = len(state)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        agent = Agent(
            device,
            state_size,
            action_size,
            buffer_size=int(1e5),
            batch_size=64,
            learning_rate=5e-4,
            discount_rate=0.99,
            eps_decay=0.995,
            tau=1e-3,
            steps_per_update=4
        )

        return train(env, agent, brain_name)


def train(env, agent, brain_name, window_size=100, max_eps=int(2e5),
          min_score=14.0):
    scores = []

    for i in range(max_eps):
        score = banana_episode(env, agent, brain_name)
        scores.append(score)
        print('\r{i} - {score}'.format(i=i + 1, score=score), end="")
        if (i + 1) % window_size == 0:
            avg_score = sum(scores[-window_size:]) / window_size
            print('\r{i} - {score}'.format(i=i + 1, score=avg_score))
            agent.save('model.p')
            if avg_score > min_score:
                break
    return scores


if __name__ == '__main__':
    main()
