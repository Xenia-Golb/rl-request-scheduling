"""Оценка без обучения на новых случайных эпизодах."""
import random

import pandas as pd

import config
from src.environment import RequestEnvironment


def evaluate(agent, episodes=config.EVALUATION_EPISODES, seed=100_000):
    if episodes <= 0:
        raise ValueError('Число эпизодов должно быть положительным')
    rows = []
    for strategy in ('q_learning', 'random', 'least_loaded', 'fastest_available'):
        for episode in range(episodes):
            # Одинаковый список новых seed для всех стратегий; RNG действий отдельно.
            env = RequestEnvironment(seed=seed + episode)
            rng = random.Random(seed + episode)
            state = env.reset()
            reward = completed = rejected = load = 0
            for _ in range(config.EPISODE_LENGTH):
                if strategy == 'q_learning':
                    values = agent.q_table[agent.state_index(state)]
                    action = rng.choice([i for i, v in enumerate(values) if v == values.max()])
                elif strategy == 'random':
                    action = rng.randrange(config.NUM_ACTIONS)
                elif strategy == 'least_loaded':
                    action = rng.choice([i for i, n in enumerate(state) if n == min(state)])
                else:
                    available = [i for i, n in enumerate(state)
                                 if n < config.MAX_REQUESTS_PER_OPERATOR]
                    action = max(available or list(range(config.NUM_ACTIONS)),
                                 key=lambda i: config.COMPLETION_PROBABILITIES[i])
                result = env.step(action)
                state = result.state
                reward += result.reward
                completed += result.completed
                rejected += result.rejected
                load += sum(state)
                if result.done:
                    break
            rows.append(dict(strategy=strategy, episode=episode + 1, reward=reward,
                             completed=completed, rejected=rejected,
                             mean_load=load / config.EPISODE_LENGTH))
    details = pd.DataFrame(rows)
    summary = details.groupby('strategy', sort=False).agg(
        reward_mean=('reward', 'mean'), reward_std=('reward', 'std'),
        completed_mean=('completed', 'mean'), rejected_mean=('rejected', 'mean'),
        mean_load=('mean_load', 'mean'))
    return details, summary
