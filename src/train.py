"""Обучение агента и запись статистики эпизодов."""
from pathlib import Path

import pandas as pd
from tqdm import trange

import config
from src.agent import QLearningAgent
from src.environment import RequestEnvironment


def train(episodes=config.TRAINING_EPISODES, output_dir=config.RESULTS_DIR,
          seed=config.RANDOM_SEED):
    if episodes <= 0:
        raise ValueError('Число эпизодов должно быть положительным')
    env = RequestEnvironment(seed=seed)
    agent = QLearningAgent(seed=seed)
    history = []
    for episode in trange(episodes, desc='Обучение'):
        state = env.reset()
        reward = completed = rejected = load = 0
        for _ in range(config.EPISODE_LENGTH):
            action = agent.choose_action(state)
            result = env.step(action)
            # Лимит времени обрывает наблюдение непрерывной задачи, поэтому
            # сохраняем bootstrap: время до конца не входит в состояние.
            agent.update(state, action, result.reward, result.state, done=False)
            state = result.state
            reward += result.reward
            completed += result.completed
            rejected += result.rejected
            load += sum(state)
            if result.done:
                break
        history.append(dict(episode=episode + 1, reward=reward,
                            completed=completed, rejected=rejected,
                            mean_load=load / config.EPISODE_LENGTH,
                            epsilon=agent.epsilon))
        agent.decay_epsilon()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history)
    agent.save(output_dir / 'q_table.npy')
    frame.to_csv(output_dir / 'training_history.csv', index=False)
    return agent, frame
