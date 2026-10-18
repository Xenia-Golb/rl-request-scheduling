"""Запуск: python main.py [--episodes 10000] [--eval-episodes 1000]."""
import argparse
from pathlib import Path

import config
from src.evaluate import evaluate
from src.train import train
from src.visualization import plot_training, visualize_agent
from src.agent import QLearningAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=config.TRAINING_EPISODES)
    parser.add_argument('--eval-episodes', type=int, default=config.EVALUATION_EPISODES)
    parser.add_argument('--output-dir', type=Path, default=config.RESULTS_DIR)
    parser.add_argument('--demo', action='store_true',
                        help='Загрузить Q-таблицу, показать решения и сохранить визуализации')
    parser.add_argument('--visualize', action='store_true',
                        help='Создать визуализации сохранённой Q-таблицы без обучения')
    parser.add_argument('--demo-seed', type=int, default=200_000)
    args = parser.parse_args()
    if args.demo or args.visualize:
        agent = QLearningAgent()
        try:
            agent.load(args.output_dir / 'q_table.npy')
        except (OSError, ValueError) as error:
            parser.error(f'Не удалось загрузить Q-таблицу: {error}. Сначала запустите обучение.')
        visualize_agent(agent, args.output_dir, seed=args.demo_seed, verbose=args.demo)
        print(f'Визуализации: {args.output_dir.resolve()}')
        return
    if args.episodes <= 0 or args.eval_episodes <= 0:
        parser.error('Число эпизодов должно быть положительным')
    agent, history = train(args.episodes, args.output_dir)
    plot_training(history, args.output_dir / 'training_reward.png')
    details, summary = evaluate(agent, args.eval_episodes)
    details.to_csv(args.output_dir / 'evaluation_history.csv', index=False)
    summary.to_csv(args.output_dir / 'evaluation_summary.csv')
    print(summary.round(3).to_string())
    visualize_agent(agent, args.output_dir, seed=args.demo_seed)
    print(f'Результаты: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
