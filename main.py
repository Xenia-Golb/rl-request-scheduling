"""Запуск: python main.py [--episodes 10000] [--eval-episodes 1000]."""
import argparse
from pathlib import Path

import config
from src.evaluate import evaluate
from src.train import train
from src.visualization import plot_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=config.TRAINING_EPISODES)
    parser.add_argument('--eval-episodes', type=int, default=config.EVALUATION_EPISODES)
    parser.add_argument('--output-dir', type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()
    if args.episodes <= 0 or args.eval_episodes <= 0:
        parser.error('Число эпизодов должно быть положительным')
    agent, history = train(args.episodes, args.output_dir)
    plot_training(history, args.output_dir / 'training_reward.png')
    details, summary = evaluate(agent, args.eval_episodes)
    details.to_csv(args.output_dir / 'evaluation_history.csv', index=False)
    summary.to_csv(args.output_dir / 'evaluation_summary.csv')
    print(summary.round(3).to_string())
    print(f'Результаты: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
