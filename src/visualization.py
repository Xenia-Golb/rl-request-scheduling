"""Сохранение графика обучения без графического интерфейса."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training(history, path, window=100):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.episode, history.reward, alpha=0.2, linewidth=0.6,
            label='Episode reward')
    ax.plot(history.episode, history.reward.rolling(window, min_periods=1).mean(),
            label=f'Rolling mean ({window} episodes)', color='darkblue')
    ax.set(xlabel='Episode', ylabel='Total reward', title='Q-learning training')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
