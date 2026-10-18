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


def run_episode(agent, seed=200_000, verbose=False):
    """Жадная политика на отдельном seed; Q-таблица не изменяется."""
    import numpy as np
    import pandas as pd
    import config
    from src.environment import RequestEnvironment

    env = RequestEnvironment(seed=seed)
    rng = np.random.default_rng(seed)
    state = env.reset()
    rows = []
    for step in range(1, config.EPISODE_LENGTH + 1):
        values = agent.q_table[agent.state_index(state)]
        action = int(rng.choice(np.flatnonzero(values == values.max())))
        result = env.step(action)
        rows.append(dict(step=step, state=state, action=action,
                         next_state=result.state, reward=result.reward,
                         completed=result.completed, rejected=result.rejected))
        if verbose:
            print(f'Шаг {step}\nЗагрузка до решения: {state}\n'
                  f'Агент выбрал: Оператор №{action + 1}\n'
                  f'Награда: {result.reward:+.1f}\n'
                  f'Загрузка после обработки: {result.state}\n'
                  f'Завершено: {result.completed}; отказ: {"да" if result.rejected else "нет"}\n')
        state = result.state
        if result.done:
            break
    return pd.DataFrame(rows)


def plot_operator_loads(episode, path):
    """Загрузка после каждого шага плюс начальное состояние при t=0."""
    import numpy as np
    import config

    loads = np.array([episode.iloc[0]['state'], *episode.next_state])
    fig, axes = plt.subplots(config.NUM_OPERATORS, 1, figsize=(12, 7), sharex=True)
    colors = ('#2563eb', '#ea580c', '#16a34a')
    for i, ax in enumerate(axes):
        ax.step(range(len(loads)), loads[:, i], where='post', color=colors[i])
        ax.set(ylabel=f'Оператор №{i + 1}', yticks=range(4), ylim=(-0.15, 3.3))
        ax.grid(alpha=0.2)
    axes[0].set_title('Загрузка операторов за один эпизод — Q-learning')
    axes[-1].set_xlabel('Шаг (0 — начальное состояние; далее — после обработки)')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_decision_map(agent, path):
    """Четыре среза по загрузке третьего оператора покрывают все 64 состояния.

    При равенстве максимумов показываются ВСЕ допустимые жадные действия.
    Нулевая строка Q не доказывает, что состояние посещалось при обучении.
    """
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    colors = ['#bfdbfe', '#fed7aa', '#bbf7d0', '#e5e7eb']
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.4))
    for third, ax in enumerate(axes):
        grid = np.zeros((4, 4), dtype=int)
        labels = {}
        for first in range(4):
            for second in range(4):
                values = agent.q_table[agent.state_index((first, second, third))]
                actions = np.flatnonzero(values == values.max())
                grid[first, second] = actions[0] if len(actions) == 1 else 3
                labels[first, second] = '/'.join(str(a + 1) for a in actions)
        ax.imshow(grid, origin='lower', cmap=ListedColormap(colors),
                  norm=BoundaryNorm(np.arange(-0.5, 4.5), 4))
        for (first, second), label in labels.items():
            ax.text(second, first, label, ha='center', va='center', fontsize=12)
        ax.set(title=f'Оператор №3: {third} заявки', xlabel='Загрузка оператора №2',
               ylabel='Загрузка оператора №1', xticks=range(4), yticks=range(4))
    fig.suptitle('Карта жадных решений Q-learning — все 64 состояния', fontsize=15)
    fig.legend(handles=[Patch(facecolor=c, label=label) for c, label in zip(
        colors, ['Выбор №1', 'Выбор №2', 'Выбор №3', 'Равные максимумы Q'])],
        loc='lower center', bbox_to_anchor=(0.5, 0.07), ncol=4)
    fig.text(0.5, 0.025, 'Числа в ячейках — номера операторов. При равенстве агент выбирает случайно среди указанных.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.25, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def visualize_agent(agent, output_dir, seed=200_000, verbose=False):
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode = run_episode(agent, seed=seed, verbose=verbose)
    episode.to_csv(output_dir / 'agent_episode.csv', index=False)
    plot_operator_loads(episode, output_dir / 'operator_loads.png')
    plot_decision_map(agent, output_dir / 'decision_map.png')
    return episode
