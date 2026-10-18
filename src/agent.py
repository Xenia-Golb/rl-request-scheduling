"""Табличный Q-learning с epsilon-greedy выбором действий."""
from numbers import Integral

import numpy as np

import config


class QLearningAgent:
    def __init__(self, seed=config.RANDOM_SEED):
        self.q_table = np.zeros((config.NUM_STATES, config.NUM_ACTIONS))
        self.epsilon = config.EPSILON_START
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def state_index(state):
        if len(state) != config.NUM_OPERATORS or any(
            not isinstance(n, Integral) or not 0 <= n <= config.MAX_REQUESTS_PER_OPERATOR
            for n in state
        ):
            raise ValueError('Некорректное состояние')
        index = 0
        for load in state:
            index = index * (config.MAX_REQUESTS_PER_OPERATOR + 1) + int(load)
        return index

    def choose_action(self, state, explore=True):
        row = self.q_table[self.state_index(state)]
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(config.NUM_ACTIONS))
        # Случайный выбор среди равных максимумов исключает перекос к оператору 0.
        return int(self._rng.choice(np.flatnonzero(row == row.max())))

    def update(self, state, action, reward, next_state, done=False):
        index = self.state_index(state)
        target = reward
        if not done:
            target += config.DISCOUNT_FACTOR * self.q_table[self.state_index(next_state)].max()
        self.q_table[index, action] += config.LEARNING_RATE * (
            target - self.q_table[index, action]
        )

    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        table = np.load(path, allow_pickle=False)
        if table.shape != self.q_table.shape or not np.isfinite(table).all():
            raise ValueError('Некорректная Q-таблица')
        self.q_table = table.astype(float)
