"""Дискретная среда распределения заявок между тремя операторами."""

from dataclasses import dataclass
from numbers import Integral
import random

import config


@dataclass(frozen=True)
class StepResult:
    state: tuple[int, ...]
    reward: float
    done: bool
    completed: int
    rejected: bool


class RequestEnvironment:
    """Одна новая заявка за шаг, без отдельной очереди ожидания.

    Сначала заявка назначается выбранному оператору. Если он заполнен,
    заявка отклоняется. Затем каждая заявка (включая новую) независимо
    завершается с вероятностью своего оператора. Штраф за нагрузку
    рассчитывается по числу заявок, оставшихся после обработки.

    Лимит шагов — граница эпизода симуляции. reset() очищает все заявки.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self.reset()

    @property
    def state(self) -> tuple[int, ...]:
        """Неизменяемый снимок текущей загрузки операторов."""
        return tuple(self._loads)

    def reset(self, seed: int | None = None) -> tuple[int, ...]:
        """Начать эпизод; seed позволяет воспроизвести случайную траекторию.

        Без seed генератор продолжает последовательность случайных чисел,
        чтобы последовательные эпизоды обучения отличались друг от друга.
        """
        if seed is not None:
            self._rng.seed(seed)
        self._loads = [0] * config.NUM_OPERATORS
        self._steps = 0
        return self.state

    def step(self, action: int) -> StepResult:
        """Распределить заявку и вернуть состояние, награду и итог шага."""
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise ValueError("Действие должно быть целым числом: 0, 1 или 2")
        if not 0 <= action < config.NUM_OPERATORS:
            raise ValueError("Номер оператора должен быть от 0 до 2")
        if self._steps >= config.EPISODE_LENGTH:
            raise RuntimeError("Эпизод завершён. Вызовите reset()")

        rejected = self._loads[action] >= config.MAX_REQUESTS_PER_OPERATOR
        if not rejected:
            self._loads[action] += 1

        completed = 0
        for operator, probability in enumerate(config.COMPLETION_PROBABILITIES):
            finished = sum(
                self._rng.random() < probability
                for _ in range(self._loads[operator])
            )
            self._loads[operator] -= finished
            completed += finished

        self._steps += 1
        reward = (
            config.COMPLETION_REWARD * completed
            - config.LOAD_PENALTY * sum(self._loads)
            - config.REJECTION_PENALTY * rejected
        )
        return StepResult(
            state=self.state,
            reward=reward,
            done=self._steps >= config.EPISODE_LENGTH,
            completed=completed,
            rejected=rejected,
        )
