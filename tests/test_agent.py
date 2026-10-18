import itertools
import unittest
import numpy as np
import config
from src.agent import QLearningAgent


class AgentTests(unittest.TestCase):
    def test_encoding_is_bijection(self):
        states = itertools.product(range(4), repeat=3)
        self.assertEqual({QLearningAgent.state_index(s) for s in states}, set(range(64)))

    def test_update_and_terminal_target(self):
        agent = QLearningAgent()
        agent.q_table[1] = [2, 4, 3]
        agent.update((0, 0, 0), 2, 1, (0, 0, 1))
        self.assertAlmostEqual(agent.q_table[0, 2], 0.1 * (1 + 0.95 * 4))
        agent.update((0, 0, 0), 1, 1, (0, 0, 1), done=True)
        self.assertAlmostEqual(agent.q_table[0, 1], 0.1)

    def test_greedy_and_exploration(self):
        agent = QLearningAgent()
        agent.q_table[0] = [0, 3, 1]
        self.assertEqual({agent.choose_action((0, 0, 0), explore=False)
                          for _ in range(100)}, {1})
        self.assertEqual({agent.choose_action((0, 0, 0)) for _ in range(100)}, {0, 1, 2})
        for _ in range(10000):
            agent.decay_epsilon()
        self.assertEqual(agent.epsilon, config.EPSILON_MIN)

    def test_evaluation_does_not_learn(self):
        from src.evaluate import evaluate
        agent = QLearningAgent()
        original = agent.q_table.copy()
        epsilon = agent.epsilon
        details, summary = evaluate(agent, episodes=2)
        np.testing.assert_array_equal(original, agent.q_table)
        self.assertEqual(agent.epsilon, epsilon)
        self.assertEqual(len(details), 8)
        self.assertEqual(len(summary), 4)


if __name__ == '__main__':
    unittest.main()
