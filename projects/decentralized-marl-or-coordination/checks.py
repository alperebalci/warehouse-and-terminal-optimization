import itertools
import unittest

import numpy as np
import torch
from study import CTDE, Warehouse, shield, train


class MARLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_local_information(self):
        e = Warehouse(robots=2)
        e.pos = np.array([[0, 0], [4, 4]])
        e.targets = np.array([[1, 1], [0, 4]])
        before = e.local()[0].copy()
        e.pos[1] = [3, 4]
        e.targets[1] = [4, 0]
        np.testing.assert_array_equal(e.local()[0], before)

    def test_actor_shape(self):
        m = CTDE()
        self.assertEqual(tuple(m.actor(torch.zeros(3, 9)).shape), (3, 5))

    def test_swap(self):
        e = Warehouse(robots=2)
        e.pos = np.array([[0, 0], [0, 1]])
        e.targets = np.array([[0, 1], [0, 0]])
        _, _, bad = e.step([4, 3])
        self.assertEqual(bad, 2)
        np.testing.assert_array_equal(e.pos, [[0, 0], [0, 1]])

    def test_shield_exhaustive(self):
        e = Warehouse(robots=2)
        e.pos = np.array([[0, 0], [0, 1]])
        e.targets = np.array([[4, 4], [4, 0]])
        scores = np.random.default_rng(3).normal(size=(2, 5))
        a = shield(e, scores)
        best = -np.inf
        for b in itertools.product(range(5), repeat=2):
            if not e.masks()[0, b[0]] or not e.masks()[1, b[1]]:
                continue
            q = e.destinations()[np.arange(2), b]
            if np.array_equal(q[0], q[1]) or (
                np.array_equal(q[0], e.pos[1]) and np.array_equal(q[1], e.pos[0])
            ):
                continue
            best = max(best, scores[0, b[0]] + scores[1, b[1]])
        self.assertAlmostEqual(scores[np.arange(2), a].sum(), best)

    def test_random_safety(self):
        for seed in range(20):
            e = Warehouse(seed)
            a = shield(e, np.random.default_rng(seed).normal(size=(3, 5)))
            _, _, bad = e.step(a)
            self.assertEqual(bad, 0)

    def test_training(self):
        torch.manual_seed(1)
        a = CTDE()
        b = train(1, rounds=2)
        self.assertTrue(
            any(not torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
        )

    def test_invalid_scores(self):
        with self.assertRaises(ValueError):
            shield(Warehouse(), np.full((3, 5), np.nan))

    def test_mask(self):
        e = Warehouse()
        self.assertTrue(e.masks()[:, 0].all())


if __name__ == "__main__":
    unittest.main()
