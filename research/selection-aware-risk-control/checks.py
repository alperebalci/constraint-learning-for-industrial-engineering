import unittest

import numpy as np
from study import calibrate, policy_losses, synthetic, upper_binomial


class RiskTests(unittest.TestCase):
    def test_zero(self):
        self.assertAlmostEqual(upper_binomial(0, 100, 0.05), 1 - 0.05**0.01)

    def test_all(self):
        self.assertEqual(upper_binomial(10, 10, 0.05), 1)

    def test_select_not_random(self):
        loss, cov, cost = policy_losses([[0.9, 0.9]], [[2, 1]], [[0, 1]], 0.8)
        self.assertTrue(loss[0])
        self.assertTrue(cov[0])
        self.assertEqual(cost[0], 1)

    def test_abstain(self):
        loss, cov, _ = policy_losses([[0.1, 0.2]], [[1, 2]], [[1, 1]], 0.9)
        self.assertFalse(loss[0])
        self.assertFalse(cov[0])

    def test_no_certification(self):
        r = calibrate(np.ones((5, 2)), np.ones((5, 2)), np.ones((5, 2)), [0.5, 0.9])
        self.assertIsNone(r["selected"])

    def test_family_penalty(self):
        s = np.ones((100, 1))
        c = s.copy()
        u = np.zeros_like(s)
        a = calibrate(s, c, u, [0.5])
        b = calibrate(s, c, u, [0.5, 0.6])
        self.assertGreater(
            b["candidates"][0]["upper_risk"], a["candidates"][0]["upper_risk"]
        )

    def test_reproducible(self):
        a = synthetic(1, 10, 3)
        b = synthetic(1, 10, 3)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            upper_binomial(2, 1, 0.05)
        with self.assertRaises(ValueError):
            calibrate([[0.5]], [[1]], [[0]], [])


if __name__ == "__main__":
    unittest.main()
