import math
import unittest

from examples.advertising_ope.core import evaluate, fit_model, predict, learned_policy


class EstimatorTests(unittest.TestCase):
    # A wrong importance ratio or omitted DR residual breaks these literals.
    def setUp(self):
        self.rows = [
            dict(customer_id='a', intent=0, action=0, reward=0, b1=.5),
            dict(customer_id='b', intent=0, action=1, reward=1, b1=.5),
            dict(customer_id='c', intent=1, action=0, reward=1, b1=.5),
            dict(customer_id='d', intent=1, action=1, reward=0, b1=.5),
        ]
        self.model = {'0:0': .2, '0:1': .6, '1:0': .8, '1:1': .4}

    def test_hand_calculated_ipw_and_dr(self):
        result = evaluate(self.rows, [1., 1., 1., 1.], self.model, 40, 7)
        self.assertAlmostEqual(result['ipw']['target'], .5)
        self.assertAlmostEqual(result['dr']['target'], .5)
        self.assertAlmostEqual(result['dr']['logging'], .5)
        self.assertAlmostEqual(result['dr']['difference'], 0.)
        self.assertAlmostEqual(result['diagnostics']['effective_sample_size'], 2.)

    def test_identity_policy_ipw_is_observed_mean(self):
        result = evaluate(self.rows, [.5] * 4, self.model, 40, 7)
        self.assertEqual(result['ipw']['target'], .5)
        self.assertEqual(result['ipw']['difference'], 0.)
        self.assertEqual(result['dr']['difference'], 0.)
        self.assertEqual(result['dr']['difference_ci95'], [0., 0.])

    def test_dr_uses_unobserved_action_predictions(self):
        rows = [dict(customer_id='x', intent=0, action=0, reward=0, b1=.5)]
        result = evaluate(rows, [1.], {'0:0': .1, '0:1': .7}, 40, 1)
        self.assertEqual(result['ipw']['target'], 0.)
        self.assertEqual(result['dr']['target'], .7)
        self.assertEqual(result['diagnostics']['effective_sample_size'], 0.)

    def test_unsupported_policy_and_invalid_rows_fail(self):
        for b in [0., 1., -.1, float('nan')]:
            rows = [dict(self.rows[0], b1=b)]
            with self.assertRaises(ValueError):
                evaluate(rows, [.5], self.model, 40, 1)
        for reward in [-1, 2, float('nan')]:
            with self.assertRaises(ValueError):
                evaluate([dict(self.rows[0], reward=reward)], [.5], self.model, 40, 1)
        with self.assertRaises(ValueError):
            evaluate([], [], self.model, 40, 1)
        with self.assertRaises(ValueError):
            evaluate(self.rows, [1.], self.model, 40, 1)
        with self.assertRaises(ValueError):
            evaluate(self.rows, [math.nan] * 4, self.model, 40, 1)
        with self.assertRaises(ValueError):
            evaluate(self.rows, [.5] * 4, self.model, 0, 1)

    def test_bootstrap_is_seeded_and_paired(self):
        a = evaluate(self.rows, [1.] * 4, self.model, 50, 19)
        b = evaluate(self.rows, [1.] * 4, self.model, 50, 19)
        self.assertEqual(a, b)
        self.assertLessEqual(a['ipw']['target_ci95'][0], .5)
        self.assertGreaterEqual(a['ipw']['target_ci95'][1], .5)

    def test_dr_corrects_wrong_response_model_when_behavior_is_correct(self):
        wrong_model = {key: .2 for key in self.model}
        result = evaluate(self.rows, [1.] * 4, wrong_model, 40, 7)
        self.assertAlmostEqual(result['dr']['target'], .5)

    def test_dr_target_survives_wrong_propensity_with_correct_outcome_model(self):
        rows = []
        for intent in (0, 1):
            for action, rewards in ((0, [0, 0, 0, 1]), (1, [0, 1, 1, 1])):
                for reward in rewards:
                    rows.append(dict(customer_id=str(len(rows)), intent=intent,
                                     action=action, reward=reward, b1=.3))
        correct_model = {'0:0': .25, '1:0': .25, '0:1': .75, '1:1': .75}
        result = evaluate(rows, [1.] * len(rows), correct_model, 40, 1)
        self.assertAlmostEqual(result['dr']['target'], .75)
        self.assertAlmostEqual(result['ipw']['target'], 1.25)

    def test_estimated_propensity_uses_actual_observed_logging_baseline(self):
        rows = [dict(customer_id='a', intent=0, action=0, reward=0, b1=.9),
                dict(customer_id='b', intent=0, action=1, reward=1, b1=.9)]
        result = evaluate(rows, [1., 1.], {'0:0': 0., '0:1': 1.}, 40, 1,
                          behavior_probabilities_known=False)
        self.assertEqual(result['dr']['target'], 1.)
        self.assertEqual(result['dr']['logging'], .5)
        self.assertAlmostEqual(result['dr']['difference'], .5)

    def test_fitting_ignores_oracle_fields_and_requires_action_support(self):
        model = fit_model(self.rows)
        poisoned = [dict(r, oracle_p0=999, oracle_p1=-100, latent_intent=1)
                    for r in self.rows]
        self.assertEqual(model, fit_model(poisoned))
        self.assertAlmostEqual(predict(model, 0, 0), 1 / 3)
        self.assertEqual(learned_policy(model, 0), 1.)
        self.assertEqual(learned_policy(model, 1), 0.)
        with self.assertRaises(ValueError):
            fit_model([self.rows[0]])


if __name__ == '__main__':
    unittest.main()
