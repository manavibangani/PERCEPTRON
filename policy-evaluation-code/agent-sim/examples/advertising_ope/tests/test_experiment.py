import json
import csv
from pathlib import Path
import tempfile
import unittest

from examples.advertising_ope.core import evaluate, fit_model
from examples.advertising_ope.simulation import simulate, oracle_values, estimate_behavior
from examples.advertising_ope.run import run_experiment


class ExperimentTests(unittest.TestCase):
    def test_reproducible_logs_separate_oracle(self):
        rows, oracle = simulate(1000, 17)
        self.assertEqual((rows, oracle), simulate(1000, 17))
        self.assertTrue(all('latent_intent' not in r and 'p0' not in r for r in rows))
        self.assertTrue(all('p0' in r and 'latent_intent' in r for r in oracle))
        self.assertEqual({r['action'] for r in rows}, {0, 1})

    def test_population_shift_reverses_naive_conclusion(self):
        train, _ = simulate(15000, 101)
        logs, oracle = simulate(15000, 102)
        after, _ = simulate(15000, 103, high_share=.8, target_probability=1.)
        result = evaluate(logs, [1.] * len(logs), fit_model(train), 20, 4)
        truth = oracle_values(oracle, [1.] * len(logs))
        naive = sum(r['reward'] for r in after) / len(after) - result['observed_logging_mean']
        self.assertGreater(naive, .05)
        self.assertLess(truth['difference'], 0)
        self.assertLess(result['dr']['difference'], 0)
        self.assertAlmostEqual(result['ipw']['target'], truth['target'], delta=.02)
        self.assertAlmostEqual(result['dr']['target'], truth['target'], delta=.02)

    def test_confounding_stress_test_exposes_bias(self):
        train, _ = simulate(15000, 201, confounding=3.)
        logs, oracle = simulate(15000, 202, confounding=3.)
        # Evaluator sees estimated observed-context propensities, not U-conditional b.
        b = estimate_behavior(train)
        observed = [dict(r, b1=b[str(r['intent'])]) for r in logs]
        result = evaluate(observed, [1.] * len(logs), fit_model(train), 20, 1)
        truth = oracle_values(oracle, [1.] * len(logs))
        self.assertGreater(abs(result['dr']['target'] - truth['target']), .02)

    def test_invalid_simulation_configuration(self):
        for kwargs in ({'n': 0}, {'n': 20, 'high_share': -1},
                       {'n': 20, 'scenario': 'invented'}, {'n': 20, 'confounding': -1}):
            with self.assertRaises(ValueError):
                simulate(seed=1, **kwargs)

    def test_runner_writes_replayable_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            result = run_experiment(Path(d), n=1600, seed=42, bootstrap=20)
            saved = json.loads((Path(d) / 'metrics.json').read_text())
            self.assertEqual(saved, result)
            self.assertTrue((Path(d) / 'logging.csv').exists())
            self.assertTrue((Path(d) / 'oracle.csv').exists())
            self.assertTrue((Path(d) / 'model.json').exists())
            self.assertIn('Synthetic', (Path(d) / 'findings.md').read_text())
            self.assertEqual(result['source'], 'synthetic')
            self.assertEqual(result['evaluation']['n'], 1600)
            with (Path(d) / 'sensitivity_3_training.csv').open() as f:
                train = list(csv.DictReader(f))
            for intent in ('0', '1'):
                # A per-row true U-conditional propensity would reveal hidden intent.
                self.assertEqual(len({r['b1'] for r in train if r['intent'] == intent}), 1)


if __name__ == '__main__':
    unittest.main()
