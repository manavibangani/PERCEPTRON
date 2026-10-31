import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from examples.advertising_ope.criteo import FEATURES, prepare_criteo_rows


def records():
    return [dict({f'f{i}': i / 10 for i in range(12)},
                 treatment=a, conversion=y, visit=1, exposure=1)
            for a, y in ((0, 0), (1, 1), (0, 1), (1, 0))]


class OptionalTests(unittest.TestCase):
    def test_criteo_uses_only_pre_treatment_features(self):
        source = records()
        x, a, y = prepare_criteo_rows(source)
        self.assertEqual(FEATURES, [f'f{i}' for i in range(12)])
        self.assertEqual(len(x[0]), 12)
        self.assertEqual(a, [0, 1, 0, 1])
        self.assertEqual(y, [0, 1, 1, 0])
        changed = [dict(r, exposure=-999, visit=math.nan) for r in source]
        self.assertEqual((x, a, y), prepare_criteo_rows(changed))

    def test_criteo_rejects_invalid_data(self):
        for patch in ({'f0': math.nan}, {'conversion': 2}, {'treatment': -1}):
            source = records()
            source[0].update(patch)
            with self.assertRaises(ValueError):
                prepare_criteo_rows(source)
        with self.assertRaises(ValueError):
            prepare_criteo_rows([])

    def test_optional_cli_help_does_not_import_runtime(self):
        for module in ('criteo', 'agentsociety_adapter'):
            result = subprocess.run([sys.executable, '-m', f'examples.advertising_ope.{module}', '--help'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_unavailable_runtime_reports_error_without_fake_results(self):
        if importlib.util.find_spec('ray') is not None:
            self.skipTest('Runtime dependency present; run real adapter test instead.')
        with tempfile.TemporaryDirectory() as d:
            result = subprocess.run([sys.executable, '-m', 'examples.advertising_ope.agentsociety_adapter',
                                     '--n', '10', '--output', d], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('AgentSociety runtime unavailable', result.stderr)
            self.assertFalse((Path(d) / 'logging.csv').exists())

    @unittest.skipUnless(importlib.util.find_spec('ray') is not None, 'AgentSociety dependencies unavailable')
    def test_real_agent_forward_emits_logs(self):
        import asyncio
        from examples.advertising_ope.agentsociety_adapter import run_agents
        rows, oracle = asyncio.run(run_agents(20, 42))
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r['customer_id'] for r in rows}), 20)
        self.assertTrue(all('p0' not in r for r in rows))
        self.assertEqual(len(oracle), 20)


if __name__ == '__main__':
    unittest.main()
