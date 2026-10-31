"""The numeric experiment must install without private city-server binaries."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PackagingTests(unittest.TestCase):
    def test_python_only_build_requires_no_binary_credentials(self):
        repo = Path(__file__).resolve().parents[3]
        env = dict(os.environ, AGENTSOCIETY_SKIP_BINARIES='1')
        env.pop('GITLAB_AUTH', None)
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run(
                [sys.executable, 'setup.py', 'build_ext', '--build-lib', output],
                cwd=repo, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(list(Path(output).rglob('agentsociety-sim')))


if __name__ == '__main__':
    unittest.main()
