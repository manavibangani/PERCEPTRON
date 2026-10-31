import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import platform
import tempfile
import unittest

from examples.advertising_ope.repair_grpc_metadata import repair_wheel_metadata


@unittest.skipUnless(platform.system() == 'Darwin' and platform.machine() == 'arm64'
                     and importlib.util.find_spec('grpc') is not None,
                     'Requires an installed macOS arm64 gRPC binary')
class InstallationTests(unittest.TestCase):
    def fixture(self, directory):
        from packaging.tags import sys_tags
        tag = next(iter(sys_tags()))
        info = Path(directory) / 'grpcio-1.71.2.dist-info'
        info.mkdir()
        (info / 'WHEEL').write_text(
            f'Wheel-Version: 1.0\nTag: {tag.interpreter}-{tag.abi}-macosx_10_14_x86_64\n')
        (info / 'RECORD').write_text(f'{info.name}/WHEEL,,\n{info.name}/RECORD,,\n')
        return info

    def test_verified_universal_binary_metadata_repair_preserves_code(self):
        from packaging.tags import parse_tag, sys_tags
        dist = importlib.metadata.distribution('grpcio')
        binary = next(dist.locate_file(f) for f in dist.files if str(f).endswith('.so'))
        before = hashlib.sha256(binary.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            info = self.fixture(directory)
            result = repair_wheel_metadata(info, [binary])
            self.assertTrue(result['changed'])
            tag = next(l.split(': ', 1)[1] for l in (info / 'WHEEL').read_text().splitlines()
                       if l.startswith('Tag: '))
            self.assertTrue(parse_tag(tag).intersection(sys_tags()))
            with (info / 'RECORD').open(newline='') as f:
                row = next(csv.reader(f))
            digest = base64.urlsafe_b64encode(hashlib.sha256((info / 'WHEEL').read_bytes()).digest()).rstrip(b'=').decode()
            self.assertEqual(row[1], 'sha256=' + digest)
            self.assertEqual(before, hashlib.sha256(binary.read_bytes()).hexdigest())
            self.assertFalse(repair_wheel_metadata(info, [binary])['changed'])

    def test_non_macho_payload_cannot_bypass_platform_check(self):
        with tempfile.TemporaryDirectory() as directory:
            info = self.fixture(directory)
            fake = Path(directory) / 'not-a-binary.so'
            fake.write_text('not an executable')
            before = (info / 'WHEEL').read_bytes()
            with self.assertRaises(ValueError):
                repair_wheel_metadata(info, [fake])
            self.assertEqual(before, (info / 'WHEEL').read_bytes())
