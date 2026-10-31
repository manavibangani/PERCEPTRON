"""Repair gRPC's x86-only WHEEL tag after verifying its universal2 Mach-O code.

Some macOS universal2 wheels contain both architectures but retain the x86-only
tag from their original build. Pip 26 checks the internal tag. This correction
changes WHEEL and its RECORD entry only; it never alters executable code.
"""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile


def atomic_write(path, content):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def repair_wheel_metadata(info, binaries):
    from packaging.tags import Tag, parse_tag, sys_tags

    info = Path(info)
    if not info.name.startswith('grpcio-') or not info.name.endswith('.dist-info'):
        raise ValueError('This narrowly scoped repair is only for grpcio metadata.')
    wheel, record = info / 'WHEEL', info / 'RECORD'
    old_wheel, old_record = wheel.read_bytes(), record.read_bytes()
    lines = old_wheel.decode().splitlines()
    tags = {tag for line in lines if line.startswith('Tag:')
            for tag in parse_tag(line.split(':', 1)[1].strip())}
    compatible = set(sys_tags())
    if tags.intersection(compatible):
        return dict(changed=False, tags=sorted(map(str, tags)))
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64' or not tags
            or not all(t.platform.startswith('macosx_') and t.platform.endswith('_x86_64') for t in tags)):
        raise ValueError('Unsupported wheel mismatch; refusing an unverified platform correction.')
    if not binaries:
        raise ValueError('No executable payload available to verify.')
    verified = []
    for binary in map(Path, binaries):
        try:
            architectures = subprocess.check_output(['lipo', '-archs', str(binary)],
                                                     text=True, stderr=subprocess.STDOUT).split()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError('Cannot verify Mach-O architecture; no metadata was changed.') from exc
        if not {'arm64', 'x86_64'}.issubset(architectures):
            raise ValueError('Executable does not contain both universal2 architectures.')
        verified.append(dict(path=str(binary), architectures=architectures,
                             sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    corrected = {Tag(t.interpreter, t.abi, t.platform[:-len('x86_64')] + 'universal2') for t in tags}
    if not corrected.intersection(compatible):
        raise ValueError('Verified universal2 tag is still incompatible with this interpreter.')
    new_lines = [line for line in lines if line.strip() and not line.startswith('Tag:')]
    new_lines += ['Tag: ' + str(t) for t in sorted(corrected, key=str)]
    new_wheel = ('\n'.join(new_lines) + '\n\n').encode()
    rows = list(csv.reader(io.StringIO(old_record.decode())))
    wheel_entry = info.name + '/WHEEL'
    matches = [row for row in rows if row and row[0] == wheel_entry]
    if len(matches) != 1:
        raise ValueError('Expected exactly one WHEEL integrity record; no metadata changed.')
    digest = base64.urlsafe_b64encode(hashlib.sha256(new_wheel).digest()).rstrip(b'=').decode()
    matches[0][:] = [wheel_entry, 'sha256=' + digest, str(len(new_wheel))]
    stream = io.StringIO(newline='')
    csv.writer(stream).writerows(rows)
    try:
        atomic_write(wheel, new_wheel)
        atomic_write(record, stream.getvalue().encode())
    except OSError:
        atomic_write(wheel, old_wheel)
        atomic_write(record, old_record)
        raise
    return dict(changed=True, original_tags=sorted(map(str, tags)),
                corrected_tags=sorted(map(str, corrected)), verified_binaries=verified,
                executable_code_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path,
                        default=Path(__file__).parent / 'output_installation/grpc-metadata-repair.json')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        print('gRPC universal2 metadata repair is not needed on this platform.')
        return
    try:
        dist = importlib.metadata.distribution('grpcio')
        wheel_file = next(f for f in dist.files if str(f).endswith('.dist-info/WHEEL'))
        info = Path(dist.locate_file(wheel_file)).parent
        binaries = [dist.locate_file(f) for f in dist.files if str(f).endswith(('.so', '.dylib'))]
        result = repair_wheel_metadata(info, binaries)
        result['version'] = dist.version
        if result['changed'] or not args.audit.exists():
            args.audit.parent.mkdir(parents=True, exist_ok=True)
            args.audit.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(1, f'gRPC metadata verification failed: {exc}\n')


if __name__ == '__main__':
    main()
