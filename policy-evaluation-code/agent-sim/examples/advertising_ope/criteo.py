"""Independent Criteo binary advertising-treatment benchmark, not ad sequencing."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random

from .core import evaluate
from .reporting import write_csv, write_json

FEATURES = [f'f{i}' for i in range(12)]


def prepare_criteo_rows(records):
    x, actions, outcomes = [], [], []
    for row in records:
        try:
            features = [float(row[k]) for k in FEATURES]
            action, outcome = float(row['treatment']), float(row['conversion'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Criteo records require f0..f11, treatment and conversion.') from exc
        if not all(math.isfinite(v) for v in features) or action not in (0, 1) or outcome not in (0, 1):
            raise ValueError('Features must be finite and treatment/conversion binary.')
        x.append(features)
        actions.append(int(action))
        outcomes.append(int(outcome))
    if not x:
        raise ValueError('No Criteo rows were loaded.')
    return x, actions, outcomes


def run_benchmark(output, sample_size=100000, seed=42, bootstrap=200, csv_path=None,
                  revision=None):
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        import joblib
    except ImportError as exc:
        raise RuntimeError('Criteo benchmark requires scikit-learn and numpy; install requirements.txt.') from exc
    if sample_size < 1000:
        raise ValueError('Use at least 1000 sampled rows; 100000+ recommended for rare conversions.')
    if csv_path:
        # Reservoir sampling avoids assuming file ordering is chronological/random.
        rng, records, total, source_indices = random.Random(seed), [], 0, []
        with Path(csv_path).open(newline='') as f:
            for i, row in enumerate(csv.DictReader(f)):
                total += 1
                if i < sample_size:
                    records.append(row)
                    source_indices.append(i)
                else:
                    j = rng.randrange(i + 1)
                    if j < sample_size:
                        records[j] = row
                        source_indices[j] = i
        source = str(Path(csv_path).resolve())
        source_identity = dict(kind='local-csv')
    else:
        os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[3] / 'data' / 'huggingface'))
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError('Install datasets to download Criteo, or supply --csv with a local CSV.') from exc
        dataset = load_dataset('criteo/criteo-uplift', split='train', revision=revision)
        total = len(dataset)
        # Uniform index sample over the FULL released table, not a prefix/buffer sample.
        indices = random.Random(seed).sample(range(total), min(sample_size, total))
        records = dataset.select(indices)
        source_indices = indices
        source = 'huggingface:criteo/criteo-uplift'
        source_identity = dict(kind='huggingface', requested_revision=revision,
                               fingerprint=dataset._fingerprint,
                               datasets_version=importlib.metadata.version('datasets'))
    x, a, y = prepare_criteo_rows(records)
    x, a, y = np.asarray(x), np.asarray(a), np.asarray(y)
    source_identity['analyzed_sample_sha256'] = hashlib.sha256(
        x.tobytes() + a.tobytes() + y.tobytes()).hexdigest()
    labels = 2 * a + y
    if any(int(((a == action) & (y == outcome)).sum()) < 10
           for action in (0, 1) for outcome in (0, 1)):
        raise ValueError('Insufficient action/outcome support; increase sample size.')
    indices = np.arange(len(y))
    train, rest = train_test_split(indices, test_size=.4, stratify=labels, random_state=seed)
    validation, test = train_test_split(rest, test_size=.5, stratify=labels[rest], random_state=seed + 1)
    models = {}
    for action in (0, 1):
        subset = train[a[train] == action]
        models[action] = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                                        min_samples_leaf=50, random_state=seed)
        models[action].fit(x[subset], y[subset])
    propensity = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                               min_samples_leaf=100, random_state=seed)
    propensity.fit(x[train], a[train])

    def inputs(split, split_name):
        b = propensity.predict_proba(x[split])[:, 1]
        predictions = list(zip(models[0].predict_proba(x[split])[:, 1],
                               models[1].predict_proba(x[split])[:, 1]))
        rows = [dict(customer_id=f'criteo:{source_indices[int(i)]}', intent=0, action=int(a[i]),
                     reward=int(y[i]), b1=float(p), split=split_name)
                for i, p in zip(split, b)]
        return rows, predictions

    val_rows, val_predictions = inputs(validation, 'validation')
    validation_values = {}
    for candidate in ('never', 'always', 'uplift'):
        target = [float(m1 > m0) if candidate == 'uplift' else float(candidate == 'always')
                  for m0, m1 in val_predictions]
        result = evaluate(val_rows, target, {}, 20, seed, predictions=val_predictions,
                          behavior_probabilities_known=False)
        validation_values[candidate] = result['dr']['target']
    selected = max(validation_values, key=validation_values.get)
    test_rows, predictions = inputs(test, 'evaluation')
    target = [float(m1 > m0) if selected == 'uplift' else float(selected == 'always')
              for m0, m1 in predictions]
    result = evaluate(test_rows, target, {}, bootstrap, seed, predictions=predictions,
                      behavior_probabilities_known=False)
    matched = [r['reward'] for r, t in zip(test_rows, target) if r['action'] == int(t)]
    result.update(source='criteo', dataset=source, source_identity=source_identity,
                  released_rows_scanned=total,
                  sample_size=len(y), selected_policy=selected,
                  validation_policy_values=validation_values,
                  splits=dict(train=len(train), validation=len(validation), evaluation=len(test)),
                  naive_matched_only_value=sum(matched) / len(matched) if matched else None,
                  assumptions=[
                      'This compares binary treatment-assignment policies, not ad sequences.',
                      'No timestamps: the matched-only baseline is NOT a before/after measurement.',
                      'Behavior probabilities are estimated on training data, not logged known propensities.',
                      'Identification assumes conditional exchangeability and overlap in the released sample.',
                      'Nonuniform release subsampling may invalidate original-trial causal interpretations.',
                      'Intervals condition on training, selected policy and fitted nuisance models.',
                      'Results do not establish effects in the original deployed campaign population.',
                  ],
                  versions={name: importlib.metadata.version(name)
                            for name in ('numpy', 'scikit-learn')}, python=platform.python_version())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for r, (m0, m1), t in zip(test_rows, predictions, target):
        r.update(m0=float(m0), m1=float(m1), target_probability=t)
    write_csv(output / 'evaluation.csv', test_rows)
    joblib.dump(dict(outcome=models, propensity=propensity, policy=selected, features=FEATURES),
                output / 'models.joblib')
    write_json(output / 'metrics.json', result)
    write_json(output / 'split_source_indices.json',
               {name: [source_indices[int(i)] for i in indices]
                for name, indices in (('train', train), ('validation', validation), ('evaluation', test))})
    write_json(output / 'config.json', dict(sample_size=sample_size, seed=seed,
                                          bootstrap=bootstrap, csv_path=str(csv_path) if csv_path else None,
                                          revision=revision))
    (output / 'findings.md').write_text(
        '# Criteo treatment-assignment benchmark\n\n'
        f"Selected policy: {selected}. Held-out DR difference: {100 * result['dr']['difference']:+.3f} pp.\n\n"
        + '\n'.join('- ' + text for text in result['assumptions']) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('examples/advertising_ope/output_criteo'))
    parser.add_argument('--sample-size', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--csv', type=Path, help='Local uncompressed Criteo CSV; bypasses download')
    parser.add_argument('--revision', help='Optional Hugging Face dataset commit/revision to pin')
    args = parser.parse_args()
    try:
        result = run_benchmark(args.output, args.sample_size, args.seed, args.bootstrap, args.csv, args.revision)
        print(json.dumps(result, indent=2))
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f'Criteo benchmark unavailable: {exc}\n')


if __name__ == '__main__':
    main()
