"""Run or replay synthetic OPE: python -m examples.advertising_ope.run."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from .core import evaluate, fit_model, learned_policy
from .simulation import simulate, oracle_values, estimate_behavior
from .reporting import write_json, write_csv, read_logs, write_findings, plot_results


def run_experiment(output, n=10000, seed=42, bootstrap=200):
    if n < 200:
        raise ValueError('Use at least 200 customers per split; 10000+ recommended.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train, _ = simulate(n, seed, split='train')
    logs, oracle = simulate(n, seed + 1)
    after, after_oracle = simulate(n, seed + 2, high_share=.8,
                                   target_probability=1., split='later-population')
    model = fit_model(train)
    target = [1.] * n
    evaluation = evaluate(logs, target, model, bootstrap, seed)
    before = evaluation['observed_logging_mean']
    after_mean = sum(r['reward'] for r in after) / n
    h_train, _ = simulate(n, seed + 10, scenario='heterogeneous', split='train')
    h_logs, h_oracle = simulate(n, seed + 11, scenario='heterogeneous')
    h_model = fit_model(h_train)
    h_target = [learned_policy(h_model, r['intent']) for r in h_logs]
    learned = dict(policy={str(x): learned_policy(h_model, x) for x in (0, 1)},
                   evaluation=evaluate(h_logs, h_target, h_model, bootstrap, seed + 10),
                   oracle=oracle_values(h_oracle, h_target))
    sensitivity = []
    for strength in (0., 1., 2., 3.):
        # Same random streams and context distribution across the sweep.
        s_train, s_train_oracle = simulate(n, seed + 20, confounding=strength, split='train')
        s_logs, s_oracle = simulate(n, seed + 21, confounding=strength)
        propensity = estimate_behavior(s_train)
        observed = [dict(r, b1=propensity[str(r['intent'])]) for r in s_logs]
        observed_training = [dict(r, b1=propensity[str(r['intent'])]) for r in s_train]
        s_model = fit_model(observed_training)
        result = evaluate(observed, target, s_model,
                          max(20, min(bootstrap, 40)), seed,
                          behavior_probabilities_known=False)
        truth = oracle_values(s_oracle, target)
        sensitivity.append(dict(strength=strength, dr_target=result['dr']['target'],
                                oracle_target=truth['target'], bias=result['dr']['target'] - truth['target'],
                                evaluation=result, oracle=truth, fitted_propensity=propensity))
        # Only observed-context propensity estimates enter these saved evaluator logs.
        for r in observed + observed_training:
            r['b0'] = 1 - r['b1']
            r['propensity'] = r['b1'] if r['action'] else 1 - r['b1']
        write_csv(output / f'sensitivity_{strength:g}_logging.csv', observed)
        write_csv(output / f'sensitivity_{strength:g}_oracle.csv', s_oracle)
        write_csv(output / f'sensitivity_{strength:g}_training.csv', observed_training)
        write_csv(output / f'sensitivity_{strength:g}_training_oracle.csv', s_train_oracle)
        write_json(output / f'sensitivity_{strength:g}_model.json', s_model)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob('*.py')):
        source_hash.update(source.name.encode())
        source_hash.update(source.read_bytes())
    metrics = dict(source='synthetic', evaluation=evaluation, oracle=oracle_values(oracle, target),
                   naive=dict(before=before, after=after_mean, difference=after_mean - before,
                              before_high_share=.2, after_high_share=.8),
                   learned=learned, sensitivity=sensitivity,
                   provenance=dict(python=platform.python_version(), revision=revision,
                                   source_sha256=source_hash.hexdigest(),
                                   uncommitted_changes_possible=True,
                                   config=dict(n=n, seed=seed, bootstrap=bootstrap)))
    for name, rows in (('training', train), ('logging', logs), ('oracle', oracle),
                       ('later_population', after), ('later_oracle', after_oracle),
                       ('learned_training', h_train), ('learned_logging', h_logs), ('learned_oracle', h_oracle)):
        write_csv(output / f'{name}.csv', rows)
    write_json(output / 'model.json', model)
    write_json(output / 'learned_model.json', h_model)
    write_json(output / 'learned_policy.json', learned['policy'])
    write_json(output / 'config.json', dict(n=n, seed=seed, bootstrap=bootstrap))
    metrics['chart_status'] = plot_results(output / 'comparison.png', metrics)
    write_json(output / 'metrics.json', metrics)
    write_findings(output / 'findings.md', metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('examples/advertising_ope/output'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--n', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--bootstrap', type=int)
    parser.add_argument('--logs', type=Path, help='Replay observed logs only; requires --model')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--estimated-propensities', action='store_true',
                        help='Replay with estimated propensities: use observed logging baseline')
    parser.add_argument('--policy', choices=['sequencing', 'learned', 'logging'], default='sequencing')
    args = parser.parse_args()
    config = dict(n=10000, seed=42, bootstrap=200)
    if args.config:
        supplied = json.loads(args.config.read_text())
        if set(supplied) - set(config):
            parser.error('Config accepts only n, seed and bootstrap.')
        config.update(supplied)
    config.update({k: getattr(args, k) for k in config if getattr(args, k) is not None})
    try:
        if args.logs:
            if not args.model:
                parser.error('--logs requires an independently fitted --model')
            logs = read_logs(args.logs)
            model = json.loads(args.model.read_text())
            target = [r['b1'] if args.policy == 'logging' else
                      learned_policy(model, r['intent']) if args.policy == 'learned' else 1. for r in logs]
            result = evaluate(logs, target, model, config['bootstrap'], config['seed'],
                              behavior_probabilities_known=not args.estimated_propensities)
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(args.output / 'replay_metrics.json', result)
            print(json.dumps(result, indent=2))
        else:
            result = run_experiment(args.output, **config)
            print(f"Synthetic naive difference: {100 * result['naive']['difference']:+.3f} pp")
            print(f"Held-out DR difference: {100 * result['evaluation']['dr']['difference']:+.3f} pp")
            print(f"Oracle difference: {100 * result['oracle']['difference']:+.3f} pp")
            print(f"Findings: {args.output / 'findings.md'}")
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'Experiment failed: {exc}\n')


if __name__ == '__main__':
    main()
