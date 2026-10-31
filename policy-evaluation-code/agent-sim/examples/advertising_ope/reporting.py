"""Write portable experiment artifacts without optional package imports."""

import csv
import json
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write empty logs.')
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                         for k, v in r.items()} for r in rows)


def read_logs(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('intent', 'action', 'reward'):
            r[k] = int(r[k])
        r['b1'] = float(r['b1'])
    return rows


def percent(x):
    return f'{100 * x:.3f}%'


def write_findings(path, metrics):
    e, truth = metrics['evaluation'], metrics['oracle']
    lines = [
        '# Synthetic advertising policy evaluation', '',
        'These are constructed simulation results, not measured iPhone campaign effects.', '',
        'One action selects an entire four-ad bundle. The deployed policy chooses',
        'sequencing with probability 0.2 and repetition with probability 0.8.',
        'The fixed target always chooses sequencing. The harmful scenario deliberately',
        'makes sequencing slightly worse for both observed customer groups.', '',
        '| Comparison | Difference (percentage points) | 95% interval (pp) |',
        '|---|---:|---:|',
        f"| Naive before/after | {100 * metrics['naive']['difference']:.3f} | Not estimated |",
    ]
    for method in ('ipw', 'dr'):
        v = e[method]
        ci = v['difference_ci95']
        lines.append(f"| {method.upper()} on reference population | {100 * v['difference']:.3f} "
                     f"| [{100 * ci[0]:.3f}, {100 * ci[1]:.3f}] |")
    lines += [f"| Simulator oracle on reference population | {100 * truth['difference']:.3f} | Known probabilities |", '',
              f"Before observed conversion: {percent(metrics['naive']['before'])}; "
              f"after observed conversion: {percent(metrics['naive']['after'])}.", '',
              'Customers are sampled with 80% high-propensity probability later versus 20% before.',
              'That change makes the before/after comparison misleading. OPE uses only',
              'the original evaluation logs and standardizes both policies to those contexts.', '',
              f"Target IPW value: {percent(e['ipw']['target'])}; DR: {percent(e['dr']['target'])}; "
              f"oracle: {percent(truth['target'])}.", '',
              '## Learned policy', '',
              'A separate heterogeneous-effect scenario trains a tabular response model',
              'on independent customers and freezes the resulting policy before evaluation.',
              'This is statistical policy learning, not language-model fine-tuning.', '',
              f"Learned action by observed group: {metrics['learned']['policy']}.",
              f"Held-out DR improvement: {100 * metrics['learned']['evaluation']['dr']['difference']:.3f} pp.", '',
              '## Hidden-confounding stress test', '',
              '| Hidden-intent assignment strength | DR target | Oracle target | Bias (pp) |',
              '|---|---:|---:|---:|']
    for r in metrics['sensitivity']:
        lines.append(f"| {r['strength']} | {percent(r['dr_target'])} | {percent(r['oracle_target'])} | {100 * r['bias']:.3f} |")
    lines += ['', 'Hidden intent affects outcomes in every sensitivity run; its influence on',
              'assignment increases across the sweep. Evaluators receive only propensities',
              'estimated from observed context, not the hidden-intent sampling probabilities.',
              'This diagnoses bias under constructed assumptions; it does not correct real confounding.', '',
              '## Uncertainty and limitations', '', e['interval_scope'], '',
              f"Effective sample size: {e['diagnostics']['effective_sample_size']:.1f}; "
              f"maximum importance weight: {e['diagnostics']['weight_max']:.2f}.",
              'No weight clipping. Finite-sample IPW/DR estimates can lie outside [0, 1].',
              ('The DR difference interval includes zero; a negative point estimate alone '
               'does not establish a statistically conclusive decline.')
              if e['dr']['difference_ci95'][0] <= 0 <= e['dr']['difference_ci95'][1]
              else 'The DR difference interval excludes zero under the stated bootstrap assumptions.',
              'The baseline is an exploratory mostly-repetitive policy, not a repetition-only policy.',
              'Without exploration, sequencing is not identifiable from repetitive-only logs.',
              'Criteo and AgentSociety runtime results are separate; this report claims neither.',
              'No simulator oracle or later-period outcomes enter OPE or training.', '',
              'Reproduce using config.json; logging.csv and model.json also support the replay CLI.',
              f"Chart status: {metrics['chart_status']}.", '']
    Path(path).write_text('\n'.join(lines))


def plot_results(path, metrics):
    """Optional scientific plot with a standard plotting library."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'Not generated: optional matplotlib dependency is unavailable'
    e = metrics['evaluation']
    values = [metrics['naive']['difference'], e['ipw']['difference'],
              e['dr']['difference'], metrics['oracle']['difference']]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['Naive\n(population changes)', 'IPW', 'Doubly robust', 'Oracle'],
           [v * 100 for v in values], color=['#aab2be', '#3666d6', '#3666d6', '#36835d'])
    for index, method in ((1, 'ipw'), (2, 'dr')):
        low, high = e[method]['difference_ci95']
        ax.vlines(index, low * 100, high * 100, color='black', linewidth=2)
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_ylabel('Conversion difference (percentage points)')
    ax.set_title('Synthetic example: apparent improvement vs counterfactual effect')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return 'Generated comparison.png; intervals are paired customer bootstrap'
