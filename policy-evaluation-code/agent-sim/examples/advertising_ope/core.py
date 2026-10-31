"""Binary contextual-bandit estimators and a small fitted response model.

Only observed pre-action context, action, reward and behavior probabilities are
consumed. A row represents an independent customer, not an individual ad.
"""

import math
import random
import statistics


def probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Probabilities must be finite and between zero and one.')
    return value


def validate_rows(rows):
    if not rows:
        raise ValueError('At least one logged customer is required.')
    ids = set()
    for row in rows:
        for field in ('intent', 'action', 'reward'):
            if row[field] not in (0, 1):
                raise ValueError(f'{field} must be binary.')
        probability(row['b1'])
        chosen = row['b1'] if row['action'] else 1 - row['b1']
        if chosen <= 0:
            raise ValueError('Logged action has zero behavior probability.')
        if row['customer_id'] in ids:
            raise ValueError('Customer IDs must be unique for customer bootstrap.')
        ids.add(row['customer_id'])


def fit_model(rows):
    """Fit Laplace-smoothed conversion rates by observed intent and action.

Use independent training data. Missing context/action cells are rejected instead
of quietly extrapolating a learned policy into unsupported contexts.
"""
    validate_rows(rows)
    model = {}
    for intent in (0, 1):
        for action in (0, 1):
            cell = [r['reward'] for r in rows
                    if r['intent'] == intent and r['action'] == action]
            if not cell:
                raise ValueError(f'No training support for intent={intent}, action={action}.')
            model[f'{intent}:{action}'] = (sum(cell) + 1) / (len(cell) + 2)
    return model


def predict(model, intent, action):
    return probability(model[f'{intent}:{action}'])


def learned_policy(model, intent):
    """Choose the action with highest estimated conversion; ties use repetition."""
    return float(predict(model, intent, 1) > predict(model, intent, 0))


def quantile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_intervals(columns, repetitions, seed):
    """Paired customer bootstrap, conditional on frozen policy/nuisance models.

    Resample identical customer indices across all contribution columns; no
    normal-approximation substitution.
"""
    if not isinstance(repetitions, int) or repetitions < 20:
        raise ValueError('Use at least 20 bootstrap repetitions (200+ recommended).')
    rng = random.Random(seed)
    n = len(columns[0])
    samples = [[] for _ in columns]
    # Sample the SAME customers for all policies/estimators in each replicate.
    for _ in range(repetitions):
        totals = [0.] * len(columns)
        for _ in range(n):
            index = rng.randrange(n)
            for j, column in enumerate(columns):
                totals[j] += column[index]
        for j, total in enumerate(totals):
            samples[j].append(total / n)
    return [[quantile(s, .025), quantile(s, .975)] for s in samples]


def evaluate(rows, target, model, bootstrap=200, seed=42, predictions=None,
             behavior_probabilities_known=True):
    """Estimate target and behavior values on the logged context population.

target[i] is P(target chooses action 1 | context_i). predictions optionally
contains independently fitted (m0, m1) pairs, for continuous-feature models.
Intervals condition on these fitted models and on a frozen target policy.
No clipping is applied; finite-sample IPW/DR values may exceed [0, 1].
    Set behavior_probabilities_known=False for estimated propensities: the DR
    comparator must then be the observed logging mean, not the fitted policy.
"""
    validate_rows(rows)
    if len(target) != len(rows):
        raise ValueError('One target probability is required per logged customer.')
    if predictions is not None and len(predictions) != len(rows):
        raise ValueError('One outcome prediction pair is required per customer.')
    columns = [[] for _ in range(6)]
    weights = []
    for i, (r, t) in enumerate(zip(rows, target)):
        t = probability(t)
        b = probability(r['b1'])
        if (t > 0 and b == 0) or (t < 1 and b == 1):
            raise ValueError('Target policy lacks behavior-policy support (overlap).')
        a, y = r['action'], r['reward']
        m0, m1 = (predictions[i] if predictions is not None else
                  (predict(model, r['intent'], 0), predict(model, r['intent'], 1)))
        m0, m1 = probability(m0), probability(m1)
        w = (t if a else 1 - t) / (b if a else 1 - b)
        ma = m1 if a else m0
        ipw = w * y
        dr = (1 - t) * m0 + t * m1 + w * (y - ma)
        if behavior_probabilities_known:
            dr_logging = (1 - b) * m0 + b * m1 + y - ma
            # Algebraic paired difference avoids cancellation for identical policies.
            dr_difference = (t - b) * (m1 - m0) + (w - 1) * (y - ma)
        else:
            # Estimated b is a nuisance model, not a definition of actual behavior.
            dr_logging = y
            dr_difference = dr - y
        values = (ipw, y, ipw - y, dr, dr_logging, dr_difference)
        for column, value in zip(columns, values):
            column.append(value)
        weights.append(w)
    intervals = bootstrap_intervals(columns, bootstrap, seed)
    result = {'n': len(rows), 'observed_logging_mean': statistics.mean(r['reward'] for r in rows),
              'behavior_probabilities_known': behavior_probabilities_known,
              'dr_logging_reference': ('Known logging policy with outcome augmentation'
                                       if behavior_probabilities_known else 'Observed logging reward mean')}
    for name, offset in (('ipw', 0), ('dr', 3)):
        result[name] = {}
        for field, j in (('target', offset), ('logging', offset + 1), ('difference', offset + 2)):
            result[name][field] = statistics.mean(columns[j])
            result[name][field + '_ci95'] = intervals[j]
    total, squares = sum(weights), sum(w * w for w in weights)
    result['diagnostics'] = {
        'action_counts': {str(a): sum(r['action'] == a for r in rows) for a in (0, 1)},
        'propensity_min': min(min(r['b1'], 1 - r['b1']) for r in rows),
        'weight_max': max(weights), 'weight_p50': quantile(weights, .5),
        'weight_p95': quantile(weights, .95), 'weight_p99': quantile(weights, .99),
        'effective_sample_size': total * total / squares if squares else 0.,
        'nonzero_target_weight_customers': sum(w > 0 for w in weights),
        'warnings': (['Fewer than 30 effective customers; intervals may be unreliable.']
                     if (total * total / squares if squares else 0.) < 30 else []),
        'weight_clipping': False,
    }
    result['interval_scope'] = ('Paired customer percentile bootstrap; conditional on frozen '
                                'target policy and fitted nuisance models. Does not cover '
                                'training uncertainty or simulator assumptions.')
    return result
