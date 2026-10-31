"""Explicit synthetic advertising assumptions, separate from estimator inputs."""

import math
import random

from .core import probability

MESSAGES = {
    0: ['Buy iPhone'] * 4,
    1: ['iPhone features', 'iPhone vs Samsung comparison',
        'Customer reviews and ratings', 'Limited-time discount offer'],
}
SCENARIOS = {
    'harmful': ((.02, .01), (.20, .18)),
    'helpful': ((.02, .06), (.20, .24)),
    'heterogeneous': ((.02, .06), (.20, .15)),
    'null': ((.02, .02), (.20, .20)),
}


def customer_decision(rng, customer_id, high_share=.2, scenario='harmful',
                      confounding=0., target_probability=None, split='evaluation'):
    intent = int(rng.random() < high_share)  # observed binary customer propensity proxy
    u = int(rng.random() < .5)  # unobserved purchase intent, independent of observed proxy
    # Both observed groups have support. Hidden intent affects action only in stress tests.
    log_odds = math.log(.2 / .8) + confounding * (2 * u - 1)
    b1 = 1 / (1 + math.exp(-log_odds))
    if target_probability is not None:
        b1 = target_probability
    action = int(rng.random() < b1)
    p0, p1 = SCENARIOS[scenario][intent]
    # Average multiplier is one; U affects outcomes even at zero confounding.
    p0, p1 = p0 * (.2 + 1.6 * u), p1 * (.2 + 1.6 * u)
    reward = int(rng.random() < (p1 if action else p0))
    row = dict(customer_id=customer_id, intent=intent, action=action, reward=reward,
               b1=b1, b0=1 - b1, propensity=b1 if action else 1 - b1,
               messages=MESSAGES[action], split=split, scenario=scenario,
               policy_version='logging-v1' if target_probability is None else 'fixed-target-v1')
    oracle = dict(customer_id=customer_id, latent_intent=u, p0=p0, p1=p1, true_b1=b1)
    return row, oracle


def simulate(n, seed, high_share=.2, scenario='harmful', confounding=0.,
             target_probability=None, split='evaluation'):
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer.')
    probability(high_share)
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {scenario}. Choose {list(SCENARIOS)}.')
    if not math.isfinite(confounding) or not 0 <= confounding <= 10:
        raise ValueError('Confounding strength must be finite and between 0 and 10.')
    if target_probability is not None:
        probability(target_probability)
    rng = random.Random(seed)
    pairs = [customer_decision(rng, f'{seed}:{i}', high_share, scenario, confounding,
                               target_probability, split) for i in range(n)]
    for row, _ in pairs:
        row.update(experiment_id=f'{scenario}-{seed}', seed=seed,
                   population_high_share=high_share)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def oracle_values(oracle, target):
    """Validation only: never passed to training or the numerical evaluator."""
    if not oracle or len(oracle) != len(target):
        raise ValueError('Oracle contexts and target probabilities must align.')
    target_value = sum((1 - probability(t)) * r['p0'] + t * r['p1']
                       for r, t in zip(oracle, target)) / len(oracle)
    logging_value = sum((1 - r['true_b1']) * r['p0'] + r['true_b1'] * r['p1']
                        for r in oracle) / len(oracle)
    return dict(target=target_value, logging=logging_value, difference=target_value - logging_value)


def estimate_behavior(training_rows):
    """Observed-context propensity fit for the confounded stress test only."""
    result = {}
    for intent in (0, 1):
        cell = [r['action'] for r in training_rows if r['intent'] == intent]
        if not cell:
            raise ValueError('Both observed intent groups are required for propensity fitting.')
        result[str(intent)] = (sum(cell) + 1) / (len(cell) + 2)
    return result
