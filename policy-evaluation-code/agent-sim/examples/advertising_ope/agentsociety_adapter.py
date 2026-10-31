"""Real Agent/Block adapter, directly scheduled without a city server or LLM.

Imports the installed/local AgentSociety framework only when run_agents is called.
It does not replace AgentSociety modules with stand-ins if dependencies are absent.
"""

import argparse
import asyncio
from pathlib import Path
import random

from .core import evaluate, fit_model, learned_policy
from .reporting import write_csv, write_json
from .simulation import customer_decision, oracle_values


def agent_classes():
    try:
        from agentsociety.agent.agent_base import Agent, AgentType
        from agentsociety.workflow.block import Block
    except (ImportError, OSError) as exc:
        raise RuntimeError(f'AgentSociety runtime unavailable: {exc}. Install the repository dependencies.') from exc

    class AdvertisingDecisionBlock(Block):
        def __init__(self, seed, customer_id, scenario, split):
            super().__init__(name='advertising-decision')
            self.rng = random.Random(seed)
            self.seed = seed
            self.customer_id, self.scenario, self.split = customer_id, scenario, split

        async def forward(self):
            row, oracle = customer_decision(self.rng, self.customer_id,
                                            scenario=self.scenario, split=self.split)
            row.update(experiment_id=f'agentsociety-{self.scenario}', seed=self.seed,
                       population_high_share=.2)
            return row, oracle

    class AdvertisingCustomerAgent(Agent):
        def __init__(self, name, seed, scenario='harmful', split='evaluation'):
            super().__init__(name=name, type=AgentType.Citizen)
            self.advertising_block = AdvertisingDecisionBlock(seed, name, scenario, split)
            self.observed_log = None
            self.oracle_log = None

        async def forward(self):
            if self.observed_log is not None:
                raise ValueError('One independent bundle decision per customer; create a new agent.')
            self.observed_log, self.oracle_log = await self.advertising_block.forward()

    return AdvertisingCustomerAgent, AdvertisingDecisionBlock


async def run_agents(n, seed, scenario='harmful', split='evaluation'):
    if n < 1:
        raise ValueError('At least one agent is required.')
    agent_class, _ = agent_classes()
    # Chunking bounds the number of live framework objects.
    rows, oracle = [], []
    for start in range(0, n, 100):
        agents = [agent_class(f'customer-{seed}-{i}', seed * 1000003 + i, scenario, split)
                  for i in range(start, min(start + 100, n))]
        await asyncio.gather(*(agent.forward() for agent in agents))
        rows.extend(agent.observed_log for agent in agents)
        oracle.extend(agent.oracle_log for agent in agents)
    return rows, oracle


async def experiment(n, seed, output, bootstrap):
    train, _ = await run_agents(n, seed, 'heterogeneous', 'train')
    logs, oracle = await run_agents(n, seed + 1, 'heterogeneous')
    model = fit_model(train)
    target = [learned_policy(model, r['intent']) for r in logs]
    result = dict(source='synthetic-agentsociety',
                  evaluation=evaluate(logs, target, model, bootstrap, seed),
                  oracle=oracle_values(oracle, target),
                  policy={str(x): learned_policy(model, x) for x in (0, 1)},
                  runtime='Actual Agent and Block subclasses; direct asyncio scheduling; no city server',
                  config=dict(n=n, seed=seed, bootstrap=bootstrap))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'training.csv', train)
    write_csv(output / 'logging.csv', logs)
    write_csv(output / 'oracle.csv', oracle)
    write_json(output / 'model.json', model)
    write_json(output / 'metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('examples/advertising_ope/output_agents'))
    args = parser.parse_args()
    try:
        asyncio.run(experiment(args.n, args.seed, args.output, args.bootstrap))
        print(f'AgentSociety logs and evaluation: {args.output}')
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
