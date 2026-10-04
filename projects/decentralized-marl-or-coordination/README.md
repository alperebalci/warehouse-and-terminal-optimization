# Local-Actor MARL with OR Coordination

Implemented bounded research baseline, v0.1, 2026-10-01.

A shared actor observes only its own position, goal displacement, four adjacent-cell sensors and time. The critic sees the joint state during training. A regression test changes a distant robot's position and goal while checking that the first actor's observation does not change.

Training uses centralized-critic PPO with a factored joint action distribution, a clipped JOINT likelihood ratio and Monte Carlo returns. This is a compact CTDE implementation, not a paper-level MAPPO reproduction. Deployment of the actor does not require the critic.

A separate one-step HiGHS MILP chooses legal joint actions maximizing supplied scores subject to vertex and edge-swap conflict constraints. The shield DOES have global information and is not claimed to be decentralized. A two-agent exhaustive oracle checks its objective; random fixtures check collision avoidance. The physical simulator blocks conflicting attempted moves to closure, so metrics report blocked attempts rather than realized physical collisions.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest checks -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python study.py --output local-results.json
```

Core API: Warehouse, CTDE, train, shield(environment,scores), evaluate(model). Eight checks passed locally. Three training seeds and eight held-out episodes compare local greedy, local PPO, PPO plus shield and centralized distance-score coordination.

The short PPO runs do NOT beat the strong classical coordinator. Shield interventions and blocked attempts are recorded independently; a gain caused by the centralized shield is not credited to a decentralized actor. Observed results are included without a universal algorithm ranking.

## Scope

Three homogeneous robots, a 5x5 grid with a central obstacle, static one-goal tasks and a 12-step horizon. No dynamic orders, communication protocol, variable-fleet generalization, multi-step conflict-based search or deadlock-freedom guarantee. Waiting remains feasible, but may prevent progress. This module is adjacent to, and does not replace, the existing RLlib warehouse project.

Primary research reference: https://arxiv.org/abs/2103.01955

Independent implementation. Existing repository license applies. Source fingerprints and local environment are in VALIDATION.json; tests execute independently of root pytest discovery.
