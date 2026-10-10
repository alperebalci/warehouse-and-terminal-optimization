# Warehouse and Terminal Optimization

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary umbrella repository for this Jors Academy research area. Related projects have been consolidated under `projects/` so the methods, implementations, experiments, and case studies can be maintained and explored from one place.

### Native flagship

The repository root contains the actively maintained **container terminal yard assignment and handling MILP** described below. The entries under `projects/` are consolidated companion projects.

### Included projects

- [`freighter-container-loading-optimization`](projects/freighter-container-loading-optimization/)
- [`multi-period-warehouse-rental-lp-optimization`](projects/multi-period-warehouse-rental-lp-optimization/)
- [`warehouse-multi-agent-robot-routing`](projects/warehouse-multi-agent-robot-routing/)
- [`warehouse-order-packing-heuristics`](projects/warehouse-order-packing-heuristics/)

Each consolidated project keeps its own files and a `SOURCE_REPOSITORY.md` provenance record. The snapshot preserves the source repository's default-branch files at consolidation time; repository-level history and metadata remain separate from the snapshot.
<!-- portfolio-umbrella:end -->

Mixed-integer linear programming (MILP) example for container terminal yard operations.

The model jointly optimizes:

- yard-block assignment;
- discrete handling time;
- equipment throughput;
- reefer / hazardous / weight compatibility;
- TEU capacity;
- storage, handling, repositioning, and tardiness costs.

## Why this version exists

This project modernizes an older prototype whose service-time formulation was too weak and could silently omit containers that had no compatible block. The revised model validates feasibility before optimization and ties handling times directly to binary scheduling decisions.

## Model

For each container `i`, compatible block `j`, and feasible handling period `t`:

- `x[i,j] = 1` when container `i` is assigned to block `j`;
- `y[i,j,t] = 1` when it is handled in period `t`;
- `delay[i] >= 0` measures tardiness beyond its latest pickup/loading time.

Main constraints:

1. every container is assigned exactly once;
2. TEU assigned to each block does not exceed its capacity;
3. a selected block implies exactly one handling period;
4. handling moves per block/time period respect equipment throughput;
5. handling cannot occur before arrival or the earliest service time;
6. tardiness is computed from the selected handling period.

## Important modeling assumption

Yard capacity is a static planning-horizon approximation. A production-grade terminal model should use time-indexed occupancy so containers consume capacity only while physically present in the yard.

## Install

```bash
python -m pip install -r requirements.txt
```

CBC is normally bundled with PuLP on common desktop Python installations.

## Run

```bash
python container_terminal_optimizer.py
```

## Requirements

- Python 3.10+
- NumPy
- pandas
- Matplotlib
- PuLP

## Native terminal CI and acceptance criteria

[Container terminal MILP CI](.github/workflows/terminal-ci.yml) now executes the **actual PuLP/CBC mathematical program** on a small seeded instance on Python 3.11 and 3.12.

The regression suite checks deterministic data generation; valid dimensions and time grids; feasible optimization status; each container assigned exactly once; independent reefer, hazardous-cargo, and weight compatibility checks; equipment throughput and TEU limits; allowed handling times; lateness consistency; finite objective; and sensible summary utilization.

This small validation problem is not a proof of scalability, production deployment, or correctness of imported `projects/` snapshots. Industrial experiments need real terminal data, controlled budgets, sensitivity analysis and independent stress tests.
