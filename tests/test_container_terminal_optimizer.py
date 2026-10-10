"""Numerical and feasibility regression tests for the native terminal MILP.

These tests exercise PuLP/CBC on small seeded terminal instances, not only
imports. They do not assert industrial-scale performance.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
import pulp

from container_terminal_optimizer import ContainerTerminalOptimizer


def test_seeded_terminal_instance_reproducible() -> None:
    left = ContainerTerminalOptimizer(n_containers=8, seed=17)
    right = ContainerTerminalOptimizer(n_containers=8, seed=17)
    pd.testing.assert_frame_equal(left.containers, right.containers)
    assert len(left.containers) == 8
    assert left.containers["container_id"].is_unique
    assert (left.containers["teu_equivalent"] > 0).all()


def test_time_grid_rejects_invalid_horizons() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ContainerTerminalOptimizer(n_containers=2, planning_horizon=71, period_hours=2)
    with pytest.raises(ValueError, match="positive"):
        ContainerTerminalOptimizer(n_containers=2, planning_horizon=0)


def test_exact_small_milp_and_independent_feasibility() -> None:
    terminal = ContainerTerminalOptimizer(
        n_containers=8,
        planning_horizon=72,
        period_hours=2,
        seed=42,
    )
    terminal.build_model()
    assert terminal.model.sense == pulp.LpMinimize
    solved = terminal.solve(time_limit_seconds=45, relative_gap=0.001)
    assert solved, f"CBC did not return an optimal feasible solution: {terminal.status}"
    rows = terminal.solution_df
    assert len(rows) == terminal.n_containers
    assert rows["container_id"].is_unique
    assert math.isfinite(float(pulp.value(terminal.model.objective)))
    assert (rows["delay"] >= -1e-6).all()

    # Recheck constraints from the raw input attributes, independently of
    # the model's variable values and extraction routines.
    for row in rows.itertuples(index=False):
        block = terminal.blocks[row.assigned_block]
        assert (not row.is_reefer) or block.has_power
        assert (not row.is_hazardous) or block.has_safety
        assert row.weight <= block.weight_limit_t + 1e-6
        assert row.service_time in terminal.time_periods
        source = terminal.containers.loc[
            terminal.containers["container_id"] == row.container_id
        ].iloc[0]
        assert row.service_time + 1e-6 >= max(
            float(source["arrival_time"]), float(source["earliest_pickup"])
        )
        assert row.delay + 1e-6 >= max(
            0.0, row.service_time - float(source["latest_pickup"])
        )

    for block_id, block in terminal.blocks.items():
        used = rows.loc[rows["assigned_block"] == block_id, "teu_equivalent"].sum()
        assert used <= block.capacity_teu + 1e-6
        block_rows = rows[rows["assigned_block"] == block_id]
        for period in terminal.time_periods:
            movements = int((block_rows["service_time"] == period).sum())
            assert movements <= block.moves_per_hour * terminal.period_hours

    summary = terminal.summary()
    assert len(summary) == len(terminal.blocks)
    assert (summary["utilization_pct"] <= 100 + 1e-6).all()
