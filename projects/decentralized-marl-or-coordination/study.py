"""Centralized-critic/local-actor PPO and a separately measured one-step MILP shield."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from torch import nn

MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])


class Warehouse:
    def __init__(self, seed=0, robots=3, size=5, horizon=12):
        if robots < 2 or size < 3 or horizon < 1:
            raise ValueError("Invalid environment dimensions")
        self.size = size
        self.horizon = horizon
        self.time = 0
        self.walls = {(size // 2, size // 2)}
        cells = [
            (r, c) for r in range(size) for c in range(size) if (r, c) not in self.walls
        ]
        if robots > len(cells):
            raise ValueError("Too many robots")
        g = np.random.default_rng(seed)
        self.pos = np.array(
            [cells[i] for i in g.choice(len(cells), robots, replace=False)]
        )
        self.targets = np.array(
            [cells[i] for i in g.choice(len(cells), robots, replace=False)]
        )

    def destinations(self):
        return self.pos[:, None, :] + MOVES[None, :, :]

    def masks(self):
        dest = self.destinations()
        valid = ((dest >= 0) & (dest < self.size)).all(axis=2)
        for i in range(len(self.pos)):
            for a in range(5):
                valid[i, a] &= tuple(dest[i, a]) not in self.walls
            if np.array_equal(self.pos[i], self.targets[i]):
                valid[i, 1:] = False
        return valid

    def local(self):
        obs = []
        for i, p in enumerate(self.pos):
            sensors = []
            for step in MOVES[1:]:
                q = p + step
                sensors.append(
                    -1
                    if np.any(q < 0) or np.any(q >= self.size) or tuple(q) in self.walls
                    else int(
                        any(
                            np.array_equal(q, v)
                            for j, v in enumerate(self.pos)
                            if i != j
                        )
                    )
                )
            obs.append(
                [
                    *(p / (self.size - 1)),
                    *((self.targets[i] - p) / (self.size - 1)),
                    *sensors,
                    self.time / self.horizon,
                ]
            )
        return np.asarray(obs, dtype=np.float32)

    def global_state(self):
        return np.r_[
            self.pos.flatten() / (self.size - 1),
            self.targets.flatten() / (self.size - 1),
            self.time / self.horizon,
        ].astype(np.float32)

    def step(self, actions):
        a = np.asarray(actions)
        if a.shape != (len(self.pos),) or not np.isin(a, range(5)).all():
            raise ValueError("Action vector")
        mask = self.masks()
        proposed = self.pos + MOVES[a.astype(int)]
        blocked = ~mask[np.arange(len(a)), a.astype(int)]
        # Block conflicts to closure, including moves into newly blocked robots.
        for _ in range(len(a) + 1):
            q = np.where(blocked[:, None], self.pos, proposed)
            new = blocked.copy()
            for i, j in itertools.combinations(range(len(a)), 2):
                same = np.array_equal(q[i], q[j])
                swap = np.array_equal(q[i], self.pos[j]) and np.array_equal(
                    q[j], self.pos[i]
                )
                if same or swap:
                    new[i] = new[j] = True
            if np.array_equal(new, blocked):
                break
            blocked = new
        before = np.abs(self.pos - self.targets).sum()
        self.pos = np.where(blocked[:, None], self.pos, proposed)
        self.time += 1
        if len({tuple(p) for p in self.pos}) != len(a):
            raise RuntimeError("Collision resolution failed")
        distance = np.abs(self.pos - self.targets).sum()
        done = bool(distance == 0 or self.time >= self.horizon)
        reward = float(before - distance - 0.1 - 0.5 * blocked.sum())
        return reward, done, int(blocked.sum())


def shield(env, scores):
    scores = np.asarray(scores, float)
    n = len(env.pos)
    if scores.shape != (n, 5) or not np.isfinite(scores).all():
        raise ValueError("Finite action scores required")
    dest = env.destinations()
    valid = env.masks()
    rows = []
    lb = []
    ub = []

    def add(cols, local_value=-np.inf, u=np.inf):
        row = np.zeros(n * 5)
        row[cols] = 1
        rows.append(row)
        lb.append(local_value)
        ub.append(u)

    for i in range(n):
        add(list(range(i * 5, (i + 1) * 5)), 1, 1)
    cells = {tuple(dest[i, a]) for i in range(n) for a in range(5) if valid[i, a]}
    for cell in cells:
        add(
            [
                i * 5 + a
                for i in range(n)
                for a in range(5)
                if valid[i, a] and tuple(dest[i, a]) == cell
            ],
            u=1,
        )
    for i, j in itertools.combinations(range(n), 2):
        for a, b in itertools.product(range(1, 5), repeat=2):
            if np.array_equal(dest[i, a], env.pos[j]) and np.array_equal(
                dest[j, b], env.pos[i]
            ):
                add([i * 5 + a, j * 5 + b], u=1)
    r = milp(
        -scores.flatten(),
        integrality=np.ones(n * 5),
        bounds=Bounds(0, valid.flatten().astype(float)),
        constraints=LinearConstraint(np.array(rows), lb, ub),
        options={"time_limit": 5.0, "mip_rel_gap": 0.0},
    )
    if r.status != 0 or r.x is None:
        raise RuntimeError("Shield could not establish its bounded optimum")
    return r.x.reshape(n, 5).argmax(axis=1)


class CTDE(nn.Module):
    def __init__(self, robots=3):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(9, 32), nn.Tanh(), nn.Linear(32, 5))
        self.critic = nn.Sequential(
            nn.Linear(4 * robots + 1, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def distribution(self, local, mask):
        logits = self.actor(local).masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)


def train(seed=0, rounds=24):
    torch.manual_seed(seed)
    m = CTDE()
    opt = torch.optim.Adam(m.parameters(), lr=0.003)
    for iteration in range(rounds):
        local = []
        glob = []
        masks = []
        actions = []
        oldlog = []
        returns = []
        for episode in range(4):
            e = Warehouse(seed + iteration * 4 + episode)
            rewards = []
            while True:
                local_value = torch.tensor(e.local())
                g = torch.tensor(e.global_state())
                ma = torch.tensor(e.masks())
                with torch.no_grad():
                    dist = m.distribution(local_value, ma)
                    a = dist.sample()
                    log = dist.log_prob(a).sum()
                r, done, _ = e.step(a.numpy())
                local.append(local_value)
                glob.append(g)
                masks.append(ma)
                actions.append(a)
                oldlog.append(log)
                rewards.append(r)
                if done:
                    break
            ret = 0
            vals = []
            for r in reversed(rewards):
                ret = r + 0.95 * ret
                vals.append(ret)
            returns += vals[::-1]
        L, G, M, A, OLD = map(torch.stack, (local, glob, masks, actions, oldlog))
        R = torch.tensor(returns)
        with torch.no_grad():
            adv = R - m.critic(G).squeeze(1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        for _ in range(4):
            dist = m.distribution(L, M)
            log = dist.log_prob(A).sum(1)
            ratio = torch.exp(log - OLD)
            pg = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            value = ((m.critic(G).squeeze(1) - R) ** 2).mean()
            loss = pg + 0.5 * value - 0.01 * dist.entropy().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite PPO loss")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m.eval()


def evaluate(m, episodes=8):
    rows = []
    for method in (
        "local_greedy",
        "local_ppo",
        "local_ppo_with_shield",
        "central_distance_shield",
    ):
        deliveries = []
        conflicts = 0
        interventions = 0
        decisions = 0
        for seed in range(800, 800 + episodes):
            e = Warehouse(seed)
            while True:
                with torch.no_grad():
                    scores = m.actor(torch.tensor(e.local())).numpy()
                distance = (
                    -np.abs(e.destinations() - e.targets[:, None, :])
                    .sum(2)
                    .astype(float)
                )
                if method == "local_greedy":
                    scores = distance
                intended = np.where(e.masks(), scores, -1e9).argmax(1)
                if method == "local_ppo_with_shield":
                    a = shield(e, scores)
                elif method == "central_distance_shield":
                    a = shield(e, distance)
                else:
                    a = intended
                if method == "local_ppo_with_shield":
                    interventions += int(np.sum(a != intended))
                _, done, blocked = e.step(a)
                conflicts += blocked
                decisions += len(a)
                if done:
                    break
            deliveries.append(int(np.all(e.pos == e.targets, axis=1).sum()))
        rows.append(
            {
                "method": method,
                "mean_delivered": float(np.mean(deliveries)),
                "blocked_attempts": conflicts,
                "shield_interventions": interventions,
                "agent_decisions": decisions,
            }
        )
    return rows


def benchmark():
    torch.set_num_threads(1)
    return {
        "scope": "3 robots; finite episodes; static tasks; CTDE PPO, not paper-level MAPPO",
        "shield_has_global_information_and_is_not_part_of_local_actor": True,
        "results": [
            {"seed": s, "evaluation": evaluate(train(s))} for s in (41, 42, 43)
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results.json")
    args = p.parse_args()
    Path(args.output).write_text(json.dumps(benchmark(), indent=2))
