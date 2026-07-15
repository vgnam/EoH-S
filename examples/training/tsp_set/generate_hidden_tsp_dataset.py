from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


DEFAULT_SCHEDULE = [
    "uniform",
    "cluster",
    "bezier",
    "rings",
    "spiral",
    "grid_holes",
    "stripes",
    "moons",
    "mixed_structures",
    "star",
    "nested_boxes",
    "corner_blobs",
    "snake",
    "cross",
]
MIXED_REGIME = "mixed_ood"


def _clip(coords):
    return np.clip(np.asarray(coords, dtype=float), 0.0, 1.0)


def sample_uniform(rng, n_cities):
    return _clip(rng.uniform(0.0, 1.0, size=(n_cities, 2)))


def sample_cluster(rng, n_cities):
    cluster_count = int(rng.integers(3, 9))
    centers = rng.uniform(0.12, 0.88, size=(cluster_count, 2))
    probs = rng.dirichlet(np.ones(cluster_count))
    assignments = rng.choice(cluster_count, size=n_cities, p=probs)
    scale = rng.uniform(0.025, 0.075)
    coords = centers[assignments] + rng.normal(0.0, scale, size=(n_cities, 2))
    return _clip(coords)


def sample_bezier(rng, n_cities):
    control = rng.uniform(0.08, 0.92, size=(4, 2))
    t = np.sort(rng.uniform(0.0, 1.0, size=n_cities))
    one_minus = 1.0 - t
    coords = (
        (one_minus**3)[:, None] * control[0]
        + (3.0 * one_minus**2 * t)[:, None] * control[1]
        + (3.0 * one_minus * t**2)[:, None] * control[2]
        + (t**3)[:, None] * control[3]
    )
    coords += rng.normal(0.0, rng.uniform(0.015, 0.045), size=coords.shape)
    if rng.random() < 0.35:
        extra_count = max(1, n_cities // 5)
        replace = rng.choice(n_cities, size=extra_count, replace=False)
        coords[replace] = rng.uniform(0.0, 1.0, size=(extra_count, 2))
    return _clip(coords)


def sample_rings(rng, n_cities):
    center = rng.uniform(0.42, 0.58, size=2)
    ring_choices = rng.choice([0.18, 0.32, 0.43], size=n_cities, p=[0.35, 0.45, 0.20])
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_cities)
    radius = ring_choices + rng.normal(0.0, 0.018, size=n_cities)
    aspect = rng.uniform(0.75, 1.35)
    coords = np.column_stack([aspect * radius * np.cos(theta), radius * np.sin(theta)])
    rotation = rng.uniform(0.0, np.pi)
    rot = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
        dtype=float,
    )
    coords = coords @ rot.T + center
    return _clip(coords)


def sample_spiral(rng, n_cities):
    turns = rng.uniform(1.5, 3.2)
    theta = np.sort(rng.uniform(0.0, turns * 2.0 * np.pi, size=n_cities))
    radius = np.linspace(0.04, rng.uniform(0.38, 0.48), n_cities)
    radius += rng.normal(0.0, 0.012, size=n_cities)
    coords = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    rotation = rng.uniform(0.0, 2.0 * np.pi)
    rot = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
        dtype=float,
    )
    coords = coords @ rot.T + rng.uniform(0.45, 0.55, size=2)
    coords += rng.normal(0.0, 0.015, size=coords.shape)
    return _clip(coords)


def sample_grid_holes(rng, n_cities):
    grid_side = int(np.ceil(np.sqrt(n_cities * 2.2)))
    xs = np.linspace(0.08, 0.92, grid_side)
    ys = np.linspace(0.08, 0.92, grid_side)
    grid = np.array([(x, y) for x in xs for y in ys], dtype=float)
    holes = rng.uniform(0.18, 0.82, size=(rng.integers(2, 5), 2))
    hole_radius = rng.uniform(0.08, 0.16, size=len(holes))
    keep = np.ones(len(grid), dtype=bool)
    for center, radius in zip(holes, hole_radius):
        keep &= np.linalg.norm(grid - center, axis=1) > radius
    candidates = grid[keep]
    if len(candidates) < n_cities:
        candidates = grid
    indices = rng.choice(len(candidates), size=n_cities, replace=len(candidates) < n_cities)
    coords = candidates[indices] + rng.normal(0.0, 0.012, size=(n_cities, 2))
    shear = rng.uniform(-0.25, 0.25)
    coords[:, 0] = coords[:, 0] + shear * (coords[:, 1] - 0.5)
    return _clip(coords)


def sample_stripes(rng, n_cities):
    stripe_count = int(rng.integers(3, 7))
    assignments = rng.integers(0, stripe_count, size=n_cities)
    along = rng.uniform(0.05, 0.95, size=n_cities)
    offsets = np.linspace(0.12, 0.88, stripe_count)[assignments]
    coords = np.column_stack([along, offsets + rng.normal(0.0, 0.018, size=n_cities)])
    rotation = rng.uniform(0.0, np.pi)
    rot = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
        dtype=float,
    )
    coords = (coords - 0.5) @ rot.T + 0.5
    coords += rng.normal(0.0, 0.008, size=coords.shape)
    return _clip(coords)


def sample_moons(rng, n_cities):
    first = n_cities // 2
    second = n_cities - first
    theta1 = rng.uniform(0.0, np.pi, size=first)
    theta2 = rng.uniform(0.0, np.pi, size=second)
    moon1 = np.column_stack([0.28 * np.cos(theta1), 0.18 * np.sin(theta1)])
    moon2 = np.column_stack([0.28 * np.cos(theta2) + 0.22, -0.18 * np.sin(theta2) + 0.12])
    coords = np.vstack([moon1, moon2])
    rotation = rng.uniform(-0.8, 0.8)
    rot = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
        dtype=float,
    )
    coords = coords @ rot.T + rng.uniform(0.38, 0.58, size=2)
    coords += rng.normal(0.0, 0.018, size=coords.shape)
    return _clip(coords)


def sample_mixed_structures(rng, n_cities):
    counts = rng.multinomial(n_cities, [0.35, 0.25, 0.25, 0.15])
    parts = []
    if counts[0]:
        parts.append(sample_rings(rng, counts[0]))
    if counts[1]:
        parts.append(sample_spiral(rng, counts[1]))
    if counts[2]:
        center = rng.uniform(0.2, 0.8, size=2)
        parts.append(_clip(center + rng.normal(0.0, 0.06, size=(counts[2], 2))))
    if counts[3]:
        t = rng.uniform(0.0, 1.0, size=counts[3])
        line = np.column_stack([t, 0.15 + 0.7 * t])
        parts.append(_clip(line + rng.normal(0.0, 0.025, size=line.shape)))
    coords = np.vstack(parts)
    rng.shuffle(coords)
    return _clip(coords[:n_cities])


def sample_star(rng, n_cities):
    spokes = int(rng.integers(5, 9))
    assignments = rng.integers(0, spokes, size=n_cities)
    base_angles = np.linspace(0.0, 2.0 * np.pi, spokes, endpoint=False)
    theta = base_angles[assignments] + rng.normal(0.0, 0.045, size=n_cities)
    radius = rng.uniform(0.06, 0.46, size=n_cities)
    coords = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    coords += rng.normal(0.0, 0.012, size=coords.shape)
    rotation = rng.uniform(0.0, 2.0 * np.pi)
    rot = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
        dtype=float,
    )
    coords = coords @ rot.T + rng.uniform(0.46, 0.54, size=2)
    return _clip(coords)


def sample_nested_boxes(rng, n_cities):
    levels = rng.choice([0.18, 0.31, 0.43], size=n_cities, p=[0.30, 0.45, 0.25])
    side = rng.integers(0, 4, size=n_cities)
    u = rng.uniform(-1.0, 1.0, size=n_cities)
    coords = np.zeros((n_cities, 2), dtype=float)
    coords[side == 0] = np.column_stack([u[side == 0], np.ones(np.sum(side == 0))])
    coords[side == 1] = np.column_stack([u[side == 1], -np.ones(np.sum(side == 1))])
    coords[side == 2] = np.column_stack([np.ones(np.sum(side == 2)), u[side == 2]])
    coords[side == 3] = np.column_stack([-np.ones(np.sum(side == 3)), u[side == 3]])
    coords *= levels[:, None]
    coords += rng.normal(0.0, 0.012, size=coords.shape)
    rotation = rng.uniform(0.0, np.pi)
    rot = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
        dtype=float,
    )
    coords = coords @ rot.T + rng.uniform(0.45, 0.55, size=2)
    return _clip(coords)


def sample_corner_blobs(rng, n_cities):
    centers = np.array(
        [[0.08, 0.08], [0.08, 0.92], [0.92, 0.08], [0.92, 0.92], [0.5, 0.5]],
        dtype=float,
    )
    probs = rng.dirichlet([1.8, 1.8, 1.8, 1.8, 0.7])
    assignments = rng.choice(len(centers), size=n_cities, p=probs)
    coords = centers[assignments] + rng.normal(0.0, rng.uniform(0.025, 0.055), size=(n_cities, 2))
    bridge_count = max(1, n_cities // 8)
    bridge = np.column_stack(
        [
            np.linspace(0.08, 0.92, bridge_count),
            np.linspace(0.92, 0.08, bridge_count),
        ]
    )
    replace = rng.choice(n_cities, size=bridge_count, replace=False)
    coords[replace] = bridge + rng.normal(0.0, 0.018, size=bridge.shape)
    return _clip(coords)


def sample_snake(rng, n_cities):
    t = np.sort(rng.uniform(0.0, 1.0, size=n_cities))
    waves = rng.uniform(2.0, 5.0)
    amplitude = rng.uniform(0.18, 0.32)
    coords = np.column_stack(
        [
            0.08 + 0.84 * t,
            0.5 + amplitude * np.sin(2.0 * np.pi * waves * t + rng.uniform(0.0, 2.0 * np.pi)),
        ]
    )
    coords += rng.normal(0.0, 0.018, size=coords.shape)
    if rng.random() < 0.5:
        coords = coords[:, ::-1]
    return _clip(coords)


def sample_cross(rng, n_cities):
    counts = rng.multinomial(n_cities, [0.40, 0.40, 0.20])
    parts = []
    if counts[0]:
        x = rng.uniform(0.05, 0.95, size=counts[0])
        parts.append(np.column_stack([x, 0.5 + rng.normal(0.0, 0.025, size=counts[0])]))
    if counts[1]:
        y = rng.uniform(0.05, 0.95, size=counts[1])
        parts.append(np.column_stack([0.5 + rng.normal(0.0, 0.025, size=counts[1]), y]))
    if counts[2]:
        t = rng.uniform(0.05, 0.95, size=counts[2])
        parts.append(np.column_stack([t, t]) + rng.normal(0.0, 0.02, size=(counts[2], 2)))
    coords = np.vstack(parts)
    rotation = rng.uniform(-0.45, 0.45)
    rot = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]],
        dtype=float,
    )
    coords = (coords - 0.5) @ rot.T + 0.5
    rng.shuffle(coords)
    return _clip(coords)


def sample_hidden_regime(rng, n_cities, regime):
    if regime == MIXED_REGIME:
        regime = str(rng.choice(DEFAULT_SCHEDULE))
    if regime == "uniform":
        return sample_uniform(rng, n_cities)
    if regime == "cluster":
        return sample_cluster(rng, n_cities)
    if regime == "bezier":
        return sample_bezier(rng, n_cities)
    if regime == "rings":
        return sample_rings(rng, n_cities)
    if regime == "spiral":
        return sample_spiral(rng, n_cities)
    if regime == "grid_holes":
        return sample_grid_holes(rng, n_cities)
    if regime == "stripes":
        return sample_stripes(rng, n_cities)
    if regime == "moons":
        return sample_moons(rng, n_cities)
    if regime == "mixed_structures":
        return sample_mixed_structures(rng, n_cities)
    if regime == "star":
        return sample_star(rng, n_cities)
    if regime == "nested_boxes":
        return sample_nested_boxes(rng, n_cities)
    if regime == "corner_blobs":
        return sample_corner_blobs(rng, n_cities)
    if regime == "snake":
        return sample_snake(rng, n_cities)
    if regime == "cross":
        return sample_cross(rng, n_cities)
    raise ValueError(f"Unknown hidden-test regime: {regime}")


def generate_hidden_dataset(seed, city_sizes, instances_per_size, schedule=None):
    schedule = list(schedule or DEFAULT_SCHEDULE)
    rng = np.random.default_rng(seed)
    rounds = []
    for round_id, regime in enumerate(schedule):
        coordinates = []
        instance_regimes = []
        for n_cities in city_sizes:
            if regime == MIXED_REGIME:
                regimes = [
                    DEFAULT_SCHEDULE[idx % len(DEFAULT_SCHEDULE)]
                    for idx in range(instances_per_size)
                ]
                rng.shuffle(regimes)
            else:
                regimes = [regime] * instances_per_size
            for instance_regime in regimes:
                coords = sample_hidden_regime(rng, n_cities, instance_regime)
                coordinates.append(coords)
                instance_regimes.append(instance_regime)
        rounds.append(
            {
                "round_id": round_id,
                "regime": regime,
                "instance_regimes": instance_regimes,
                "coordinates": coordinates,
            }
        )
    return {
        "format": "eohs-open-world-tsp-hidden-coords-v1",
        "seed": int(seed),
        "city_sizes": [int(size) for size in city_sizes],
        "instances_per_size": int(instances_per_size),
        "schedule": schedule,
        "rounds": rounds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=12026)
    parser.add_argument("--city-sizes", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--instances-per-size", type=int, default=128)
    parser.add_argument("--schedule", nargs="+", default=[MIXED_REGIME])
    args = parser.parse_args()

    dataset = generate_hidden_dataset(
        args.seed,
        args.city_sizes,
        args.instances_per_size,
        schedule=args.schedule,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    total = sum(len(item["coordinates"]) for item in dataset["rounds"])
    print(
        f"saved {args.output} with {len(dataset['rounds'])} rounds, "
        f"sizes={dataset['city_sizes']}, total_instances={total}"
    )


if __name__ == "__main__":
    main()
