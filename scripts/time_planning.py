"""Time each planning step of an exploration run.

Usage:

    ./env/bin/python scripts/time_planning.py OUT_PREFIX START_INDEX [tunable=value ...]

Runs one exploration from the given start location (1-based, as in
start_indices) without plotting, and times every call to plan(). Writes
OUT_PREFIX.csv with one row per planning step (step, plan seconds, number of
frontiers scored) and OUT_PREFIX.json with a summary. Set NUMBA_NUM_THREADS to
control parallelism.
"""

import csv
import json
import os
import sys
from time import perf_counter

import numpy as np

from da import main as M
from da.utils import parse_args, tunable


def main() -> None:
    out_prefix, start_index, *args = sys.argv[1:]
    # Evaluate tunables in da.main's namespace, as the exploration script does,
    # so that algorithm=DistanceAdvantage names the class.
    parse_args(args, locals={}, globals=vars(M))

    world = M.load_world()
    body, st = world.start_state(world.start_locations[int(start_index) - 1])
    alg = tunable('algorithm', M.DistanceAdvantage)(body, st)
    max_steps = tunable('max_steps', None)

    # Warm-up on a throwaway copy so JIT compilation isn't timed.
    warm = type(alg)(*world.start_state(world.start_locations[int(start_index) - 1]))
    warm.plan()

    rows = []
    while max_steps is None or len(rows) < max_steps:
        t0 = perf_counter()
        try:
            alg.plan()
        except M.ExplorationFinished:
            break
        dt = perf_counter() - t0
        rows.append((alg.state.step, dt, len(alg.frontier_scores)))
        alg.refine()
        alg.execute()

    with open(f'{out_prefix}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'plan_seconds', 'num_frontiers'])
        w.writerows(rows)

    times = np.array([r[1] for r in rows])
    summary = {
        'algorithm': type(alg).__name__,
        'start_index': int(start_index),
        'args': args,
        'numba_threads': os.environ.get('NUMBA_NUM_THREADS'),
        'finished': max_steps is None or len(rows) < max_steps,
        'distance_m': alg.state.cells_traveled * world.scale,
        'planning_steps': len(rows),
        'plan_seconds_total': float(times.sum()),
        'plan_ms_median': float(np.median(times) * 1e3),
        'plan_ms_p90': float(np.percentile(times, 90) * 1e3),
        'plan_ms_max': float(times.max() * 1e3),
    }
    with open(f'{out_prefix}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
