"""Proof-of-concept implementation of Distance Advantage

The easiest way to understand distance advantage is to rephrase exploration as
a degenerate MDP. The actions are moving (by shortest path) to a cell, and the
state is the pair the observed map M and the current cell u. By moving from u
to v, i.e., taking action v in cell u, the map is revealed. When there are no
more cells to reveal, the game is over. The reward r(u, v) for taking an action
v in cell u is minus the length, -d(u, v). The only actions considered are
frontier cells F(M), where a frontier cell is a free cell neighboring an
unexplored cell. There is no time dimension; the MDP terminates after one
action and a new MDP is solved in the succeeding state. The reward of an action
is therefore the return R.

Nearest frontier exploration can then be understood as a greedy policy on the
state-action value function Q(s, a):

    π_NF(s) = argmax_{a in F(M)} Q(s, a) with Q(s, a) = -d(s, a)

Distance advantage, as the name implies, simply takes the advantage instead,
i.e., instead of considering which action yields most reward, consider which
action yields more reward _in the current state_:

    π_DA(s) = argmax_{a in F(M)} Adv_s(a),

with advantage

    Adv_s(a) = Q(s, a) - V(s),

and the state value function V(s) is the expected return, which is

    V(s)    = Expect_v -d(s, v), and
    Q(s, a) =          -d(s, a).

Putting it together, we obtain distance advantage:

    Adv_s(a) = Expect_v d(s, v) - d(s, a)

"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from importlib import resources
from typing import IO, Any, Callable, Generator, Mapping, Sequence
from warnings import warn

import numpy as np
from scipy import ndimage
from tqdm.auto import tqdm

from . import lidar
from .gridmaps import CellIndex, CellState, Gridmap, indices_mask
from .utils import (edge_view, meshgrid_shape, parse_args, permute_nonzero,
                    savetxt_row, tunable)


@dataclass(frozen=True)
class World:
    """World (environment) representation.

    Coordinate convention is matplotlib. Vertices are at the centers of the
    gridmap cells, so in other words, a cell center is an integer coordinate.
    """

    origin:      np.ndarray  # meters
    bounds:      np.ndarray  # meters
    shape:       np.ndarray  # cells
    scale:       float       # meters/cell
    walls:       np.ndarray  # (N, 2, 2) in subpixel cells

    # Rendered walls
    grid:        Gridmap

    # Connected components of free space
    grid_labels: np.ndarray

    # Mask for interior of grid
    interior:    np.ndarray

    rng:         np.random.Generator

    # All valid starting locations, randomly permuted by rng
    start_locations: Sequence[CellIndex]

    @classmethod
    def loadtxt(cls, filename: str | IO, **kwargs) -> World:
        print('Loading map', filename)
        print()
        walls = np.loadtxt(filename, delimiter=',').reshape(-1, 2, 2)
        return cls.load(walls, **kwargs)

    @classmethod
    def load(cls, walls: np.ndarray, *, pad: float, cells_per_meter: float, rng: np.random.Generator) -> World:

        print(f'{walls.shape = }')

        if 0 < (num_random_triangles := tunable('num_random_triangles', 0)):

            lo = walls.min(axis=(0, 1))
            hi = walls.max(axis=(0, 1))
            side_length = 1.0
            n = num_random_triangles

            circumradius = side_length / np.sqrt(3)
            theta = rng.uniform(0, 2 * np.pi, size=n)
            theta0 = theta
            theta1 = theta + 2*np.pi/3
            theta2 = theta - 2*np.pi/3
            A = circumradius * np.c_[np.cos(theta0), np.sin(theta0)]
            B = circumradius * np.c_[np.cos(theta1), np.sin(theta1)]
            C = circumradius * np.c_[np.cos(theta2), np.sin(theta2)]

            assert np.allclose(np.linalg.norm(B-A, axis=-1), np.linalg.norm(A-C, axis=-1))
            assert np.allclose(np.linalg.norm(B-C, axis=-1), np.linalg.norm(A-B, axis=-1))

            center = rng.uniform(lo, hi, size=(n, 1, 2))
            triangles = center + np.stack((A, B, C, A), axis=1)

            walls = np.concatenate((walls, edge_view(triangles, flat=True)), axis=0)

        origin = np.min(walls, axis=(0, 1)) - pad
        bounds = np.ptp(walls, axis=(0, 1)) + 2*pad
        bounds = np.ceil(cells_per_meter*bounds)/cells_per_meter
        shape  = np.ceil(cells_per_meter*bounds).astype(int)
        scale  = float(max(bounds/shape))

        print()
        print(f'Origin: {tuple(origin)} meters')
        print(f'Bounds: {tuple(bounds)} meters')
        print()
        print(f'Scale:  {1/scale:.5g} cells/meter')
        print(f'Scale:  {tuple(shape/bounds)} cells/meter')
        print(f'Bounds: {tuple(bounds/scale)} cells')
        print(f'Shape:  {tuple(shape)} cells')
        print()

        # Center at origin and normalize to cells.
        walls -= origin
        walls /= scale

        # Here is a crucial step to align the grid boundary to the segments.
        walls -= 0.5

        grid = Gridmap.full(shape, CellState.FREE)
        grid.update_lines(CellState.OCCUPIED, *np.moveaxis(walls, 1, 0))
        grid.resolve_conflicting_states()

        grid_labels, num_labels = ndimage.label(grid == CellState.FREE)

        # Determine the exterior and interior by assuming the cell at (0,0) is
        # exterior and finding its connected component.
        assert grid[0, 0] == CellState.FREE
        exterior_label = grid_labels[0, 0]

        # Set exterior to background label (i.e. occupied)
        grid_labels[grid_labels == exterior_label] = 0

        # interior is the largest connected component.
        interior_label = np.argmax(np.bincount(grid_labels[0 < grid_labels]))

        interior = grid_labels == interior_label
        # outside(x) = not occupied(x) and not interior(x)
        outside = (grid_labels != 0) & (grid_labels != interior_label)
        grid[outside] = CellState.UNKNOWN

        start_xs, start_ys = permute_nonzero(interior, rng=rng)
        start_locations = list(zip(start_xs, start_ys))

        return cls(origin=origin, shape=shape, bounds=bounds, scale=scale,
                   walls=walls, grid=grid, grid_labels=grid_labels,
                   interior=interior, rng=rng, start_locations=start_locations)

    def start_state(world: World, start_location: CellIndex) -> tuple[Body, State]:
        body = Body(position=start_location, world=world)
        grid = Gridmap.full(world.shape, CellState.UNKNOWN)
        st = State(step=0, trajectory=[body.position], grid=grid)
        st, _ = st.integrate(body.scan())
        return body, st

    def nth_start_state(world: World, n: int) -> tuple[Body, State]:
        return world.start_state(world.start_locations[n])


def make_lidar() -> lidar.LIDAR:
    return lidar.LIDAR(r_max=tunable('lidar_r_max', 4.5) * tunable('cells_per_meter', 4.5),
                       num_points=tunable('lidar_num_points', 360))


@dataclass
class Body:
    "Mår magen bra, mår själen bra."

    position: CellIndex
    world: World
    sensor: lidar.LIDAR = field(default_factory=make_lidar)

    def move(body, target):
        if not body.line_of_sight(target):
            raise ValueError('move is a bonk')
        body.position = target
        return target

    def scan(body) -> lidar.LIDARScan:
        return body.sensor.scan(body.world.walls, *body.position)

    def line_of_sight(body, v: CellIndex) -> bool:
        u = body.position
        if u == v:
            return True
        vu       = (np.r_[v] - np.r_[u]).astype(np.float64)
        vu_norm  = np.linalg.norm(vu)
        vu      /= vu_norm
        r_max    = vu_norm
        ux, uy   = u
        vux, vuy = vu
        r        = lidar.min_x_int(r_max, ux, uy, vux, vuy, body.world.walls)
        return r_max <= r

    def predict(body: Body, grid: Gridmap) -> Gridmap:

        unknown = grid == CellState.UNKNOWN

        if (predictions := tunable('predictions', default=True)) is True:
            predict = unknown
        elif predictions is False:
            predict = np.zeros_like(unknown)
        else:
            predict = ndimage.binary_dilation(~unknown, mask=unknown, iterations=predictions)

        # Predictions start out as the current map.
        predicted = grid.copy()

        # Oracle predictor up to prediction distance.
        predicted[predict] = body.world.grid[predict]

        if tunable('unknown_is_free', False):
            # Anything beyond the prediction distance is assumed to be free.
            # This makes DA worse.
            predicted[unknown & ~predict] = CellState.FREE

        return predicted


@dataclass(frozen=True, slots=True)
class State:

    step: int
    trajectory: Sequence[CellIndex]
    grid: Gridmap

    @property
    def position(st: State) -> CellIndex:
        return st.trajectory[-1]

    @property
    def cells_traveled(st: State) -> float:
        return np.sum(np.linalg.norm(np.diff(st.trajectory, axis=0), axis=-1))

    def tick(st: State) -> State:
        return State(st.step + 1, st.trajectory, st.grid)

    def move(st: State, target: CellIndex) -> State:
        return State(st.step, [*st.trajectory, target], st.grid)

    def integrate(st: State, scan: lidar.LIDARScan) -> tuple[State, int]:
        # Conservative rendering simply means that we consider free cells only
        # those that have been surrounded by known cells /in a single sensor
        # scan/. This prevents some fairly convoluted corner cases where the
        # gridmap will be incorrectly marked free where it is not because the
        # wall is aligned exactly radially with respect to the sensor, i.e.,
        # generates no incidences.
        if tunable('conservative_rendering', False):
            grid_scan = Gridmap.full(st.grid.shape, CellState.UNKNOWN)
        else:
            grid_scan = st.grid.copy()
        grid_scan.update_lines(CellState.FREE, scan.position, scan.points)
        grid_scan.update_points(CellState.OCCUPIED, scan.points[scan.hits])
        grid_scan.resolve_conflicting_states()
        scan_determined = grid_scan != CellState.UNKNOWN
        # Integrate rendered scan into gridmap by superpositioning the scan
        # results with the existing map, then resolving conflicts.
        grid = st.grid.copy()
        grid[scan_determined] |= grid_scan[scan_determined]
        grid.resolve_conflicting_states()
        num_changed = np.count_nonzero(st.grid != grid)
        return State(st.step, st.trajectory, grid), num_changed


class ExplorationFinished(Exception):
    pass


class Algorithm:
    "An algorithm is both the implementation of an algorithm and its state."

    short_name = '??'

    body: Body

    state: State

    current_cell: CellIndex

    frontier_cells: Sequence[CellIndex]

    current_distances: np.ndarray

    frontier_scores: Mapping[CellIndex, float]

    # Path plan
    unrefined_path: list[CellIndex]
    planned_path: list[CellIndex]

    def __init__(alg, body: Body, state: State) -> None:
        alg.body = body
        alg.state = state

    def plan(alg: Algorithm) -> None:
        raise NotImplementedError()

    def refine(alg: Algorithm) -> None:

        if not tunable('refine_paths', True):
            return

        path = alg.planned_path

        # Find last vertex on path that is line-of-sight.
        first_nonvis = bisect_left(path, True, key=lambda v: not alg.body.line_of_sight(v))
        last_visible = first_nonvis - 1

        assert 0 <= last_visible, 'some vertex must be visible (collision-free)'

        new_path = path[:1] + path[last_visible:]

        def straight_line_path(u: CellIndex, v: CellIndex, *, max_step=1.5) -> np.ndarray:
            d = np.linalg.norm(np.r_[v] - np.r_[u])
            steps = max(1, int(np.ceil(d / max_step))) + 1
            return np.linspace(u, v, steps)

        alg.unrefined_path = alg.planned_path
        alg.planned_path = [(x, y)
                            for (u, v) in zip(new_path[:-1], new_path[1:])
                            for (x, y) in straight_line_path(u, v)]

    def execute_iter(alg: Algorithm) -> Generator[State, None, None]:
        path_it = iter(alg.planned_path)
        assert next(path_it) == alg.state.position
        alg.state = alg.state.tick()
        for waypoint in path_it:
            alg.state = alg.state.move(alg.body.move(waypoint))
            alg.state, num_updated = alg.state.integrate(alg.body.scan())
            yield alg.state
            if 0 < num_updated:
                break
        else:
            #assert False, 'gridmap must have been updated during path execution'
            warn('gridmap was not updated during path execution')
        del alg.unrefined_path
        del alg.planned_path
        del alg.current_distances
        del alg.frontier_cells
        del alg.frontier_scores
        del alg.current_cell

    def execute(alg: Algorithm) -> None:
        for st in alg.execute_iter():
            pass

    def step_n(alg: Algorithm, n: int) -> None:
        for _ in tqdm(range(n), leave=False):
            alg.plan()
            alg.refine()
            assert 1 < len(alg.planned_path)
            alg.execute()


class NearestFrontier(Algorithm):

    short_name = 'nf'

    def find_frontier_scores(alg: NearestFrontier) -> dict[CellIndex, float]:
        return {u: -float(alg.current_distances[u])
                for u in alg.frontier_cells
                if 1 < alg.current_distances[u]}

    def find_nearest_free_cell(alg: NearestFrontier) -> CellIndex:
        st = alg.state
        for vs in st.grid.neighbors_by_level(u := st.position, state=CellState.FREE):
            if (vs := [v for v in vs if alg.body.line_of_sight(v)]):
                return min(vs, key=lambda v: np.linalg.norm(np.r_[u] - v))
        raise ValueError('no line-of-sight neighbor found')

    def plan(alg: NearestFrontier) -> None:

        st = alg.state

        alg.current_cell = alg.find_nearest_free_cell()

        alg.frontier_cells = st.grid.frontier_cells(current=alg.current_cell)

        alg.frontier_mask = indices_mask(st.grid.shape, alg.frontier_cells)

        sol = st.grid.sssp(source=alg.current_cell, target_mask=alg.frontier_mask)

        alg.current_distances = sol.distances

        # Here's a little trick: mark unreachable frontier cells unknown. These
        # frontiers will never be visited because they are not reachable, and
        # can make the BFS unnecessarily slow.
        alg.unreachable = 0 == sol.distances
        st.grid[alg.unreachable & alg.frontier_mask] = CellState.UNKNOWN

        alg.frontier_scores = alg.find_frontier_scores()

        if not alg.frontier_scores:
            raise ExplorationFinished

        alg.max_score_cell = max(alg.frontier_scores, key=alg.frontier_scores.__getitem__)

        alg.planned_path = [st.position] + sol.path_to(alg.max_score_cell, key=None)


class DistanceAdvantage(NearestFrontier):

    short_name = 'da'

    # Grid prediction
    predicted_grid: Gridmap

    # Which source cells are used when computing average path length
    source_mask: np.ndarray

    def plan(alg: DistanceAdvantage) -> None:

        st = alg.state

        alg.current_cell = alg.find_nearest_free_cell()

        alg.frontier_cells = st.grid.frontier_cells(current=alg.current_cell)

        alg.frontier_mask = indices_mask(st.grid.shape, alg.frontier_cells)

        alg.predicted_grid = alg.body.predict(st.grid)

        cell_coords = np.stack(meshgrid_shape(alg.predicted_grid.shape, indexing='ij'), axis=-1)
        cell_dists = np.linalg.norm(cell_coords - alg.current_cell, axis=-1, ord=np.inf)

        # Source mask is the cells from which we compute the distances from.
        alg.source_mask  = cell_dists < tunable('max_target_distance', 61)
        alg.source_mask &= alg.predicted_grid == CellState.FREE

        sol = alg.predicted_grid.sssp(source=alg.current_cell,
                                      target_mask=alg.source_mask | alg.frontier_mask)

        alg.current_distances = sol.distances

        # Here's a little trick: mark unreachable frontier cells unknown. These
        # frontiers will never be visited because they are not reachable, and
        # can make the BFS unnecessarily slow.
        alg.unreachable = 0 == sol.distances
        st.grid[alg.unreachable & alg.frontier_mask] = CellState.UNKNOWN

        alg.source_mask &= ~alg.unreachable

        alg.frontier_scores = alg.find_frontier_scores()

        if not alg.frontier_scores:
            raise ExplorationFinished

        alg.max_score_cell = max(alg.frontier_scores, key=alg.frontier_scores.__getitem__)

        alg.planned_path = [st.position] + sol.path_to(alg.max_score_cell, key=None)

    def find_nearest_frontiers(alg: DistanceAdvantage) -> list[CellIndex]:

        d = alg.current_distances

        frontiers = [u for u in alg.frontier_cells if 1 < d[u]]
        if not frontiers:
            return []

        nearest_frontiers = sorted(frontiers, key=d.__getitem__)

        max_frontiers         = tunable('max_frontiers', 500)
        max_frontier_distance = tunable('max_frontier_distance', 61)

        cutoff = bisect_right(nearest_frontiers, max_frontier_distance, key=d.__getitem__) + 1
        cutoff = min(cutoff, max_frontiers)
        del nearest_frontiers[cutoff:]

        return nearest_frontiers

    def find_frontier_scores(alg: DistanceAdvantage) -> dict[CellIndex, float]:
        if not (nearest_frontiers := alg.find_nearest_frontiers()):
            return {}
        mean_distances = alg.predicted_grid.find_mean_distances(
            sources=nearest_frontiers,
            target_mask=alg.source_mask
        )
        return {u: mean_distances[u] - alg.current_distances[u]
                for u in nearest_frontiers
                if 1 < alg.current_distances[u]}


class InformationGain(DistanceAdvantage):

    short_name = 'ig'

    score_formula = lambda a, g, d: (a.lambda_*np.log1p(g) - d) * a.body.world.scale
    lambda_ = 3.0

    def find_frontier_scores(alg: InformationGain) -> dict[CellIndex, float]:
        f = alg.score_formula
        d = alg.current_distances
        return {u: f(alg.info_gain(u), d[u])
                for u in alg.find_nearest_frontiers()
                if 1 < d[u]}

    def info_gain(alg: InformationGain, u: CellIndex):
        # NOTE This is always true the information gain, i.e., cheating. To
        # estimate naive information gain, we would have to do LIDAR simulation
        # against an occupancy grid.
        body = alg.body
        st = alg.state.move(u)
        scan = body.sensor.scan(body.world.walls, *st.position)
        _, num_changed = st.integrate(scan)
        return num_changed

class InformationGainSquareRoot(InformationGain):
    "As in Shrestha et al."

    short_name = 'igsqrt'

    lambda_ = 3.0
    score_formula = lambda a, g, d: (a.lambda_*np.sqrt(g) - d) * a.body.world.scale


def report(alg: Algorithm, *, savefig=False):

    from matplotlib import pyplot as plt

    from .utils import plot_polylines, reset_data_limits

    plt.rc('image', cmap='turbo')

    fig: plt.Figure  # type: ignore
    ax: plt.Axes  # type: ignore
    fig, ax = plt.subplots(figsize=(8, 4))
    xs: Any
    ys: Any

    st = alg.state
    body = alg.body
    world = body.world

    # For debugging sources, i.e., which vertices are considered in DA.
    if tunable('debug_sources', False) and hasattr(st, 'source_mask'):
        xs, ys = np.nonzero(st.source_mask & (0 < alg.current_distances))
        ax.scatter(xs, ys, c='lightgray', s=5, marker='s')
        xs, ys = np.nonzero(st.source_mask & (0 == alg.current_distances))
        ax.scatter(xs, ys, c='red', s=5, marker='s')

    with reset_data_limits():
        #for x in range(world.shape[0]+1): ax.axline((x-0.5, 0-0.5), (x-0.5, world.shape[1]-0.5), c='gray', lw=0.4)
        #for y in range(world.shape[1]+1): ax.axline((0-0.5, y-0.5), (world.shape[0]-0.5, y-0.5), c='gray', lw=0.4)
        #ax.scatter(*zip(*graph), c='C0', s=10)
        plot_polylines(world.walls, lw=0.2)
        ax.imshow(st.grid.T, alpha=0.5, cmap='tab10', vmin=0-0.5, vmax=10-0.5)
        #ax.imshow(np.where(0 < st.current_distances, st.current_distances, np.nan).T)

    # Past and future (planned) trajectory
    #assert False, 'make it a rainbow'
    plot_polylines([st.trajectory], ec='C5')
    ax.scatter(*st.position, marker='*', c='C5', s=30)
    if hasattr(alg, 'planned_path'):
        plot_polylines([alg.planned_path], ec='C3')
        xs, ys = zip(*alg.planned_path)
        ax.scatter(xs, ys, fc='C3', lw=0, s=10)
    if hasattr(alg, 'unrefined_path'):
        plot_polylines([alg.unrefined_path], ec='C4')
        xs, ys = zip(*alg.unrefined_path)
        ax.scatter(xs, ys, fc='C4', lw=0, s=10)

    # For debugging sensor and gridmap rendering
    if tunable('debug_scan', False):
        scan = body.scan()
        xs, ys = scan.points[ scan.hits].T
        ax.scatter(xs, ys, fc='C4', ec='w', lw=0.5, s=20)
        xs, ys = scan.points[~scan.hits].T
        ax.scatter(xs, ys, fc='C2', ec='w', lw=0.5, s=20)

    print(f'{st.step} steps, {len(st.trajectory)} states, '
          f'{st.cells_traveled*world.scale:.2f} meters, '
          f'{st.grid.coverage} cells covered, ',
          end='')

    # For debugging the frontier and their scoring.
    if hasattr(alg, 'current_distances'):

        frontiers = set(alg.frontier_cells)

        if (unreachable_frontiers := {u for u in alg.frontier_cells if 0 == alg.current_distances[u]}):
            with reset_data_limits():
                xs, ys = zip(*unreachable_frontiers)
                ax.scatter(xs, ys, fc='red', ec='w', lw=0.5, s=20)

        print(f'{len(frontiers)} frontiers, ', end='')

        if hasattr(alg, 'frontier_scores'):
            if (unscored_frontiers := frontiers - unreachable_frontiers - set(alg.frontier_scores)):
                with reset_data_limits():
                    xs, ys = zip(*unscored_frontiers)
                    ax.scatter(xs, ys, fc='gray', ec='w', lw=0.5, s=20)
            if alg.frontier_scores:
                xs, ys = zip(*alg.frontier_scores)
                m = ax.scatter(xs, ys, c=list(alg.frontier_scores.values()), ec='w', lw=0.5, s=20)
                fig.colorbar(m)
            print(f'{len(alg.frontier_scores)} scored')
    else:
        print('frontiers not computed')

    ax.set_aspect('equal')

    if savefig:
        plt.savefig(savefig, dpi=150)
        if tunable('show_savefigs', False):
            plt.show()
    else:
        plt.show()

    plt.close()


def run(alg: Algorithm, report_format: Callable[[str], str]) -> float:
    while True:
        try:
            while True:
                alg.plan()
                alg.refine()
                if (report_steps := tunable('report_steps', 100)) == 1:
                    report(alg)
                    alg.execute()
                else:
                    report(alg, savefig=report_format(f'_t{alg.state.step:04d}.png'))
                    alg.step_n(report_steps)
        except ExplorationFinished:
            print('Exploration finished!')
            return alg.state.cells_traveled*alg.body.world.scale
        except KeyboardInterrupt:
            print()
            print('Interrupted! Planning and plotting current state.')
            print('Interrupt again to quit, close plot to continue.')
            alg.plan()
            alg.refine()
            report(alg)
        except Exception:
            import traceback
            traceback.print_exc()
            print('Showing post-mortem plot.')
            report(alg)
            raise
        finally:
            report(alg, savefig=report_format(f'_t{alg.state.step:04d}_end.png'))


def load_world() -> World:
    input_filename = tunable('input_filename', resources.open_text('da', 'office_walls.csv'))
    rng = np.random.default_rng(seed=tunable('seed', 42))
    return World.loadtxt(input_filename,
                         pad=tunable('pad_sides_meter', 10),
                         cells_per_meter=tunable('cells_per_meter', 4),
                         rng=rng)


def main() -> None:
    parse_args(locals=locals(), globals=globals())

    from pprint import pprint

    from .utils import tunables

    print('Parameters:')
    pprint(tunables)

    algorithm_type = tunable('algorithm', DistanceAdvantage)
    print(f'{algorithm_type = }')

    world = load_world()

    if (start_location := tunable('start_location', None)) is None:
        start_indices   = tunable('start_indices', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        start_locations = world.start_locations
    else:
        start_indices   = [1]
        start_locations = [start_location]

    for i, ind in enumerate(start_indices):

        body, st = world.start_state(start_locations[ind - 1])

        print(f' [{i+1}/{len(start_indices)}] {ind}. {st.position = }')

        alg = algorithm_type(body, st)

        d_T = run(alg, report_format=f'report_{alg.short_name}_i{ind:02d}{{}}'.format)

        savetxt_row(f'completion_distances_{alg.short_name}.txt', ind - 1, d_T)


if __name__ == '__main__':
    main()
