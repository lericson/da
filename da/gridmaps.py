from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import floor
from typing import Generator, Iterable, TypeAlias

import numba as nb
import numpy as np
from scipy import ndimage

from .utils import tunable


CellIndex: TypeAlias = tuple[int, int]

CS_DTYPE = np.uint8

class CellState:
    NOT_SET  = CS_DTYPE(0b000)
    UNKNOWN  = CS_DTYPE(0b001)
    FREE     = CS_DTYPE(0b010)
    OCCUPIED = CS_DTYPE(0b100)
    MAXIMUM  = OCCUPIED


# Numba things
CS_NOT_SET  = CellState.NOT_SET
CS_UNKNOWN  = CellState.UNKNOWN
CS_FREE     = CellState.FREE
CS_OCCUPIED = CellState.OCCUPIED
CS_MAXIMUM  = CS_OCCUPIED


def neighbors_by_level(rank=5):
    levels = [[] for _ in range(2*rank + 1)]
    for dx, dy in product(range(-rank, rank + 1), repeat=2):
        level = abs(dx) + abs(dy)
        levels[level].append((dx, dy))
    return levels


NEIGHBORS_BY_LEVEL = neighbors_by_level()


def decimate(mask):
    even = slice(0, None, 2)
    odd  = slice(1, None, 2)
    mask[0:-1:2,  odd] &= ~mask[1:  :2,  odd]
    mask[2:-1:2,  odd] &= ~mask[1:-2:2,  odd]
    mask[1:  :2, even] &= ~mask[0:-1:2, even]
    mask[1:-2:2, even] &= ~mask[2:-1:2, even]
    mask[ odd, 0:-1:2] &= ~mask[ odd, 1:  :2]
    mask[ odd, 2:-1:2] &= ~mask[ odd, 1:-2:2]
    mask[even, 1:  :2] &= ~mask[even, 0:-1:2]
    mask[even, 1:-2:2] &= ~mask[even, 2:-1:2]
    return mask



class Gridmap(np.ndarray):
    """A gridmap backed by a numpy array.

    Uses matplotlib-style coordinates, i.e., the center of a cell is at an
    integer, and the boundary of a cell is ±0.5 from that.

    Controversially we consider the first axis to be the X axis, i.e., to find
    the xth and yth cell, you do array[x, y], not array[y, x]. This is not how
    matplotlib expects things, so please plt.imshow(array.T).
    """

    default_dtype = CS_DTYPE

    def __new__(cls, *a, **kw) -> Gridmap:
        kw.setdefault('dtype', cls.default_dtype)
        return super().__new__(cls, *a, **kw)

    @classmethod
    def full(cls, shape: tuple[int, ...] | np.ndarray, state: CS_DTYPE) -> Gridmap:
        return np.full(shape, state).view(Gridmap)

    @property
    def coverage(grid: Gridmap) -> int:
        return np.count_nonzero(grid != CellState.UNKNOWN)

    def update_lines(grid, state, origins, ends):
        for origin, end in zip(*(np.broadcast_arrays(origins, ends))):
            _draw_ray_2d(grid, state, *origin, *end)

    def update_points(grid, state, points):
        points = np.floor(np.asarray(points) + 0.5).astype(int)
        inside = np.all((0 <= points) & (points < grid.shape), axis=-1)
        grid[*np.moveaxis(points[inside], -1, 0)] |= state

    def resolve_conflicting_states(grid):
        # OCCUPIED takes first precedence.
        occupiedish = (grid & CellState.OCCUPIED) != 0
        grid[occupiedish] = CellState.OCCUPIED

        freeish = (grid & CellState.FREE) != 0
        if tunable('classical_rendering', False):
            # FREE is anything intersected by a ray
            free = freeish
        else:
            # FREE is only FREE if no neighbors are pure unknown
            unknown = grid == CellState.UNKNOWN
            free = freeish & ~ndimage.binary_dilation(unknown)
        grid[free] = CellState.FREE

        # UNKNOWN takes last precedence.
        unknownish = (grid & CellState.UNKNOWN) != 0
        grid[unknownish] = CellState.UNKNOWN

    def sssp(grid: Gridmap, *,
             source: CellIndex,
             targets: Iterable[CellIndex] | None = None,
             target_mask: np.ndarray | None = None,
             mark_source_free: bool = True,
            ) -> SSSPSolution:
        if mark_source_free:
            grid[source] = CellState.FREE
        return grid.mssp(sources=[source],
                         targets=targets,
                         target_mask=target_mask)[source]

    def mssp(grid: Gridmap, *,
             sources: Iterable[CellIndex],
             targets: Iterable[CellIndex] | None = None,
             target_mask: np.ndarray | None = None,
            ) -> dict[CellIndex, SSSPSolution]:
        #grid[source] = CellState.FREE
        sources = [(x, y) for (x, y) in sources]
        distances_sources = np.zeros((len(sources), *grid.shape), dtype=np.int64)
        parents_sources   = np.zeros((len(sources), *grid.shape, 4), dtype=np.bool_)
        triplets = sources, distances_sources, parents_sources
        target_mask = _homogenize_target_mask(grid.shape, targets=targets, target_mask=target_mask)
        _mssp_numba(grid, target_mask, *triplets)
        return {s: SSSPSolution(s, d, p) for s, d, p in zip(*triplets)}

    def find_mean_distances(grid: Gridmap,
                            sources: Iterable[CellIndex],
                            targets: Iterable[CellIndex] | None = None,
                            target_mask: np.ndarray | None = None,
                            ) -> dict[CellIndex, float]:

        if not sources:
            return {}

        target_mask = _homogenize_target_mask(grid.shape, targets=targets, target_mask=target_mask)

        solutions = grid.mssp(sources=sources, target_mask=target_mask)

        return {source: np.mean(sol.distances, where=target_mask & (0 < sol.distances))
                for source, sol in solutions.items()}

    def frontier_mask(grid: Gridmap, *, current: CellIndex) -> np.ndarray:
        free       = grid == CellState.FREE
        unknownish = (grid & CellState.UNKNOWN) != 0
        frontiers  = free & ndimage.binary_dilation(unknownish)

        # Current cell cannot be a frontier. It can happen with conservative
        # gridmap rendering.
        frontiers[current] = False

        if (dilate_frontiers := tunable('dilate_frontiers', False)) is not False:

            # Propagate frontiers by dilating the frontier into free space.
            mask = free.copy()

            # Prevent propagating the frontier behind the current cell. This is
            # done with a mask (rather than frontiers[neighbors] = False) so
            # that true frontiers cannot be removed.
            ux, uy = current
            neighbors = [(ux + dx, uy + dy)
                         for level_neighbors in NEIGHBORS_BY_LEVEL[:dilate_frontiers + 1]
                         for (dx, dy) in level_neighbors]
            mask[neighbors] = False

            frontiers = ndimage.binary_dilation(frontiers, mask=mask, iterations=dilate_frontiers)

        # Frontier decimation
        if tunable('decimate_frontiers', True):
            frontiers = decimate(frontiers)

        return frontiers

    def frontier_cells(grid: Gridmap, *, current: CellIndex) -> list[CellIndex]:
        xs, ys = np.nonzero(grid.frontier_mask(current=current))
        return list(zip(xs, ys))

    def neighbors_by_level(grid: Gridmap,
                           u: CellIndex, *,
                           state: CS_DTYPE
                          ) -> Generator[list[CellIndex], None, None]:
        xlen, ylen = grid.shape
        ux, uy = np.floor(np.r_[u] + 0.5).astype(int)
        for neighbors in NEIGHBORS_BY_LEVEL:
            yield [(vx, vy) for (dx, dy) in neighbors
                   if 0 <= (vx := ux + dx) < xlen
                   if 0 <= (vy := uy + dy) < ylen
                   if grid[vx, vy] == state]


def _homogenize_target_mask(shape: tuple[int, ...], *,
                            targets: Iterable[CellIndex] | None = None,
                            target_mask: np.ndarray | None = None) -> np.ndarray:

    assert targets is None or target_mask is None

    if targets is not None:
        target_mask = np.zeros(shape, dtype=np.bool_)
        targets_arr = np.asarray(targets)
        if 0 < len(targets_arr):
            target_mask[*targets_arr.T] = True
        return target_mask
    elif target_mask is None:
        return np.ones(shape, dtype=np.bool_)

    return target_mask


@nb.njit(parallel=True)
def _mssp_numba(grid_array, target_mask, sources, distances_sources, parents_sources) -> None:
    for i in nb.prange(len(sources)):
        _sssp_numba(grid_array, target_mask, sources[i], distances_sources[i], parents_sources[i])


@nb.njit()
def _sssp_numba(grid_array, target_mask, source, distances, parents) -> None:

    xlen, ylen = grid_array.shape

    distances[source] = 1
    num_targets = np.count_nonzero(target_mask)

    QUEUE_MAX = 2*max(grid_array.shape)

    # Ring queue.
    queue_buf = np.empty((QUEUE_MAX, 3), dtype=np.int64)
    queue_ptr = 0
    queue_end = 0

    queue_buf[queue_end] = (*source, distances[source])
    queue_end += 1

    while queue_end != queue_ptr:

        # Pop queue
        x, y, dist_u = queue_buf[queue_ptr]
        queue_ptr += 1
        queue_ptr %= QUEUE_MAX

        if target_mask[x, y]:
            num_targets -= 1
            if num_targets == 0:
                break

        for i, v in successors(grid_array, (x, y)):

            dist_uv = dist_u + 1

            if distances[v] == 0:
                distances[v] = dist_uv
                queue_buf[queue_end] = (*v, dist_uv)
                queue_end += 1
                queue_end %= QUEUE_MAX
                assert queue_end != queue_ptr

            if distances[v] == dist_uv:
                parents[*v, i] = True


ADJ = ((-1, 0), (+1, 0), (0, -1), (0, +1))


@nb.njit(boundscheck=False)
def successors(grid_array: np.ndarray, u: CellIndex) -> Iterable[tuple[int, CellIndex]]:
    xlen, ylen = grid_array.shape
    x, y = u
    adj = []
    if ((0 <= x < xlen) and (0 <= y < ylen)):
        for i in range(len(ADJ)):
            vx, vy = v = nth_successor(u, i)
            if ((0 <= vx < xlen) and (0 <= vy < ylen)) and grid_array[v] == CS_FREE:
                adj.append((i, v))
    return adj


@nb.njit
def nth_successor(u: CellIndex, i: int) -> CellIndex:
    x, y = u
    dx, dy = ADJ[i]
    return x + dx, y + dy


@nb.njit
def nth_parent(u: CellIndex, i: int) -> CellIndex:
    x, y = u
    dx, dy = ADJ[i]
    return x - dx, y - dy


def indices_mask(shape: tuple[int, ...], indices: Iterable[tuple[int, ...]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.bool_)
    indices_arr = np.asarray(indices)
    if 0 < len(indices_arr):
        mask[*indices_arr.T] = True
    return mask


@dataclass(frozen=True)
class SSSPSolution:
    "Single-source shortest path solution"
    source: CellIndex
    distances: np.ndarray
    parents_mask: np.ndarray

    def parents(self, cell: CellIndex) -> list[CellIndex]:
        return [nth_parent(cell, i) for i in range(4) if self.parents_mask[*cell, i]]

    def path_to(self, target: CellIndex, *, key) -> list[CellIndex]:
        path_rev = [target]
        while (u := path_rev[-1]) != self.source:
            path_rev.append(max(self.parents(u), key=key))
        return path_rev[::-1]


@nb.njit
def _draw_ray_2d(grid: np.ndarray, state: int, x0: float, y0: float, x1: float, y1: float) -> None:
    "Subpixel-accurate line drawing algorithm"

    eps = 1e-9
    xlen, ylen = grid.shape
    dx = x1 - x0
    dy = y1 - y0

    step_x = np.sign(dx)*(1 + eps)
    step_y = np.sign(dy)*(1 + eps)

    xc, yc = x0, y0

    while True:

        xint, yint = int(floor(xc + 0.5)), int(floor(yc + 0.5))
        if (0 <= xint < xlen) and (0 <= yint < ylen):
            grid[xint, yint] |= state

        # Next horizontal and vertical grid line coordinates.
        xl = np.floor(xc + 0.5) + 0.5*step_x
        yl = np.floor(yc + 0.5) + 0.5*step_y

        # Find interesction of ray p0-p1 and the grid lines (x, yl) and (xl,
        # y), i.e., a given xl for vertical grid lines, or a given yl for
        # horizontal grid lines.
        #
        # Points on the ray are
        #   (x, y) = p0 + t(dx, dy) with 0 <= t <= 1.
        #
        # Set equal to the vertical grid line (xl, y):
        #   (x, y) = (xl, y) = p0 + t(dx, dy)
        #
        # This is a pair of simultaneous equations
        #   x = xl = x0 + tdx
        #   y      = y0 + tdy
        #
        # Solve for t:
        #   t = (xl - x0) / dx
        #   t = (y - y0) / dy
        #
        # Set the two equations equal:
        #   (xl - x0) / dx = (y - y0) / dy
        #
        # Solve for y by multiplying dy and adding y0 on both sides:
        #   x = xl
        #   y = (xl - x0) / dx * dy + y0
        tvert = (xl - x0) / dx if dx != 0.0 else np.inf
        xvert = xl
        yvert = tvert * dy + y0

        # Similarly, the solution for the horizontal line is:
        #   x = (yl - y0) / dy * dx + x0
        #   y = yl
        thorz = (yl - y0) / dy if dy != 0.0 else np.inf
        xhorz = thorz * dx + x0
        yhorz = yl

        # (xhorz, yhorz) and (xvert, yvert) are the next grid line intersection
        # points for the ray, and thorz and tvert the t parameter for the
        # intersection point _on the ray line_. The closest intersection point
        # is the lower of the two.
        if 0.0 <= thorz <= 1.0 and thorz < tvert:
            xc, yc = xhorz, yhorz
        elif 0.0 <= tvert <= 1.0:
            xc, yc = xvert, yvert
        else:
            # Ray ended before next grid line intersection, (xc, yc) is the
            # terminal grid cell.
            break
