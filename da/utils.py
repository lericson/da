from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterable, TypeVar
from numpy.typing import ArrayLike, DTypeLike

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.path import Path


def savetxt_row(filename: str, index: int, row: ArrayLike, *, dtype: DTypeLike = float, na: ArrayLike = np.nan) -> None:
    row = np.atleast_1d(row).astype(dtype)
    ncols, = row.shape
    try:
        rows = np.loadtxt(filename, ndmin=2, dtype=dtype)
    except FileNotFoundError:
        rows = np.empty((0, ncols), dtype=dtype)
    assert rows.shape[1:] == row.shape
    if len(rows) <= index:
        rows_pad = np.full((index + 1, ncols), na, dtype=dtype)
        rows_pad[:len(rows)] = rows
        rows = rows_pad
    rows[index] = row
    np.savetxt(filename, rows)


def permute_nonzero(mask: np.ndarray, *, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    all_inds_flat, = np.nonzero(mask.flat)
    sampled_inds_flat = rng.permutation(all_inds_flat)
    return np.unravel_index(sampled_inds_flat, mask.shape)


def meshgrid_shape(shape: tuple[int, ...], indexing='xy') -> list[np.ndarray]:
    return np.meshgrid(*[np.arange(di) for di in shape], indexing=indexing)


def edge_view(poly: np.ndarray, *, flat: bool = False) -> np.ndarray:
    from numpy.lib.stride_tricks import as_strided
    "(..., num_verts, num_coords) -> (..., num_verts - 1, 2, num_coords)"
    poly_arr = np.atleast_2d(poly)
    *shape, num_verts, num_coords = poly_arr.shape
    *strides, stride_vert, stride_coord = poly_arr.strides
    segs_shape   = (*shape, num_verts - 1, 2, num_coords)
    segs_strides = (*strides, stride_vert, stride_vert, stride_coord)
    segs = as_strided(poly_arr, shape=segs_shape, strides=segs_strides)
    if flat:
        segs = segs.reshape((-1, 2, num_coords))
    return segs


def plot_polylines(polys: Iterable, *,
                   ax: plt.Axes | None = None,
                   edgecolor = None,
                   facecolor='none',
                   **kw) -> None:
    ax = ax if ax is not None else plt.gca()
    paths = [Path(poly) for poly in polys]
    assert 'ec' not in kw or edgecolor is None
    edgecolor = kw.pop('ec', None) if edgecolor is None else edgecolor
    edgecolor = ax._get_patches_for_fill.get_next_color() if edgecolor is None else edgecolor  # type: ignore
    assert 'fc' not in kw or facecolor is None
    facecolor = kw.pop('fc', None) if facecolor is None else facecolor
    ax.add_collection(PathCollection(paths, edgecolor=edgecolor, facecolor=facecolor, **kw))


@contextmanager
def reset_data_limits(ax=None):
    "Reset data limit on exit to temporarily stop view autoscaling"
    ax = plt.gca() if ax is None else ax
    previous = ax.dataLim.frozen()
    yield
    ax.dataLim = previous


# Rudimentary configuration system

tunables: dict[str, Any] = {'parsed_args': False}

def parse_args(args=sys.argv[1:], *, locals=None, globals=None):
    tunables['parsed_args'] = True
    for arg in args:
        key, _, value = arg.partition('=')
        if not value:
            raise ValueError(f'tunable specified as {arg!r} does not contain an equal sign')
        if key.startswith('--'):
            key = key[2:].replace('-', '_')
        tunables[key] = eval(value, globals, locals)

T = TypeVar('T')

def tunable(name: str, default: T) -> T:
    if not tunables['parsed_args']:
        raise ValueError('Do not call tunable() before parse_args(), e.g., at the module level')
    tunables.setdefault(name, default)
    return tunables[name]
