#!/usr/bin/env python3

import argparse
from itertools import combinations
from os import PathLike, fspath
from typing import Generator, Iterable

import numpy as np


def load_single(filenames: Iterable[str | PathLike[str]]) -> Generator[tuple[str, np.ndarray], None, None]:
    for fn in filenames:
        arr = np.loadtxt(fn, ndmin=2)
        yield fspath(fn), arr


def load_pairs(filenames: Iterable[str | PathLike[str]]) -> Generator[tuple[str, np.ndarray], None, None]:
    for (name1, arr1), (name2, arr2) in combinations(load_single(filenames), 2):
        arr1, arr2 = nanpad((arr1, arr2))
        yield shell_compress((name1, name2)), arr2 - arr1


def nanpad(arrays: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    max_shape = tuple(max(sizes) for sizes in zip(*(arr.shape for arr in arrays)))
    return tuple(
        np.pad(arr, [(0, max_i - dim_i) for dim_i, max_i in zip(arr.shape, max_shape)], constant_values=np.nan)
        for arr in arrays
    )


def shell_compress(strs: Iterable[str]) -> str:

    str0, *rest = strs
    str_list = [str0, *rest]

    if not rest:
        return str0

    rts0, *_ = rts_list = [s[::-1] for s in str_list]

    n = min(map(len, str_list))

    prefix_len = next((i for i in range(n)              if any(str0[i] != s[i] for s in str_list)), n)
    suffix_len = next((i for i in range(n - prefix_len) if any(rts0[i] != r[i] for r in rts_list)), 0)

    prefix = str0[:prefix_len]
    suffix = rts0[:suffix_len][::-1]

    diffs = [s[prefix_len:len(s) - suffix_len or None] for s in str_list]

    return f"{prefix}{{{','.join(diffs)}}}{suffix}"


parser = argparse.ArgumentParser()
parser.add_argument('--single', action='store_const', dest='loader', const=load_single)
parser.add_argument('--pairs',  action='store_const', dest='loader', const=load_pairs)
parser.add_argument('filenames', nargs='+')
parser.set_defaults(loader=load_single)


def main(argv=None) -> None:

    args = parser.parse_args(args=argv)

    for name, arr in args.loader(args.filenames):
        nrows, ncols = arr.shape
        stats = dict(
                     len=np.count_nonzero(~np.isnan(arr), axis=0),
                     mean=np.nanmean(arr, axis=0),
                     pstd=np.nanstd(arr, axis=0, ddof=1),
                     std=np.nanstd(arr, axis=0),
                     median=np.nanmedian(arr, axis=0),
                     min=np.nanmin(arr, axis=0),
                     max=np.nanmax(arr, axis=0),
                    )
        print(name, ', '.join(f'{key}[{i}]: {value[i]:.5g}'
                              for key, value in stats.items()
                              for i in range(ncols)))

if __name__ == '__main__':
    main()
