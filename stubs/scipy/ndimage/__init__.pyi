from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

def binary_dilation(
    input: ArrayLike,
    structure: ArrayLike | None = ...,
    iterations: int = ...,
    mask: ArrayLike | None = ...,
    output: NDArray | None = ...,
    border_value: bool = ...,
    origin: int | Sequence[int] = ...,
    brute_force: bool = ...
) -> NDArray:
    ...

def label(
    input: ArrayLike,
    structure: ArrayLike | None = ...,
    output: NDArray | np.dtype | None = ...
) -> tuple[NDArray[np.int_], int]:
    ...

__all__ = ['affine_transform', 'binary_closing', 'binary_dilation', 'binary_erosion', 'binary_fill_holes', 'binary_hit_or_miss', 'binary_opening', 'binary_propagation', 'black_tophat', 'center_of_mass', 'convolve', 'convolve1d', 'correlate', 'correlate1d', 'distance_transform_bf', 'distance_transform_cdt', 'distance_transform_edt', 'extrema', 'find_objects', 'fourier_ellipsoid', 'fourier_gaussian', 'fourier_shift', 'fourier_uniform', 'gaussian_filter', 'gaussian_filter1d', 'gaussian_gradient_magnitude', 'gaussian_laplace', 'generate_binary_structure', 'generic_filter', 'generic_filter1d', 'generic_gradient_magnitude', 'generic_laplace', 'geometric_transform', 'grey_closing', 'grey_dilation', 'grey_erosion', 'grey_opening', 'histogram', 'iterate_structure', 'label', 'labeled_comprehension', 'laplace', 'map_coordinates', 'maximum', 'maximum_filter', 'maximum_filter1d', 'maximum_position', 'mean', 'median', 'median_filter', 'minimum', 'minimum_filter', 'minimum_filter1d', 'minimum_position', 'morphological_gradient', 'morphological_laplace', 'percentile_filter', 'prewitt', 'rank_filter', 'rotate', 'shift', 'sobel', 'spline_filter', 'spline_filter1d', 'standard_deviation', 'sum', 'sum_labels', 'uniform_filter', 'uniform_filter1d', 'value_indices', 'variance', 'watershed_ift', 'white_tophat', 'zoom']

# Names in __all__ with no definition:
#   affine_transform
#   binary_closing
#   binary_erosion
#   binary_fill_holes
#   binary_hit_or_miss
#   binary_opening
#   binary_propagation
#   black_tophat
#   center_of_mass
#   convolve
#   convolve1d
#   correlate
#   correlate1d
#   distance_transform_bf
#   distance_transform_cdt
#   distance_transform_edt
#   extrema
#   find_objects
#   fourier_ellipsoid
#   fourier_gaussian
#   fourier_shift
#   fourier_uniform
#   gaussian_filter
#   gaussian_filter1d
#   gaussian_gradient_magnitude
#   gaussian_laplace
#   generate_binary_structure
#   generic_filter
#   generic_filter1d
#   generic_gradient_magnitude
#   generic_laplace
#   geometric_transform
#   grey_closing
#   grey_dilation
#   grey_erosion
#   grey_opening
#   histogram
#   iterate_structure
#   labeled_comprehension
#   laplace
#   map_coordinates
#   maximum
#   maximum_filter
#   maximum_filter1d
#   maximum_position
#   mean
#   median
#   median_filter
#   minimum
#   minimum_filter
#   minimum_filter1d
#   minimum_position
#   morphological_gradient
#   morphological_laplace
#   percentile_filter
#   prewitt
#   rank_filter
#   rotate
#   shift
#   sobel
#   spline_filter
#   spline_filter1d
#   standard_deviation
#   sum
#   sum_labels
#   uniform_filter
#   uniform_filter1d
#   value_indices
#   variance
#   watershed_ift
#   white_tophat
#   zoom
