from dataclasses import dataclass

import numba as nb
import numpy as np


@nb.njit
def min_x_int(x_int_max, px, py, dx, dy, segs):
    "Determine first intersection along ray [dx,dy] at point [px, py]"

    Rxx =  dx
    Rxy =  dy
    Ryx = -dy
    Ryy =  dx

    x_int_min = x_int_max

    for (lx0, ly0), (lx1, ly1) in segs:

        lx0 -= px
        lx1 -= px
        ly0 -= py
        ly1 -= py

        lx0_ = lx0*Rxx + ly0*Rxy
        ly0_ = lx0*Ryx + ly0*Ryy
        lx1_ = lx1*Rxx + ly1*Rxy
        ly1_ = lx1*Ryx + ly1*Ryy

        # Determine if line crosses upper/lower half plane. Weak inequality
        # guarantees non-zero vy later.
        if 0 <= ly0_*ly1_:
            continue

        # Ignore hits parallel to the ray.
        if abs(ly0_ - ly1_) < 1e-3:
            continue

        vx = lx1_ - lx0_
        vy = ly1_ - ly0_
        x_int = lx1_ - ly1_*vx/vy

        # Ignore intersections on the left half plane.
        if 1e-5 < x_int < x_int_min:
            x_int_min = x_int

    return x_int_min


@nb.njit
def intersect_lidar_rays(px, py, rays, lines, r_max):

    dists = np.empty((rays.shape[0],))

    for i in range(dists.shape[0]):
        dx, dy = rays[i]
        dists[i] = min_x_int(r_max, px, py, dx, dy, lines)

    return dists


@dataclass
class LIDARScan:
    "A LIDAR scan in world reference frame"
    position: np.ndarray  # shape: (2,)
    ranges: np.ndarray  # shape: (N,)
    points: np.ndarray  # shape: (N, 2)
    hits: np.ndarray  # shape: (N,)


@dataclass
class LIDAR():

    r_max: float
    num_points: int
    angle_min: float = -np.pi
    angle_max: float =  np.pi
    min_y: float = 0.0

    perp = np.array([[ 0, -1],
                     [+1,  0]])

    def __post_init__(self):
        a = np.linspace(self.angle_min, self.angle_max, self.num_points, endpoint=False)
        R_rays = np.array([( np.cos(a), -np.sin(a)),
                           ( np.sin(a),  np.cos(a))]).transpose((2, 0, 1))
        self.angles = a
        self.rotmats = np.ascontiguousarray(R_rays)
        self.rays = np.c_[np.cos(a), np.sin(a)]

    def simulate(self, lines, x=0.0, y=0.0, theta=0.0):
        return intersect_lidar_rays(x, y, self.rays, lines, self.r_max)

    def cartesian(self, ranges):
        ranges = np.atleast_1d(ranges)
        return ranges[:, None]*self.rays

    def is_hit(self, ranges):
        ranges = np.atleast_1d(ranges)
        return ranges < self.r_max

    def scan(self, wall_segments, x, y):
        ranges = self.simulate(wall_segments, x, y, theta=0.0)
        points = self.cartesian(ranges) + (x, y)
        hits   = self.is_hit(ranges)
        return LIDARScan(np.r_[x, y], ranges, points, hits)
