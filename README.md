# Minimal Reproduction of Distance Advantage Exploration



This repository is a distilled version of the algorithms developed for the
paper ``Information Gain Is Not All You Need''.

Note that though the code is not identical to the code that was used to produce
the paper, the results stay mostly the same. This version is optimized for
readability and reproducibility rather than exact duplication. Key differences
include:

 1. Path planning is 4-connected whereas the paper was 8-connected. To
    compensate for this, a line-of-sight path simplifier is applied. Briefly,
    the robot will move in a straight line towards the last vertex on its
    planned path that it can see from the start. This path refinement can be
    disabled with the setting `--refine_paths=False`.

 2. Starting locations are sampled uniformly without regard for minimum
    distances, the implementation in the paper uses farthest-point sampling.

 3. Information gain is computed only at the given frontier cell, not along its
    path; and, it is always true information gain.

## Installation

```bash
git clone https://git.lericson.se/da.git
cd da
python3.12 -m venv env
./env/bin/pip install -e .
```

## Running

The main file is appropriately called `main.py`, it is invoked with the
`exploration` script:

```bash
./env/bin/exploration
```

This will output images like `report_i01_t0000.png`. Each of these is a plot
showing the state of exploration and the planner at that start location and
time step.

The default algorithm is distance advantage, a few baselines are also
implemented:

```bash
./env/bin/exploration --algorithm=NearestFrontier
./env/bin/exploration --algorithm=InformationGain
```

All methods can also be run without predictions by specifying
`predictions=False`.

You can also use GNU parallel to run many instances in parallel (or sequence):

```bash
parallel -j12 NUMBA_NUM_THREADS=2 ./env/bin/exploration \
  ::: --algorithm={DistanceAdvantage,NearestFrontier,InformationGain} \
  ::: --start-locations=[{1..10}]
```

`NUMBA_NUM_THREADS` controls how many parallel BFS searches are performed for
distance advantage, otherwise it does nothing. `-j12` specifies how many jobs
to run in parallel. It should typically be the number of actual cores (not
threads).

## Results

+--------------------------------------+---------------+--------+
| Algorithm                            | d_T (meters)  | Change |
+--------------------------------------+---------------+--------+
| `DistanceAdvantage` with predictions | 1435.9 ± 30.2 |     0% |
| `DistanceAdvantage` w/o  predictions | 1602.9 ± 26.0 | +11.6% |
| `NearestFrontier`                    | 1905.4 ± 54.4 | +32.7% |
| `InformationGain`                    | 2014.2 ± 43.4 | +40.3% |
+--------------------------------------+---------------+--------+

In other words, the results agree with those in the paper, except for
information gain which is better here because this implementation
underestimates the information gain, as it only counts information gain for a
single vertex, not along a path as the paper does. This changes its behavior to
more nearest frontier-like for the same value of λ.

## Citation

```bibtex
@article{ericson2025da,
  title={Information Gain Is Not All You Need},
  author={Ericson, Ludvig and Pedro, Jos{\'e} and Jensfelt, Patric},
  eprint={2504.01980},
  eprinttype={arxiv}
  year={2025}
}
```
