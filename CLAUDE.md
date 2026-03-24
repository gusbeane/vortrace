# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vortrace performs fast one-dimensional integrals through unstructured Voronoi meshes for astrophysics. It uses a recursive ray-tracing algorithm that minimizes kDTree queries to handle dynamic cell size ranges in cosmological simulations.

## Environment

Always activate the project venv before running any Python/pip commands:

```bash
source /home/abeane/Projects/vortrace/venv/bin/activate
```

## Build & Install

```bash
pip install ./                  # Standard install (builds C++ via scikit-build-core + CMake)
pip install -e .                # Editable install
pip install -r requirements.txt # Install dependencies only
```

For C++ tests, build with Catch2 support:
```bash
mkdir build && cd build
cmake .. -DBUILD_CPP_TESTS=ON
cmake --build .
```

## Testing

```bash
pytest                              # All Python tests
pytest tests/test_vortrace.py       # Single test file
pytest tests/test_vortrace.py::test_basic -v  # Single test
pytest -m "not benchmark"           # Skip benchmarks
ctest --test-dir build --output-on-failure  # C++ tests (after cmake build)
```

## Linting

```bash
pylint --rcfile=.pylintrc vortrace
```

Uses Google Python style guide with many checks disabled (see `.pylintrc`).

## Architecture

**Two-layer design: C++ core + Python frontend, connected via pybind11.**

### C++ Core (`include/`, `src/`)

- **Ray** — single ray integration through the mesh via recursive split-point algorithm
- **PointCloud** — manages particle data and kDTree (nanoflann) for nearest-neighbor queries; handles adaptive padding and subbox filtering
- **Projection** — batches rays into grid-based projections
- **BruteProjection** — alternative brute-force projection method
- **Slice** — 2D slicing through the mesh
- **bindings.cpp** — pybind11 bindings exposing C++ classes as the `Cvortrace` extension module

The C++ code builds as: `vortrace_core` (static library) → `Cvortrace` (shared Python extension linked against it).

### Python Frontend (`vortrace/`)

- **`vortrace.py`** — `ProjectionCloud` class: the main user-facing API for grid projections, direct ray projections, and single-ray segment queries. Supports rotation (yaw/pitch/roll), multiple fields, and reduction modes (Sum/Max/Min).
- **`grid.py`** — grid generation with numba `@njit` acceleration and Tait-Bryan rotation support
- **`io.py`** — save/load projections in NPZ (default) or HDF5 format; lazy h5py import
- **`plot.py`** — `plot_grid()` and `plot_ray()` visualization; lazy matplotlib import

### Key types (C++ side)

- `Float` = `double`, `Point` = `std::array<Float, 3>`
- `ReductionMode` enum: `Sum`, `Max`, `Min`
- Global `verbose` flag in `vortrace` namespace (runtime, not compile-time)

## Design Principles

- Core logic belongs in C++ — the C++ library is exposed directly, so Python is a thin wrapper
- Library code should not print to stdout by default; use the runtime verbose flag
- Optional dependencies (h5py, matplotlib) are imported lazily with clear error messages

## Code Intelligence

Prefer LSP over Grep/Glob/Read for code navigation:
- `goToDefinition` / `goToImplementation` to jump to source
- `findReferences` to see all usages across the codebase
- `workspaceSymbol` to find where something is defined
- `documentSymbol` to list all symbols in a file
- `hover` for type info without reading the file
- `incomingCalls` / `outgoingCalls` for call hierarchy

Before renaming or changing a function signature, use
`findReferences` to find all call sites first.

Use Grep/Glob only for text/pattern searches (comments,
strings, config values) where LSP doesn't help.

After writing or editing code, check LSP diagnostics before
moving on. Fix any type errors or missing imports immediately.
