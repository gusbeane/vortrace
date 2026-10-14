# Code comparison timings

## small

Arepo test-suite galaxy interaction (documentation): 7217 cells, 256^2 pixels, 75.0 kpc window, z in [0.0, 649.201] kpc.  Best of 3 run(s); machine: macOS-27.0-arm64-arm-64bit-Mach-O, 8 cores.

| code | threads | mesh / tree [s] | ray tracing [s] | executable [s] | total incl. file I/O [s] | peak RSS [GiB] | max rel. diff. vs vortrace |
|---|---:|---:|---:|---:|---:|---:|---:|
| vortrace | 1 | 0.00 | 1.79 | -- | 1.79 | 0.1 | -- |
| colt | 1 | 0.12 | 0.11 | 0.25 | 0.25 | 0.2 | 1.7e-11 |
| colt | 8 | 0.06 | 0.02 | 0.10 | 0.10 | 0.2 | 1.7e-11 |
| skirt | 1 | 1.03 | 0.25 | 1.32 | 1.32 | 0.0 | 6.0e-08 |
| skirt | 8 | 0.20 | 0.08 | 0.32 | 0.32 | 0.0 | 6.0e-08 |

## large_w1

isolated dwarf galaxy (paper yt comparison): 27323638 cells (4794733 handed to the mesh codes), 256^2 pixels, 1.0 kpc window, z in [88.0, 92.0] kpc.  Best of 1 run(s); machine: macOS-27.0-arm64-arm-64bit-Mach-O, 8 cores.

| code | threads | mesh / tree [s] | ray tracing [s] | executable [s] | total incl. file I/O [s] | peak RSS [GiB] | max rel. diff. vs vortrace |
|---|---:|---:|---:|---:|---:|---:|---:|
| vortrace | 1 | 8.91 | 24.75 | -- | 33.66 | 4.5 | -- |
| colt | 1 | 168 | 2.34 | 171 | 172 | 6.1 | 9.9e-11 |
| colt | 8 | 79.08 | 0.46 | 80.29 | 80.83 | 7.7 | 9.9e-11 |
| skirt | 1 | 319 | 3.88 | 329 | 329 | 3.6 | 1.2e-06 |
| skirt | 8 | 68.09 | 0.98 | 74.67 | 74.95 | 3.7 | 1.2e-06 |
