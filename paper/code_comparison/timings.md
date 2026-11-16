# Code comparison timings

## small

Arepo test-suite galaxy interaction (documentation): 7217 cells, 256^2 pixels, 75.0 kpc window, z in [0.0, 649.201] kpc.  Best of 3 run(s); machine: Linux-6.8.0-136-generic-x86_64-with-glibc2.35, 32 cores.

| code | threads | mesh / tree [s] | ray tracing [s] | executable [s] | total incl. file I/O [s] | peak RSS [GiB] | max rel. diff. vs vortrace |
|---|---:|---:|---:|---:|---:|---:|---:|
| vortrace | 32 | 0.00 | 0.16 | -- | 0.16 | 0.1 | -- |
| colt | 32 | 0.09 | 0.03 | 0.16 | 0.16 | 0.0 | 1.7e-11 |
| skirt | 32 | 0.12 | 0.02 | 0.21 | 0.21 | 0.0 | 6.0e-08 |

## large

isolated dwarf galaxy (paper yt comparison): 22497992 cells inside [[87.0, 87.0, 88.0], [93.0, 93.0, 92.0]] kpc (of 27323638 in the snapshot), 1024^2 pixels, 6.0 kpc window, z in [88.0, 92.0] kpc.  Best of 1 run(s); machine: Linux-6.8.0-138-generic-x86_64-with-glibc2.35, 32 cores.

| code | threads | mesh / tree [s] | ray tracing [s] | executable [s] | total incl. file I/O [s] | peak RSS [GiB] | max rel. diff. vs vortrace |
|---|---:|---:|---:|---:|---:|---:|---:|
| vortrace | 32 | 8.41 | 16.74 | 27.33 | 28.43 | 3.2 | -- |
| skirt | 32 | 110 | 3.44 | 181 | 182 | 16.0 | 1.8e-03 |
| colt | 32 | 338 | 2.29 | 344 | 345 | 57.1 | 2.0e-04 |
