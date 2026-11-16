# Code comparison

22497992 cells, 1024^2 pixels, 32 threads.

| code | mesh or tree build [s] | ray tracing [s] | total [s] | peak memory [GiB] |
|---|---:|---:|---:|---:|
| vortrace | 8.3 | 16.7 | 27.9 | 3.2 |
| colt | 337.7 | 2.3 | 343.5 | 57.1 |
| skirt | 110.8 | 3.4 | 182.8 | 16.0 |

Relative difference from vortrace, per pixel:

| code | 1st percentile | 99th percentile | pixels beyond 1e-6 | largest |
|---|---:|---:|---:|---:|
| colt | -9.0e-13 | 8.9e-13 | 1 | 2.0e-04 |
| skirt | -5.1e-08 | 4.9e-08 | 2 | 1.8e-03 |
