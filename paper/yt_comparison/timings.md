# yt comparison

| code | threads | index or tree build [s] | projection [s] | total [s] | peak memory [GiB] |
|---|---:|---:|---:|---:|---:|
| vortrace | 1 | 13.3 | 281.3 | 294.6 | 4.1 |
| yt | 1 | 18.6 | 21.3 | 40.0 | 1.0 |
| vortrace | 32 | 13.2 | 19.2 | 32.4 | 4.1 |
| yt | 32 | 18.7 | 14.4 | 33.0 | 1.2 |

yt saves its index next to the snapshot.  Later runs load it in 0.9 s instead of building it.

Relative error of yt, (yt - vortrace) / vortrace, per pixel of the main image:

| percentile | 1 | 5 | 50 | 95 | 99 |
|---|---:|---:|---:|---:|---:|
| error | -0.123 | -0.077 | -0.002 | +0.074 | +0.123 |
