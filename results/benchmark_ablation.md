LWE ablation, 18 states, 200 steps, median of 5 runs

| configuration | enc us | eval us | dec us | call crit us | compute crit us | total us | speed-up | max u err | max x err | ct bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 original (python ints, sigma=sqrt18) | 104.040 | 6.664 | 12.324 | 122.551 | **122.551** | 122.551 | 1.0x | 6.12e-02 | 3.10e-03 | 72 |
| 2 + sigma=1 | 109.105 | 6.913 | 12.680 | 129.508 | **129.508** | 129.508 | 0.9x | 1.44e-02 | 7.25e-04 | 72 |
| 3 + NumPy int64 | 19.216 | 7.103 | 12.365 | 39.261 | **39.261** | 39.261 | 3.1x | 1.44e-02 | 7.25e-04 | 72 |
| 4 + Numba kernels | 2.807 | 0.855 | 1.007 | 4.669 | **0.754** | 0.754 | 162.6x | 1.44e-02 | 7.25e-04 | 72 |
| 5 + offline pad/FS split | 1.273 | 0.864 | 1.058 | 3.242 | **0.194** | 0.750 | 633.0x | 1.44e-02 | 7.25e-04 | 72 |
| 6 + wide modulus 2^61-1 (DEFAULT) | 1.093 | 0.848 | 0.957 | 2.878 | **0.224** | 1.217 | 546.6x | 3.29e-06 | 2.16e-07 | 144 |
