// Assignment 1 – Multithreading Conversion and Runtime Comparison
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()

#set page("a4")
#set text(font: "New Computer Modern", 12pt)

#set heading(numbering: "1.")

#set document(title: "Assignment 2", date: auto)

#align(center, text(17pt)[
  *MICS6002C-Assignment 2*
])
#align(center, text(12pt)[
  *Xiwei Pan (50038447)*
])

= Show CPU and GPU information
 - CPU: Intel(R) Xeon(R) Platinum 8378A CPU \@ 3.00GHz
 - GPU: NVIDIA A800 80GB PCIe

= Test size 1024x1024
Note: Using simple tiled kernel with 16$times$16 tile size.

- Compile command:```
❯ nvcc -o assignment2 assignment2.cu -O3 -arch=sm_80
```
- Result:```
=== Testing Matrix Size: 1024x1024 ===
Check Suceeded
CPU GEMM Time: 303.928 ms
GPU GEMM Time: 0.569119 ms
Speedup: 534.033x
```

= Test different matrix sizes
Note: Using simple tiled kernel with 16$times$16 tile size.

#table(
  columns: (auto, auto, auto, auto),
  align: center,
  table.header([*Matrix Size*], [GPU Time (ms)], [CPU Time (ms)], [Speedup]),
  [128$times$128], [0.02], [0.56], [32.71$times$],
  [256$times$256], [0.03], [4.42], [129.97$times$],
  [512$times$512], [0.12], [35.36], [287.45$times$],
  [1024$times$1024], [0.52], [307.23], [596.57$times$],
)