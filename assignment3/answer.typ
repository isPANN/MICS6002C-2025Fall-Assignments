// Assignment 1 – Multithreading Conversion and Runtime Comparison
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()

#set page("a4")
#set text(font: "New Computer Modern", 12pt)

#set heading(numbering: "1.")

// #set document(title: "Assignment 3", date: auto)

#align(center, text(17pt)[
  *MICS6002C-Assignment 3*
])
#align(center, text(12pt)[
  *Xiwei Pan (50038447)*
])

= Show CPU and GPU information
 - CPU: Intel(R) Xeon(R) Platinum 8378A CPU \@ 3.00GHz
 - GPU: NVIDIA A800 80GB PCIe

= Compile command
```
❯ nvcc -O3 -std=c++17 -arch=sm_80 -o main main.cu
```
= Result

#table(
  columns: (auto, auto, auto, auto, auto),
  align: center,
  table.header(table.cell(colspan: 2)[*Task*], [CPU (ms)], [GPU Total (ms)], [GPU Kernel-only (ms)]),
  table.cell(colspan: 2)[A], [2.492], [6.954], [3.709],
  table.cell(colspan: 2)[B], [2542.773], [2322.903], [23.679],
  table.cell(colspan: 2)[C], [/], [3783.439], [37.397],
)