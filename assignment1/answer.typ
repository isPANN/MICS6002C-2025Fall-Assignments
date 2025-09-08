// Assignment 1 – Multithreading Conversion and Runtime Comparison
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()

#set page("a4")
#set text(font: "New Computer Modern", 12pt)

#set heading(numbering: "1.")

#set document(title: "Assignment 1", date: auto)

#align(center, text(17pt)[
  *MICS6002C-Assignment 1*
])
#align(center, text(12pt)[
  *Xiwei Pan (50038447)*
])

= Description of Single-threaded Version
The original single-threaded program initializes four $N times N$ integer matrices with pseudorandom values and performs four matrix multiplications (index $0 times 1$, $1 times 2$, $2 times 3$, $1 times 3$), accumulating the sum of all resulting elements modulo $10,000,000$ and printing the final total.


= Runtime Experimental Results
To evaluate the performance difference, both the single-threaded and multithreaded versions were compiled with identical options and executed multiple times on the same machine. The average runtimes over 10 runs were measured using the `time` command. The results, summarized in the table below, clearly show that the multithreaded implementation achieves a substantial reduction in execution time compared to the single-threaded version.

#codly(languages: codly-languages)
- CPU: Apple M3 with 8 cores
- Compile command:```
❯ g++ -std=c++14 -O2 -pthread -o multi multi.cpp
```

#align(center, 
figure(caption: "Runtime Comparison",
table(
  columns: (auto, auto, auto),
  align: center,
  table.header([*Averaged over 10 runs*], [Single-threaded], [Multithreaded (4 threads)]),
  [Real Runtime (s)], [2.791], [0.793],
  [User CPU Time (s)], [2.779], [2.859],
  [System CPU Time (s)], [0.005], [0.012],
  [Speedup($times$)], [--], [3.52],
)))

In the multithreaded results, the User CPU time is less than four times the Real time, indicating that the workload did not fully scale across all four threads. This contrasts with the single-threaded case, where the User CPU time closely matches the Real time, since all computation is executed on a single core without parallel overlap.
= Description of Multithreaded Version
Each of the four threads multiplies two matrices and stores partial results in `sum`. After computing, the threads acquire a `mutex` to safely update the shared global `total`.
