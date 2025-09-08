#!/bin/bash
# usage: ./benchmark.sh ./multi 10
# arg1 = the executable file path
# arg2 = the number of runs

prog=$1
runs=${2:-5}   # default 5 runs

if [ ! -x "$prog" ]; then
  echo "Error: $prog not found or not executable"
  exit 1
fi

echo "Benchmarking $prog for $runs runs..."

tmpfile=$(mktemp)

for i in $(seq 1 $runs); do
  /usr/bin/time -p $prog > /dev/null 2>> "$tmpfile"
done

echo "--- Average over $runs runs ---"
for metric in real user sys; do
  avg=$(grep $metric "$tmpfile" | awk '{sum+=$2} END {if (NR>0) print sum/NR; else print 0}')
  echo "$metric: $avg seconds"
done

rm -f "$tmpfile"