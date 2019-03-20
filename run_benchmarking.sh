
#!/bin/bash

make

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
LOG_DIR=logs/
mkdir $OUTPUT_DIR 2>/dev/null
mkdir $LOG_DIR 2>/dev/null
for i in $INPUT_DIR/*gif ; do
  touch log.txt
  LOG=$LOG_DIR`basename $i .gif`.log
  DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
  echo "Running test on $i -> $DEST"
  for N in 1 4 16
  do
    for k in 1 3 5
    do
      for t in 1 4 8
      do
        OMP_NUM_THREADS=$t salloc -n $[$k*$N] -N $N mpirun ./sobelf $i $DEST > "log-tmp.out"
        ./test_regression.sh
      done
    done
  done

  mv log.txt $LOG

done
