#!/bin/bash
# Benchmark shell script

axis=0

APP="softmax_csr"
HW_DIR=$(dirname $(realpath “${BASH_SOURCE:-$0}”))
SW_DIR="$HW_DIR/../../../sw/dphpc"
RT_DIR="$HW_DIR/../../.."
RESULTS_DIR="$RT_DIR/results_softmax_mc_$APP"
echo $HW_DIR
echo $SW_DIR
echo $RT_DIR

for dim in 8 16 #10 20 30 40 50 60
do
#mkdir "$RESULTS_DIR/dim_$dim"
    for i in 1 #{1..50}
    do
        echo "Iteration $i"
        echo "$SW_DIR/data/data_gen_$APP.py" -d $dim -n 8 -ax $axis -t "$SW_DIR/data/data_$APP.h.tpl" -o "$SW_DIR/data/"
        "$SW_DIR/data/data_gen_$APP.py" -d $dim -n 8 -ax $axis -t "$SW_DIR/data/data_$APP.h.tpl" -o "$SW_DIR/data"
        make $APP -C sw/build/dphpc
        ./bin/snitch_cluster.vlt "./sw/build/dphpc/$APP"
        make traces
        "$SW_DIR/perf_extr.py" --nproc 8 --section 1 --input "./logs"

        #mv "$HW_DIR/logs" "$RESULTS_DIR/dim_$dim/logs$i"
        #cd $SW_DIR
        #echo "$RESULTS_DIR/dim_$dim/logs$i/results.csv"
        #./perf_extr.py --nproc 8 --section 1 --input "$RESULTS_DIR/dim_$dim/logs$i" #--output "$RESULTS_DIR/dim_$dim/logs$i/results.csv"
        #cd $HW_DIR
    done
done
