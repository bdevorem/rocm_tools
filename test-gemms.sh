#!/bin/bash

#set -x

LOGFILE=./mlir.perf.$1
LOGSUMMARYFILE=./mlir_summary.perf.$1
#DRIVERPATH="/opt/rocm/bin/migraphx-driver"
DRIVERPATH="./build/bin/migraphx-driver"

export HIP_FORCE_DEV_KERNARG=1

echo "###########################################" >>  $LOGFILE
echo "New Run $(pwd)" >>  $LOGFILE
date >> $LOGFILE
echo "GPU: $(rocminfo |grep -o -m 1 'gfx.*')" >> $LOGFILE
echo "MIGX: $($DRIVERPATH --version)" >> $LOGFILE
echo "MIGX Commit: $(git -C /workspace/AMDMIGraphX log -n 1  --pretty=oneline)" >> $LOGFILE
ls -l /etc/alternatives |grep "rocm ->" >> $LOGFILE
echo "###########################################" >>  $LOGFILE

COUNTER=0
mkdir -p gemmtests

function run_test {
    title=$1
    env_vars=$2
    m=$3
    k1=$4
    n1k2=$5
    n2=$6
    batch=$7
    datatype=$8
    modelname="gemmtests/testkernel_""$m""_""$k1""_""$n1k2""_""$n2"".py"
    display_type=${datatype:-'--fp32'}

    sed "s/\\\$m\\\$/$m/g ; s/\\\$k1\\\$/$k1/g ; s/\\\$n1k2\\\$/$n1k2/g ; s/\\\$n2\\\$/$n2/g" test_kernel.py > "$modelname"
    (( COUNTER++ ))
    echo "TEST: $COUNTER, $title $env_vars m=$m k1=$k1 n1k2=$n1k2 n2=$n2 batch=$b datatype=$display_type" >> $LOGFILE
    
    ( if [ -n "$env_vars" ]; then export $env_vars; fi; time $DRIVERPATH perf "$modelname" $datatype --batch $batch ) 2>&1 |tee raw_perf.txt
    cat raw_perf.txt |sed -n '/Summary:/,$p'  >>  $LOGFILE
    
    runtime=$(tail raw_perf.txt |grep "real"|cut -f2- )
    totaltime=$(grep 'Total time:' raw_perf.txt|cut -d ' ' -f 3 |cut -d 'm' -f 1)
    percentiles=$(grep 'Percentiles (90%, 95%, 99%):' raw_perf.txt|cut -d '(' -f 2| cut -d ')' -f1| tr ',' ' ')
    insttime=$(grep 'Total instructions time:' raw_perf.txt|cut -d ':' -f 2 | tr ',' ' ')
    overheadtime=$(grep 'Overhead time:' raw_perf.txt|cut -d ':' -f 2 | tr ',' ' ')
    overhead=$(grep 'Overhead:' raw_perf.txt|cut -d ':' -f 2 | tr ',' ' ')
    echo "TEST: $COUNTER, $runtime, $title, $modelname, $display_type, $batch, $totaltime, $percentiles, $insttime, $overheadtime, $overhead" >> $LOGSUMMARYFILE
}

# MIGRAPHX_MLIR_TRACE=1 MIGRAPHX_TRACE_BENCHMARKING=2
MIGRAPHX_PROBLEM_CACHE=/workspace/perf.json

while read m k1 n1k2 n2
do

    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 1 --fp16
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 1
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 16 --fp16
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 16
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 32 --fp16
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 32
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 64 --fp16
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 64
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 128 --fp16
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 128
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 256 --fp16
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 256
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 512 --fp16
    run_test "DEFAULT" "" "$m" "$k1" "$n1k2" "$n2" 512

# m      k1       n1k2     n2
done <<TESTLIST
168 128 816 1648
168 128 976 664
168 128 2960 1776
168 1 1 1776
168 360 128 360
168 976 664 88
168 368 128 368
168 128 1336 1104
168 128 360 1776
168 200 200 1776
168 128 368 1776
168 128 328 1776
168 128 536 1776
168 816 128 816
168 328 128 328
168 2120 128 2120
168 1000 128 1000
168 128 448 1776
168 20 20 1776
168 128 1000 1776
168 50 50 1776
168 70 70 1776
168 448 128 448
168 2960 128 2960
168 128 296 1776
168 100 100 1776
168 160 160 1776
168 296 128 296
168 536 128 536
168 128 2120 1776
10       20       30       40
100      20       300      400
8192     128      8192     128
16384    128      16384    128
2        2        2        2
4        4        4        4
8        8        8        8
16       16       16       16
32       32       32       32
64       64       64       64
128      128      128      128
256      256      256      256
512      512      512      512
1024     1024     1024     1024
2048     2048     2048     2048
4096     4096     4096     4096
8192     8192     8192     8192
2        4        8        16
4        8        16       32
8        16       32       64
16       32       64       128
32       64       128      256
64       128      256      512
128      256      512      1024
256      512      1024     2048
512      1024     2048     4096
1024     2048     4096     8192
1024     128      1024     128
2048     128      2048     128
4096     128      4096     128
8192     128      8192     128
128      1024     128      1024
128      2048     128      2048
128      4096     128      4096
128      8192     128      8192
1024     64       1024     64
2048     64       2048     64
4096     64       4096     64
8192     64       8192     64
64       1024     64       1024
64       2048     64       2048
64       4096     64       4096
64       8192     64       8192
1024     256      1024     256
2048     256      2048     256
4096     256      4096     256
8192     256      8192     256
256      1024     256      1024
256      2048     256      2048
256      4096     256      4096
256      8192     256      8192
1024     512      1024     512
2048     512      2048     512
4096     512      4096     512
8192     512      8192     512
512      1024     512      1024
512      2048     512      2048
512      4096     512      4096
512      8192     512      8192
TESTLIST
