#!/bin/bash

# policy -> figure legend mapping
# shockwave: Shockwave
# min_total_duration: OSSP
# finish_time_fairness: Themis
# max_min_fairness: Gavel
# allox: AlloX
# max_sum_throughput_perf: MST
# gandiva_fair: Gandiva-Fair

# FYI: In our development environment,
# Simulating Shockwave on this trace takes ~1 hour. 
# The other policies take less than a minute each to complete.

for POLICY in shockwave min_total_duration finish_time_fairness max_min_fairness allox max_sum_throughput_perf gandiva_fair
do
    python3 ../scripts/drivers/simulate_scheduler_with_trace.py \
        --trace_file ../traces/reproduce/460_0.2_5_100_10_1_0,0.5,0.5_0.6,0.3,0.09,0.01_multigpu_dynamic.trace \
        --policy $POLICY \
        --throughputs_file ../wisr_throughputs.json \
        --cluster_spec 128:0:0 \
        --seed 0 --solver ECOS \
        --time_per_iteration 120 \
        --config ../configurations/scale_128gpus.json \
        --pickle_output_dir ../../reproduce/pickles \
        > ./test_$POLICY.txt 2>&1
done