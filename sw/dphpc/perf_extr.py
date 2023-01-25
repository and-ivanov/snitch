#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

import pandas as pd
import argparse
import pathlib
from scipy import stats


def extract_perf_data(log_file: str, proc_id: int) -> list:

    # Read log file
    with open(log_file, "r") as f:
        extracting = False
        metrics_list = []
        metrics = {}
        # Read linse until Performance metrics are found
        for line in f:
            if line.startswith("Performance metrics"):
                extracting = True
                metrics = {"desc": f"proc_{proc_id}"}
                continue
            # If line is empty or has newline, stop extracting
            if line == "\n" and extracting:
                extracting = False
                metrics_list.append(metrics)
                metrics = {}
                continue
            if extracting:
                # extract first word of line
                metric_key = line.split()[0]
                # detect if last metric is a float or a hex value
                try:
                    metric_value = float(line.split()[-1])
                except ValueError:
                    metric_value = int(line.split()[-1], 16)
                # extract number from string
                metrics[metric_key] = metric_value

        # Append metrics to list
        if metrics != {}:
            metrics_list.append(metrics)
        return metrics_list


def main():

    script_path = pathlib.Path(__file__).parent.absolute()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Extract performance data from log files')
    parser.add_argument("-i", "--input", type=pathlib.Path, default=script_path / "build/logs", required=False,
                        help='Path to log files')
    parser.add_argument("-o", "--output", type=pathlib.Path, default=script_path / "perf_data.csv", required=False,
                        help='Path to output file')
    parser.add_argument("-s", "--section", type=int, default=1, required=False,
                        help='Section to extract data from')
    parser.add_argument("-n", "--nproc", type=int, default=8, required=False, help='Number of processes')
    parser.add_argument("-v", "--verbose", action='store_true', help='Set verbose')

    args = parser.parse_args()

    # Get all n first log files
    log_files = sorted(args.input.glob("trace_*.txt"))[:args.nproc]

    metrics_list = []
    metrics_barrier_list = []
    metrics_transform_list = []

    # Extract performance data
    for p, log_file in enumerate(log_files[:args.nproc]):
        # Extract performance data of current core file
        metrics_of_current_proc = extract_perf_data(log_file, p)
        # Append specific section to list of all cores
        try:
            metrics_list.append(metrics_of_current_proc[args.section])
        except IndexError:
            # print with orange color
            print(f"\033[33mWarning: Section {args.section} not found in {log_file}\033[0m")
        try:
            metrics_barrier_list.append(metrics_of_current_proc[args.section+1])
        except IndexError:
            # print with orange color
            print(f"\033[33mWarning: Section {args.section+1} not found in {log_file}\033[0m")
        try:
            metrics_transform_list.append(metrics_of_current_proc[args.section+2])
        except IndexError:
            # print with orange color
            print(f"\033[33mWarning: Section {args.section+2} not found in {log_file}\033[0m")

    # Convert list of dict to pandas dataframe
    df_kernel = pd.DataFrame(metrics_list)
    df_barrier = pd.DataFrame(metrics_barrier_list)
    df_transform = pd.DataFrame(metrics_transform_list)

    df = df_kernel

    if (args.nproc > 1):
        df.loc[:, "cycles"] = df_kernel.loc[:, "cycles"] + df_barrier.loc[:, "cycles"] + df_transform.loc[:, "cycles"]
        df.loc[:, "fpss_loads"] = df_kernel.loc[:, "fpss_loads"] + df_barrier.loc[:, "fpss_loads"] + df_transform.loc[:, "fpss_loads"]
        for k in ["snitch_occupancy", "fpss_occupancy", "fpss_fpu_occupancy", "fpss_fpu_rel_occupancy", "total_ipc"]:
            df.loc[:, k] = (df_kernel.loc[:, "cycles"] * df_kernel.loc[:, k]
                            + df_barrier.loc[:, "cycles"] * df_barrier.loc[:, k]
                            + df_transform.loc[:, "cycles"] * df_transform.loc[:, k]) / df.loc[:, "cycles"]
        df.loc[:, "synch_overhead"] = df_barrier.loc[:, "cycles"] / df.loc[:, "cycles"]
        df.loc[:, "transform_overhead"] = df_transform.loc[:, "cycles"] / df.loc[:, "cycles"]
        df.loc[:, "synch_cycles"] = df_barrier.loc[:, "cycles"]
        df.loc[:, "transform_cycles"] = df_transform.loc[:, "cycles"]
        del df["snitch_avg_load_latency"]
        del df["snitch_fseq_rel_offloads"]
        del df["fseq_yield"]
        del df["fseq_fpu_yield"]
        del df["fpss_section_latency"]
        del df["fpss_avg_fpu_latency"]
        del df["fpss_avg_load_latency"]

    # compute metrics
    df_mean = df.mean(numeric_only=True)
    df_hmean = pd.Series(stats.hmean(df.iloc[:, 1:], axis=0), index=df_mean.index)
    df_std = df.std(numeric_only=True)
    df_min = df.min()
    df_max = df.max()
    df_total = df.sum()

    # append mean and std to dataframe
    # concat as a new row
    df = pd.concat([df,
                   df_mean.to_frame().T,
                   df_hmean.to_frame().T,
                   df_std.to_frame().T,
                   df_min.to_frame().T,
                   df_max.to_frame().T,
                   df_total.to_frame().T])

    # rename last two rows
    df.iloc[-6, 0] = "mean"
    df.iloc[-5, 0] = "hmean"
    df.iloc[-4, 0] = "std"
    df.iloc[-3, 0] = "min"
    df.iloc[-2, 0] = "max"
    df.iloc[-1, 0] = "total"
    # pretty print dataframe without index
    print(df)

    # write dataframe to csv file
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
