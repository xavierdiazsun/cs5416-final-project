# First run pipeline.py then to get the CSVs, then run plot_profiling.py

import os
import pandas as pd
import matplotlib.pyplot as plt

# Which nodes we expect
NODES = [0, 1, 2]
STAGES = ["embed", "ann", "docfetch", "rerank", "generate", "sentiment", "safety"]

def load_node_csv(node: int):
    fname = f"profiling_node{node}.csv"
    if not os.path.exists(fname):
        print(f"[WARN] {fname} does not exist, skipping.")
        return None
    df = pd.read_csv(fname)
    df["node"] = node
    return df

def add_stage_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your stage columns are cumulative times.
    For some plots it's useful to compute per-batch deltas.
    """
    for stage in STAGES:
        col = stage
        delta_col = stage + "_delta"
        if col in df.columns:
            df[delta_col] = df[col].diff().fillna(df[col])
    return df

def plot_throughput(dfs):
    plt.figure()
    for node, df in dfs.items():
        plt.plot(df["elapsed_seconds"], df["throughput_rps"], label=f"Node {node}")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Throughput (requests/s)")
    plt.title("Throughput vs Time per Node")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("throughput_vs_time.png")
    print("[OUT] Saved throughput_vs_time.png")

def plot_memory(dfs):
    plt.figure()
    for node, df in dfs.items():
        plt.plot(df["elapsed_seconds"], df["memory_mb"], label=f"Node {node}")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Memory (MB)")
    plt.title("Memory usage vs Time per Node")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("memory_vs_time.png")
    print("[OUT] Saved memory_vs_time.png")

def plot_stage_breakdown_cumulative(dfs):
    """
    For each node, take the last row (latest measurement) and show
    cumulative time spent in each stage as a bar chart.
    """
    for node, df in dfs.items():
        last = df.iloc[-1]
        values = [last[stage] for stage in STAGES]
        plt.figure()
        plt.bar(STAGES, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Cumulative time spent (s)")
        plt.title(f"Per-stage cumulative time - Node {node}")
        plt.tight_layout()
        fname = f"stage_breakdown_node{node}.png"
        plt.savefig(fname)
        print(f"[OUT] Saved {fname}")

def plot_stage_time_per_batch(dfs):
    """
    Optional: average per-batch stage time, using *_delta columns.
    This is nice to show that Node 2 is dominated by 'generate', etc.
    """
    for node, df in dfs.items():
        # Only use rows where we have deltas (skip early weird ones if needed)
        mean_deltas = []
        for stage in STAGES:
            col = stage + "_delta"
            if col in df.columns:
                mean_deltas.append(df[col].mean())
            else:
                mean_deltas.append(0.0)

        plt.figure()
        plt.bar(STAGES, mean_deltas)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Average per-batch time (s)")
        plt.title(f"Average per-batch stage time - Node {node}")
        plt.tight_layout()
        fname = f"stage_per_batch_node{node}.png"
        plt.savefig(fname)
        print(f"[OUT] Saved {fname}")

def main():
    dfs = {}
    for node in NODES:
        df = load_node_csv(node)
        if df is not None:
            df = add_stage_deltas(df)
            dfs[node] = df

    if not dfs:
        print("No profiling_node*.csv files found. Run the server/client first.")
        return

    plot_throughput(dfs)
    plot_memory(dfs)
    plot_stage_breakdown_cumulative(dfs)
    plot_stage_time_per_batch(dfs)

if __name__ == "__main__":
    main()