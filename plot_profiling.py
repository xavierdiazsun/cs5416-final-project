import os
import pandas as pd
import matplotlib.pyplot as plt

# Which nodes we expect
NODES = [0, 1, 2]
STAGES = ["embed", "ann", "docfetch", "rerank", "generate", "sentiment", "safety"]

IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

def load_node_csv(node: int):
    fname = f"profiling_node{node}.csv"
    if not os.path.exists(fname):
        print(f"[WARN] {fname} does not exist, skipping.")
        return None
    df = pd.read_csv(fname)
    df["node"] = node
    return df

def add_stage_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """ Compute per-batch deltas for cumulative stage columns. """
    for stage in STAGES:
        delta_col = stage + "_delta"
        if stage in df.columns:
            df[delta_col] = df[stage].diff().fillna(df[stage])
    return df

def plot_local_throughput(dfs):
    """ Plot local throughput over time for each node. """
    plt.figure()
    for node, df in dfs.items():
        if "local_throughput_rps" in df.columns:
            col = "local_throughput_rps"
        else:
            print(f"[WARN] Node {node} missing local_throughput_rps (using total_throughput_rps).")
            col = "total_throughput_rps"

        plt.plot(df["elapsed_seconds"], df[col], label=f"Node {node}")

    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Local throughput (requests/s)")
    plt.title("Local Throughput vs Time per Node")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(IMG_DIR, "local_throughput_vs_time.png")
    plt.savefig(fname)
    print(f"[OUT] Saved {fname}")

def plot_peak_memory(dfs):
    """
    Show peak memory used by each node (single bar per node).
    """
    nodes = []
    peaks = []
    for node, df in dfs.items():
        if "memory_mb" not in df.columns:
            print(f"[WARN] Node {node} missing memory_mb, skipping in peak memory plot.")
            continue
        nodes.append(node)
        peaks.append(df["memory_mb"].max())

    plt.figure()
    plt.bar([str(n) for n in nodes], peaks)
    plt.xlabel("Node")
    plt.ylabel("Peak memory (MB)")
    plt.title("Peak Memory Usage per Node")
    plt.tight_layout()

    fname = os.path.join(IMG_DIR, "peak_memory_per_node.png")
    plt.savefig(fname)
    print(f"[OUT] Saved {fname}")

def plot_stage_breakdown_cumulative(dfs):
    for node, df in dfs.items():
        if node == 0:
            node_stages = ["embed"]
        elif node == 1:
            node_stages = ["ann", "docfetch", "rerank"]
        elif node == 2:
            node_stages = ["generate", "sentiment", "safety"]
        else:
            node_stages = STAGES

        node_stages = [s for s in node_stages if s in df.columns]
        if not node_stages:
            print(f"[WARN] Node {node} has no valid stage columns, skipping cumulative plot.")
            continue

        last = df.iloc[-1]
        values = [last[stage] for stage in node_stages]

        plt.figure()
        plt.bar(node_stages, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Cumulative time spent (s)")
        plt.title(f"Per-stage cumulative time - Node {node}")
        plt.tight_layout()

        fname = os.path.join(IMG_DIR, f"stage_breakdown_node{node}.png")
        plt.savefig(fname)
        print(f"[OUT] Saved {fname}")

def plot_stage_time_per_batch(dfs):
    for node, df in dfs.items():
        if node == 0:
            node_stages = ["embed"]
        elif node == 1:
            node_stages = ["ann", "docfetch", "rerank"]
        elif node == 2:
            node_stages = ["generate", "sentiment", "safety"]
        else:
            node_stages = STAGES

        delta_stages = []
        mean_deltas = []
        for stage in node_stages:
            col = stage + "_delta"
            if col in df.columns:
                delta_stages.append(stage)
                mean_deltas.append(df[col].mean())

        if not delta_stages:
            print(f"[WARN] Node {node} has no delta columns for selected stages, skipping per-batch plot.")
            continue

        plt.figure()
        plt.bar(delta_stages, mean_deltas)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Avg per-batch time (s)")
        plt.title(f"Average per-batch stage time - Node {node}")
        plt.tight_layout()

        fname = os.path.join(IMG_DIR, f"stage_per_batch_node{node}.png")
        plt.savefig(fname)
        print(f"[OUT] Saved {fname}")

def plot_stage_breakdown_cumulative_combined(dfs):
    plt.figure(figsize=(10, 6))

    bar_width = 0.25
    x = range(len(STAGES))

    for i, (node, df) in enumerate(dfs.items()):
        last = df.iloc[-1]
        values = [last[stage] for stage in STAGES]

        offsets = [p + i * bar_width for p in x]
        plt.bar(offsets, values, width=bar_width, label=f"Node {node}")

    plt.xticks(
        [p + bar_width for p in x], 
        STAGES, rotation=45, ha="right"
    )
    plt.ylabel("Cumulative time spent (s)")
    plt.title("Cumulative Per-Stage Time (All Nodes)")
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(IMG_DIR, "stage_breakdown_cumulative_combined.png")
    plt.savefig(fname)
    print(f"[OUT] Saved {fname}")

def plot_stage_time_per_batch_combined(dfs):
    plt.figure(figsize=(10, 6))

    bar_width = 0.25
    x = range(len(STAGES))

    for i, (node, df) in enumerate(dfs.items()):
        mean_deltas = []
        for stage in STAGES:
            col = stage + "_delta"
            mean_deltas.append(df[col].mean() if col in df.columns else 0.0)

        offsets = [p + i * bar_width for p in x]
        plt.bar(offsets, mean_deltas, width=bar_width, label=f"Node {node}")

    plt.xticks(
        [p + bar_width for p in x],
        STAGES, rotation=45, ha="right"
    )
    plt.ylabel("Average per-batch time (s)")
    plt.title("Average Per-Batch Stage Time (All Nodes)")
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(IMG_DIR, "stage_per_batch_combined.png")
    plt.savefig(fname)
    print(f"[OUT] Saved {fname}")

def plot_memory_vs_throughput_by_batch(dfs):
    """
    Show tradeoffs between throughput and memory usage across different choices
    of batch size, per node.
    """
    for node, df in dfs.items():
        if "batch_size" not in df.columns:
            print(f"[WARN] Node {node} has no batch_size column, skipping memory-vs-throughput plot.")
            continue

        grouped = df.groupby("batch_size").agg({
            "memory_mb": "max",
            "local_throughput_rps": "max"
        }).reset_index()

        plt.figure()
        plt.scatter(grouped["memory_mb"], grouped["local_throughput_rps"])
        for _, row in grouped.iterrows():
            plt.annotate(f"bs={int(row['batch_size'])}",
                         (row["memory_mb"], row["local_throughput_rps"]),
                         textcoords="offset points", xytext=(5,5), fontsize=8)

        plt.xlabel("Peak memory (MB)")
        plt.ylabel("Max local throughput (req/s)")
        plt.title(f"Memory vs Throughput by Batch Size - Node {node}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = os.path.join(IMG_DIR, f"memory_vs_throughput_node{node}.png")
        plt.savefig(fname)
        print(f"[OUT] Saved {fname}")

def plot_max_throughput_summary(dfs):
    nodes = []
    max_rpm = []
    for node, df in dfs.items():
        if "local_throughput_rps" not in df.columns:
            print(f"[WARN] Node {node} has no local_throughput_rps, skipping in max throughput summary.")
            continue
        nodes.append(node)
        peak_rps = df["local_throughput_rps"].max()
        max_rpm.append(peak_rps * 60.0)

    plt.figure()
    plt.bar([str(n) for n in nodes], max_rpm)
    plt.xlabel("Node")
    plt.ylabel("Max local throughput (requests/min)")
    plt.title("Max Observed Throughput per Node")
    plt.tight_layout()

    fname = os.path.join(IMG_DIR, "max_throughput_per_node.png")
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

    plot_local_throughput(dfs)

    plot_peak_memory(dfs)

    plot_stage_breakdown_cumulative(dfs)
    plot_stage_time_per_batch(dfs)
    plot_stage_breakdown_cumulative_combined(dfs)
    plot_stage_time_per_batch_combined(dfs)

    plot_memory_vs_throughput_by_batch(dfs)

    plot_max_throughput_summary(dfs)

if __name__ == "__main__":
    main()