#!/usr/bin/env python3
"""
High-load client for measuring max throughput of the ML pipeline.

Spawns multiple worker threads. Each worker sends requests in a tight loop
until TEST_DURATION seconds have elapsed. At the end, we compute
total successful requests and convert to requests/minute.
"""

import os
import time
import threading
import requests
from datetime import datetime
from typing import Dict, List

NODE_0_IP = os.environ.get("NODE_0_IP", "localhost:8000")
SERVER_URL = f"http://{NODE_0_IP}/query"

NUM_WORKERS = 1
TEST_DURATION = 120

TEST_QUERIES = [
    "How do I return a defective product?",
    "What is your refund policy?",
    "My order hasn't arrived yet, tracking number is ABC123",
    "How do I update my billing information?",
    "Is there a warranty on electronic items?",
    "Can I change my shipping address after placing an order?",
    "What payment methods do you accept?",
    "How long does shipping typically take?"
]

results_lock = threading.Lock()
success_count = 0
failure_count = 0
latencies: List[float] = []  # optional: end-to-end latency under high load


def worker_thread(worker_id: int, stop_time: float):
    global success_count, failure_count
    i = 0
    session = requests.Session()

    while time.time() < stop_time:
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        request_id = f"worker{worker_id}_req{i}_{int(time.time())}"
        payload = {"request_id": request_id, "query": query}

        start = time.time()
        try:
            resp = session.post(SERVER_URL, json=payload, timeout=300)
            elapsed = time.time() - start

            with results_lock:
                if resp.status_code == 200:
                    success_count += 1
                    latencies.append(elapsed)
                else:
                    failure_count += 1
        except Exception:
            with results_lock:
                failure_count += 1

        i += 1


def main():
    print("=" * 70)
    print("HIGH-LOAD THROUGHPUT TEST CLIENT")
    print("=" * 70)
    print(f"Server URL: {SERVER_URL}")
    print(f"Workers: {NUM_WORKERS}, Duration: {TEST_DURATION}s")
    print("=" * 70)

    # Optional: health check
    try:
        health = requests.get(f"http://{NODE_0_IP}/health", timeout=5)
        print("Health:", health.status_code, health.text)
    except Exception as e:
        print("Health check failed:", e)

    start_time = time.time()
    stop_time = start_time + TEST_DURATION

    threads = []
    for w in range(NUM_WORKERS):
        t = threading.Thread(target=worker_thread, args=(w, stop_time))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    total_time = time.time() - start_time

    # Compute throughput (req/min)
    with results_lock:
        total_success = success_count
        total_fail = failure_count
        total = total_success + total_fail

    rpm = total_success / (total_time / 60.0)

    print("\n" + "=" * 70)
    print("THROUGHPUT SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total requests: {total}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_fail}")
    print(f"Max observed throughput: {rpm:.1f} successful requests/min")

    # Optional: latency stats under high load
    if latencies:
        import statistics
        avg = statistics.mean(latencies)
        med = statistics.median(latencies)
        p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1]
        print("\nLatency under high load:")
        print(f"  Avg: {avg:.2f}s, Median: {med:.2f}s, p95: {p95:.2f}s")


if __name__ == "__main__":
    main()