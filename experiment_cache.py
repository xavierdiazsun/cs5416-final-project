#!/usr/bin/env python3
import os
import time
import requests
import random
import string
import statistics

NODE_0_IP = os.environ.get("NODE_0_IP", "localhost:8000")
SERVER_URL = f"http://{NODE_0_IP}/query"
NUM_REQUESTS = 10  

RUN_ID = int(time.time())
BASE_QUERIES = [
    f"How do I return a defective product I bought yesterday? [{RUN_ID}]",
    f"What is your refund policy for electronics vs clothing? [{RUN_ID}]",
    f"My order hasn't arrived yet, tracking number is ABC12345. [{RUN_ID}]",
    f"How do I update my billing information on the dashboard? [{RUN_ID}]",
    f"Is there a manufacturer warranty on the new headphones? [{RUN_ID}]",
    f"Can I change my shipping address after placing an order? [{RUN_ID}]",
    f"What payment methods do you accept for international orders? [{RUN_ID}]",
    f"How long does standard shipping typically take to New York? [{RUN_ID}]",
    f"I received the wrong item in my package, what should I do? [{RUN_ID}]",
    f"Do you offer gift wrapping services for holiday items? [{RUN_ID}]"
]

def generate_first_pass(i):
    """First pass: These should be MISSES."""
    return BASE_QUERIES[i]

def generate_second_pass(i):
    """Second pass: Same queries, should be HITS."""
    return BASE_QUERIES[i]

def run_experiment(name, query_func):
    print(f"\n--- Running Experiment: {name} ---")
    latencies = []
    
    session = requests.Session()
    
    start_global = time.time()
    
    for i in range(NUM_REQUESTS):
        query = query_func(i)
        req_id = f"exp_{name}_{i}_{int(time.time())}"
        
        t0 = time.time()
        try:
            resp = session.post(SERVER_URL, json={'request_id': req_id, 'query': query}, timeout=300)
            t1 = time.time()
            
            if resp.status_code == 200:
                latencies.append((t1 - t0) * 1000)
                print(f"Req {i+1}/{NUM_REQUESTS}: {(t1-t0)*1000:.1f}ms")
            else:
                print(f"Req {i+1}/{NUM_REQUESTS}: Failed ({resp.status_code})")
        except Exception as e:
            print(f"Req {i+1}/{NUM_REQUESTS}: Error {e}")

    total_time = time.time() - start_global
    
    if not latencies:
        return 0, 0
        
    avg_lat = statistics.mean(latencies)
    throughput = len(latencies) / (total_time / 60) 
    
    print(f"Result: Avg Latency = {avg_lat:.1f}ms | Throughput = {throughput:.1f} RPM")
    return avg_lat, throughput

def main():
    print("="*60)
    print("CACHING PERFORMANCE EXPERIMENT")
    print("="*60)
    
    try:
        requests.get(f"http://{NODE_0_IP}/health", timeout=2)
    except:
        print("Could not connect to server. Check IP.")
        return

    # cache misses
    lat_miss, rpm_miss = run_experiment("No_Cache_Pass", generate_first_pass)
    
    # cache hits
    lat_hit, rpm_hit = run_experiment("With_Cache_Pass", generate_second_pass)
    
    print("\n" + "="*60)
    print("FINAL RESULTS TABLE")
    print("="*60)
    print(f"{'Metric':<20} | {'Pass 1 (Miss)':<20} | {'Pass 2 (Hit)':<20}")
    print("-" * 66)
    print(f"{'Avg Latency':<20} | {lat_miss:.1f} ms{'':<13} | {lat_hit:.1f} ms")
    print(f"{'Throughput':<20} | {rpm_miss:.1f} req/min{'':<9} | {rpm_hit:.1f} req/min")
    print("-" * 66)
    
    improvement = lat_miss / lat_hit if lat_hit > 0 else 0
    print(f"\nOptimization Factor: {improvement:.1f}x faster")

if __name__ == "__main__":
    main()