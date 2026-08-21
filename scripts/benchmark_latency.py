#!/usr/bin/env python3
"""
Standalone Performance Benchmark for Local RAG

This script measures Time-to-First-Token (TTFT), total response latency, 
tokens/sec, and internal RAG telemetry (if exposed).
It handles HTTP SSE streaming for accurate TTFT measurement.

Usage:
    python scripts/benchmark_latency.py --host http://localhost:8000 --warm
    python scripts/benchmark_latency.py --host http://localhost:8000 --cold
"""

import sys
import time
import json
import argparse
import asyncio
import statistics
import httpx
from typing import Dict, List, Any

# Questions derived from user requirements
SCENARIOS = {
    "Simple": "What is the leave policy?",
    "Medium": "What are the backend deployment requirements?",
    "Complex": "Compare the frontend and backend deployment configurations and explain how HTTPS is configured."
}

async def run_scenario(client: httpx.AsyncClient, base_url: str, scenario_name: str, question: str) -> Dict[str, Any]:
    url = f"{base_url}/api/chat"
    payload = {"question": question, "stream": True}
    
    start_time = time.monotonic()
    first_token_time = None
    end_time = None
    tokens_generated = 0
    telemetry = {}

    try:
        # Fallback to standard /api/chat if it auto-negotiates SSE via stream=True
        headers = {"Accept": "text/event-stream"}
        
        async with client.stream("POST", url, json=payload, headers=headers, timeout=120.0) as response:
            if response.status_code != 200:
                body = await response.aread()
                print(f"[{scenario_name}] HTTP {response.status_code}: {body.decode(errors='replace')}")
                return {"error": f"HTTP {response.status_code}"}

            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                        # Detect first token
                        if first_token_time is None and (data.get("token") or data.get("text") or data.get("type") == "token"):
                            first_token_time = time.monotonic()
                        
                        # Count tokens
                        if data.get("token") or data.get("type") == "token":
                            tokens_generated += 1
                            
                        # Capture telemetry if present
                        if "telemetry" in data or data.get("type") == "telemetry":
                            meta = data.get("telemetry", data)
                            telemetry.update({k: v for k, v in meta.items() if "latency" in k or "time" in k})
                            
                        # Alternative schema handling (if backend returns final usage stats)
                        if "usage" in data:
                            if "total_tokens" in data["usage"]:
                                tokens_generated = data["usage"].get("completion_tokens", tokens_generated)
                            
                    except json.JSONDecodeError:
                        pass # Ignore malformed SSE lines
                        
    except httpx.ReadTimeout:
        return {"error": "Timeout"}
    except httpx.RequestError as e:
        return {"error": f"Connection Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}

    end_time = time.monotonic()
    
    # If no streaming tokens were detected, assume it was a synchronous JSON response
    if first_token_time is None:
        first_token_time = end_time

    ttft = (first_token_time - start_time) * 1000
    total_latency = (end_time - start_time) * 1000
    generation_latency = (end_time - first_token_time) * 1000
    
    tps = 0.0
    if generation_latency > 0 and tokens_generated > 0:
        tps = tokens_generated / (generation_latency / 1000)

    return {
        "scenario": scenario_name,
        "ttft_ms": ttft,
        "total_latency_ms": total_latency,
        "generation_latency_ms": generation_latency,
        "tokens": tokens_generated,
        "tps": tps,
        "telemetry": telemetry,
        "error": None
    }

def print_stats(name: str, results: List[Dict[str, Any]]):
    valid = [r for r in results if not r.get("error")]
    if not valid:
        print(f"{name}: All requests failed.")
        return

    ttft_list = [r["ttft_ms"] for r in valid]
    total_list = [r["total_latency_ms"] for r in valid]
    tps_list = [r["tps"] for r in valid]
    
    def p(lst, pctl):
        lst_sorted = sorted(lst)
        idx = int(len(lst_sorted) * pctl)
        return lst_sorted[idx]

    print(f"\n{name} Results ({len(valid)}/{len(results)} successful):")
    print(f"  TTFT:  p50={p(ttft_list, 0.5):.0f}ms, p95={p(ttft_list, 0.95):.0f}ms, min={min(ttft_list):.0f}ms, max={max(ttft_list):.0f}ms")
    print(f"  Total: p50={p(total_list, 0.5):.0f}ms, p95={p(total_list, 0.95):.0f}ms, avg={statistics.mean(total_list):.0f}ms")
    print(f"  TPS:   avg={statistics.mean(tps_list):.1f} tokens/sec")

async def run_cold_start(base_url: str) -> Dict[str, Any]:
    print("\n--- Running Cold Start Measurement ---")
    print("Ensure the backend was JUST RESTARTED before running this.")
    async with httpx.AsyncClient() as client:
        result = await run_scenario(client, base_url, "Cold_Start", SCENARIOS["Simple"])
    
    if result.get("error"):
        print(f"Cold start failed: {result['error']}")
    else:
        print(f"First request:")
        print(f"  TTFT:       {result['ttft_ms']:.0f}ms")
        print(f"  Total:      {result['total_latency_ms']:.0f}ms")
        print(f"  Tokens/sec: {result['tps']:.1f}")
        if result['telemetry']:
            print(f"  Telemetry:  {json.dumps(result['telemetry'], indent=2)}")
    return result

async def run_warm_measurements(base_url: str, iterations: int) -> List[Dict[str, Any]]:
    print(f"\n--- Running Warm Measurements ({iterations} iterations per scenario) ---")
    results = []
    
    # We use a single persistent client to simulate a frontend connection pool
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Pre-warm query to ensure everything is loaded
        print("Pre-warming...")
        await run_scenario(client, base_url, "Warmup", "Hello")
        
        # Summary table header
        print("\nScenario       Samples   P50 TTFT  P95 TTFT  P50 Total Avg TPS")
        print("-" * 62)
        
        for name, question in SCENARIOS.items():
            scenario_results = []
            for i in range(iterations):
                res = await run_scenario(client, base_url, name, question)
                scenario_results.append(res)
            
            results.extend(scenario_results)
            
            valid = [r for r in scenario_results if not r.get("error")]
            if valid:
                ttft_list = sorted([r["ttft_ms"] for r in valid])
                total_list = sorted([r["total_latency_ms"] for r in valid])
                tps_list = [r["tps"] for r in valid]
                
                p50_ttft = ttft_list[int(len(ttft_list) * 0.5)]
                p95_ttft = ttft_list[int(len(ttft_list) * 0.95)]
                p50_tot = total_list[int(len(total_list) * 0.5)]
                avg_tps = statistics.mean(tps_list)
                
                print(f"{name:<14} {len(valid):<9} {p50_ttft:<9.0f} {p95_ttft:<9.0f} {p50_tot:<9.0f} {avg_tps:.1f}")
            else:
                print(f"{name:<14} 0         -         -         -         -")
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Standalone Local RAG Performance Benchmark")
    parser.add_argument("--host", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--cold", action="store_true", help="Run a single cold-start measurement")
    parser.add_argument("--warm", action="store_true", help="Run multiple warm measurements")
    parser.add_argument("--iter", type=int, default=10, help="Iterations per scenario for warm runs")
    parser.add_argument("--out", type=str, help="Output JSON file path (e.g. benchmark_after.json)")
    
    args = parser.parse_args()
    
    if not args.cold and not args.warm:
        print("Please specify --cold or --warm (or both).")
        sys.exit(1)
        
    all_results = []
    
    if args.cold:
        res = asyncio.run(run_cold_start(args.host))
        all_results.append(res)
        
    if args.warm:
        res_list = asyncio.run(run_warm_measurements(args.host, args.iter))
        all_results.extend(res_list)
        
    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.out}")

if __name__ == "__main__":
    main()
