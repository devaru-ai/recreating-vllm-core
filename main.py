import os
import time
import random
import torch
import matplotlib.pyplot as plt
import logging
import numpy as np
from model.loader import load_model_and_tokenizer, encode
from engine.request import Request
from engine.kv_cache_manager import KVCacheManager
from engine.engine_core import EngineCore

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def generate_prompts(n, min_len=5, max_len=15):
    base = "The quick brown fox jumps over the lazy dog. " * 10
    return [base[:random.randint(min_len, max_len)] for _ in range(n)]

def run_engine(engine, requests, metrics, max_steps):
    ttft, tpot, e2e = {}, {}, {}
    token_times, start_times, end_times = {}, {}, {}
    total_tokens, completed = 0, 0

    def print_fn(msg):
        nonlocal total_tokens, completed
        if "] >> " not in msg:
            return
        rid = msg.split("]")[0][1:]
        now = time.time()
        if "[FORCED E2E END]" in msg:
            e2e[rid] = now - metrics["submit_times"][rid]
            completed += 1
            return
        if rid not in start_times:
            start_times[rid] = now
            ttft[rid] = now - metrics["submit_times"][rid]
        raw = msg.split(">>")[1].strip()
        toks = raw.split()
        if not toks:
            toks = ["<BLANK>"]
        token_times.setdefault(rid, []).append(now)
        total_tokens += len(toks)
        max_toks = requests[metrics["rid_map"][rid]].max_tokens
        last_tok = toks[-1] if toks else ""
        if len(toks) >= max_toks or last_tok == str(engine.eos_token_id):
            e2e[rid] = now - metrics["submit_times"][rid]
            completed += 1

    for req in requests:
        engine.add_request(req)
    engine.run(max_steps=max_steps, print_fn=print_fn)
    for rid, times in token_times.items():
        if len(times) > 1:
            tpot[rid] = sum(t2 - t1 for t1, t2 in zip(times[:-1], times[1:])) / (len(times) - 1)
        else:
            tpot[rid] = None
    current = time.time()
    for rid in list(set(ttft.keys()) | set(token_times.keys())):
        if rid not in e2e:
            e2e[rid] = current - metrics["submit_times"][rid]
            completed += 1
    metrics.update({
        "ttft": ttft,
        "tpot": tpot,
        "e2e": e2e,
        "total_tokens": total_tokens,
        "completed": completed
    })

def run_naive_baseline(model, tokenizer, requests, device, metrics):
    ttft, tpot, e2e = {}, {}, {}
    token_times, total_tokens, completed = {}, 0, 0
    for req in requests:
        rid = req.request_id[:8]
        metrics["submit_times"][rid] = time.time()
        input_ids = torch.tensor(req.prompt_token_ids, dtype=torch.long, device=device).unsqueeze(0)
        attn_mask = torch.ones_like(input_ids)
        pos_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, position_ids=pos_ids)
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).item()
        ttft[rid] = time.time() - metrics["submit_times"][rid]
        req.append_output_token(next_token)
        token_times[rid] = [time.time()]
        for _ in range(req.max_tokens - 1):
            input_ids = torch.tensor(req.prompt_token_ids + req.output_token_ids, dtype=torch.long, device=device).unsqueeze(0)
            attn_mask = torch.ones_like(input_ids)
            pos_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask, position_ids=pos_ids)
                logits = out.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).item()
            req.append_output_token(next_token)
            token_times[rid].append(time.time())
            if next_token == tokenizer.eos_token_id:
                break
        e2e[rid] = time.time() - metrics["submit_times"][rid]
        total_tokens += len(req.output_token_ids)
        completed += 1
        if len(token_times[rid]) > 1:
            tpot[rid] = sum(t2 - t1 for t1, t2 in zip(token_times[rid][:-1], token_times[rid][1:])) / (len(token_times[rid]) - 1)
        else:
            tpot[rid] = None
    metrics.update({
        "ttft": ttft,
        "tpot": tpot,
        "e2e": e2e,
        "total_tokens": total_tokens,
        "completed": completed
    })

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    kv_cache = KVCacheManager(
        num_blocks=32, block_size=16, num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_attention_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        device=device,
    )

    prompt_lengths = [8, 32]
    batch_sizes = [1, 4, 8]
    max_tokens_list = [8, 16]
    max_steps = 80

    results = []

    for prompt_len in prompt_lengths:
        for batch_size in batch_sizes:
            for max_tokens in max_tokens_list:
                engine = EngineCore(model, tokenizer, kv_cache, max_batch_size=batch_size, device=device)
                prompts = generate_prompts(batch_size, min_len=prompt_len, max_len=prompt_len)
                requests = [Request(encode(p, tokenizer), max_tokens=max_tokens) for p in prompts]
                rid_map = {r.request_id[:8]: i for i, r in enumerate(requests)}
                submit_times = {r.request_id[:8]: time.time() for r in requests}
                metrics = {"submit_times": submit_times, "rid_map": rid_map}
                run_engine(engine, requests, metrics, max_steps=max_steps)

                naive_requests = [Request(encode(p, tokenizer), max_tokens=max_tokens) for p in prompts]
                naive_metrics = {
                    "submit_times": {r.request_id[:8]: time.time() for r in naive_requests},
                    "rid_map": {r.request_id[:8]: i for i, r in enumerate(naive_requests)}
                }
                run_naive_baseline(model, tokenizer, naive_requests, device, naive_metrics)

                results.append({
                    "prompt_len": prompt_len,
                    "batch_size": batch_size,
                    "max_tokens": max_tokens,
                    "optimized_tokens": metrics["total_tokens"],
                    "naive_tokens": naive_metrics["total_tokens"],
                    "optimized_e2e": np.mean(list(metrics["e2e"].values())),
                    "naive_e2e": np.mean(list(naive_metrics["e2e"].values())),
                    "speedup": metrics["total_tokens"] / max(naive_metrics["total_tokens"], 1)
                })

    # Print all results as a table
    print("\nBenchmark Results (per configuration):")
    print("| PromptLen | BatchSize | MaxTokens | OptTokens | NaiveTokens | Speedup | OptE2E(s) | NaiveE2E(s) |")
    for r in results:
        print(f"| {r['prompt_len']:>9} | {r['batch_size']:>9} | {r['max_tokens']:>9} |"
              f" {r['optimized_tokens']:>9} | {r['naive_tokens']:>11} |"
              f" {r['speedup']:>6.2f} | {r['optimized_e2e']:.3f} | {r['naive_e2e']:.3f} |")

    # Plot throughput and speedup for each parameter
    os.makedirs("plots", exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for i, prompt_len in enumerate(prompt_lengths):
        for j, batch_size in enumerate(batch_sizes):
            subset = [r for r in results if r["prompt_len"] == prompt_len and r["batch_size"] == batch_size]
            x = [r["max_tokens"] for r in subset]
            y_opt = [r["optimized_tokens"] for r in subset]
            y_naive = [r["naive_tokens"] for r in subset]
            y_speedup = [r["speedup"] for r in subset]
            axs[i, 0].plot(x, y_opt, label=f"Batch {batch_size}")
            axs[i, 0].plot(x, y_naive, linestyle="--", label=f"Naive {batch_size}")
            axs[i, 1].plot(x, y_speedup, label=f"Batch {batch_size}")
        axs[i, 0].set_title(f"Throughput (Prompt len={prompt_len})")
        axs[i, 0].set_xlabel("Max tokens")
        axs[i, 0].set_ylabel("Total tokens")
        axs[i, 0].legend()
        axs[i, 1].set_title(f"Speedup (Prompt len={prompt_len})")
        axs[i, 1].set_xlabel("Max tokens")
        axs[i, 1].set_ylabel("Speedup (Optimized/Naive)")
        axs[i, 1].legend()
    plt.tight_layout()
    plt.savefig("plots/benchmark_grid.png")
    print("Saved benchmark plots to plots/benchmark_grid.png")

if __name__ == "__main__":
    main()
