from typing import Any, Callable, Dict, Optional, Tuple

import psutil
import subprocess
import time

_proc = psutil.Process()


def get_gpu_usage() -> Tuple[Optional[float], Optional[float]]:
    """Return (used_mb, total_mb) for the first GPU, or (None, None) if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        used, total = map(float, out.strip().split(","))
        return used, total
    except Exception:
        return None, None


def get_ram_mb() -> float:
    """Return resident memory (MB) for this Python process."""
    return _proc.memory_info().rss / (1024 * 1024)


def measure_retrieval(retriever: Any, query: str) -> Dict[str, Any]:
    """Measure a retriever.invoke call; returns docs plus latency info."""

    start = time.perf_counter()
    docs = retriever.invoke(query)
    duration_s = time.perf_counter() - start

    return {
        "docs": docs,
        "duration_s": duration_s,
        "doc_count": len(docs) if docs is not None else 0,
    }


def measure_call(call_fn: Callable[[], Any]) -> Dict[str, Any]:
    """Measure a callable: runtime, RAM delta, VRAM delta, and token stats if present.

    call_fn should return a LangChain/Ollama response (or any object) possibly
    with `response_metadata` exposing Ollama stats.
    """

    ram_before = get_ram_mb()
    vram_before, vram_total = get_gpu_usage()
    start = time.perf_counter()

    result = call_fn()

    duration_s = time.perf_counter() - start
    ram_after = get_ram_mb()
    vram_after, _ = get_gpu_usage()

    meta = getattr(result, "response_metadata", {}) or {}
    prompt_tokens = meta.get("prompt_eval_count")
    gen_tokens = meta.get("eval_count")
    gen_duration_ns = meta.get("eval_duration")
    tok_per_s = None
    if gen_tokens and gen_duration_ns:
        tok_per_s = gen_tokens / (gen_duration_ns / 1e9)

    return {
        "result": result,
        "duration_s": duration_s,
        "ram_before_mb": ram_before,
        "ram_after_mb": ram_after,
        "ram_delta_mb": ram_after - ram_before,
        "vram_before_mb": vram_before,
        "vram_after_mb": vram_after,
        "vram_delta_mb": None if vram_before is None or vram_after is None else vram_after - vram_before,
        "vram_total_mb": vram_total,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "gen_duration_ns": gen_duration_ns,
        "tok_per_s": tok_per_s,
    }


def print_stats(stats: Dict[str, Any]) -> None:
    """Pretty-print stats dictionary produced by measure_call."""

    print("--- Stats ---")
    print(f"Wall time:        {stats['duration_s']:.3f} s")
    print(f"RAM delta:        {stats['ram_delta_mb']:.1f} MB (after: {stats['ram_after_mb']:.1f} MB)")

    if stats.get("vram_before_mb") is not None and stats.get("vram_after_mb") is not None:
        print(
            "VRAM delta:       "
            f"{stats['vram_delta_mb']:.1f} MB (after: {stats['vram_after_mb']:.1f} / {stats['vram_total_mb']:.0f} MB)"
        )

    if stats.get("prompt_tokens") is not None:
        print(f"Prompt tokens:    {stats['prompt_tokens']}")
    if stats.get("gen_tokens") is not None:
        print(f"Generated tokens: {stats['gen_tokens']}")
    if stats.get("gen_duration_ns") is not None:
        print(f"Gen duration:     {stats['gen_duration_ns'] / 1e9:.3f} s")
    if stats.get("tok_per_s") is not None:
        print(f"Speed:            {stats['tok_per_s']:.2f} tokens/s")


def print_retrieval_stats(stats: Dict[str, Any]) -> None:
    """Pretty-print retrieval stats produced by measure_retrieval."""

    print("--- Retrieval ---")
    print(f"Latency:          {stats['duration_s']:.3f} s")
    print(f"Docs returned:    {stats['doc_count']}")