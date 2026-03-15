from __future__ import annotations

import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Iterable

import numpy as np

TAG_RE = re.compile(r"<\s*([a-zA-Z][a-zA-Z0-9]*)\b")


def _time_it(fn, *args, **kwargs):
    t0 = perf_counter()
    res = fn(*args, **kwargs)
    return perf_counter() - t0, res


def _chunk_ranges(n: int, chunks: int) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    if chunks <= 0:
        chunks = 1
    step = math.ceil(n / chunks)
    out = []
    for i in range(0, n, step):
        out.append((i, min(i + step, n)))
    return out


def _split_list(items: list, threshold: int) -> list[list]:
    if len(items) <= threshold:
        return [items]
    mid = len(items) // 2
    return _split_list(items[:mid], threshold) + _split_list(items[mid:], threshold)


def _split_array(arr: np.ndarray, threshold: int) -> list[np.ndarray]:
    if len(arr) <= threshold:
        return [arr]
    mid = len(arr) // 2
    return _split_array(arr[:mid], threshold) + _split_array(arr[mid:], threshold)


def _split_ranges(start: int, end: int, threshold: int) -> list[tuple[int, int]]:
    if end - start <= threshold:
        return [(start, end)]
    mid = (start + end) // 2
    return _split_ranges(start, mid, threshold) + _split_ranges(mid, end, threshold)



def _results_equal(left, right, *, atol: float = 1e-6, rtol: float = 1e-9) -> bool:
    if isinstance(left, Counter) and isinstance(right, Counter):
        return left == right

    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.allclose(left, right, atol=atol, rtol=rtol)

    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        for key in left:
            lv = left[key]
            rv = right[key]

            if isinstance(lv, (int, float, np.integer, np.floating)) or isinstance(
                rv, (int, float, np.integer, np.floating)
            ):
                if not np.isclose(float(lv), float(rv), atol=atol, rtol=rtol):
                    return False
            else:
                if lv != rv:
                    return False
        return True

    return left == right


def _pick_best(seq_time: float, variants: dict[int, float]) -> dict:
    candidates = [("sequential", float(seq_time))]
    for workers, value in variants.items():
        candidates.append((str(workers), float(value)))
    best_label, best_time = min(candidates, key=lambda x: x[1])
    best_speedup = float(seq_time / best_time) if best_time > 0 else None
    return {
        "label": best_label,
        "time": best_time,
        "speedup_vs_sequential": best_speedup,
    }


# HTML TAGS

def _count_tags_in_file(path: str) -> Counter:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return Counter(tag.lower() for tag in TAG_RE.findall(text))


def html_sequential(paths: list[str]) -> Counter:
    total = Counter()
    for p in paths:
        total.update(_count_tags_in_file(p))
    return total


def html_map_reduce(paths: list[str], workers: int) -> Counter:
    if workers <= 1:
        return html_sequential(paths)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        mapped = list(ex.map(_count_tags_in_file, paths))

    total = Counter()
    for part in mapped:
        total.update(part)
    return total


def html_worker_pool(paths: list[str], workers: int) -> Counter:
    if workers <= 1:
        return html_sequential(paths)

    total = Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_count_tags_in_file, p) for p in paths]
        for fut in as_completed(futures):
            total.update(fut.result())
    return total


def html_fork_join(paths: list[str], workers: int, threshold: int = 32) -> Counter:
    if workers <= 1 or len(paths) <= threshold:
        return html_sequential(paths)

    chunks = _split_list(paths, threshold)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        partials = list(ex.map(html_sequential, chunks))

    total = Counter()
    for part in partials:
        total.update(part)
    return total


# ARRAY STATS

def _stats_chunk(arr: np.ndarray) -> dict:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
        "count": int(arr.size),
        "values": arr,
    }


def _merge_stats(parts: Iterable[dict]) -> dict:
    parts = list(parts)
    min_v = min(p["min"] for p in parts)
    max_v = max(p["max"] for p in parts)
    total_sum = sum(p["sum"] for p in parts)
    total_count = sum(p["count"] for p in parts)
    merged = np.concatenate([p["values"] for p in parts])

    return {
        "min": float(min_v),
        "max": float(max_v),
        "mean": float(total_sum / total_count),
        "median": float(np.median(merged)),
    }


def array_sequential(arr: np.ndarray) -> dict:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
    }


def array_map_reduce(arr: np.ndarray, workers: int) -> dict:
    if workers <= 1:
        return array_sequential(arr)

    ranges = _chunk_ranges(len(arr), workers)
    chunks = [arr[a:b] for a, b in ranges]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        mapped = list(ex.map(_stats_chunk, chunks))

    return _merge_stats(mapped)


def array_worker_pool(arr: np.ndarray, workers: int) -> dict:
    if workers <= 1:
        return array_sequential(arr)

    ranges = _chunk_ranges(len(arr), workers)
    chunks = [arr[a:b] for a, b in ranges]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_stats_chunk, chunk) for chunk in chunks]
        parts = [f.result() for f in futures]

    return _merge_stats(parts)


def array_fork_join(arr: np.ndarray, workers: int, threshold: int = 100_000) -> dict:
    if workers <= 1 or len(arr) <= threshold:
        return array_sequential(arr)

    chunks = _split_array(arr, threshold)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_stats_chunk, chunks))

    return _merge_stats(parts)


# MATRICES

def _matmul_rows(a: np.ndarray, b: np.ndarray, start: int, end: int) -> tuple[int, np.ndarray]:
    return start, a[start:end] @ b


def matrix_sequential(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def matrix_map_reduce(a: np.ndarray, b: np.ndarray, workers: int) -> np.ndarray:
    if workers <= 1:
        return matrix_sequential(a, b)

    ranges = _chunk_ranges(a.shape[0], workers)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        mapped = list(ex.map(lambda rg: _matmul_rows(a, b, rg[0], rg[1]), ranges))

    out = np.empty((a.shape[0], b.shape[1]), dtype=np.float64)
    for start, block in mapped:
        out[start:start + block.shape[0], :] = block
    return out


def matrix_worker_pool(a: np.ndarray, b: np.ndarray, workers: int) -> np.ndarray:
    if workers <= 1:
        return matrix_sequential(a, b)

    ranges = _chunk_ranges(a.shape[0], workers)
    out = np.empty((a.shape[0], b.shape[1]), dtype=np.float64)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_matmul_rows, a, b, s, e) for s, e in ranges]
        for fut in as_completed(futures):
            start, block = fut.result()
            out[start:start + block.shape[0], :] = block

    return out


def matrix_fork_join(a: np.ndarray, b: np.ndarray, workers: int, threshold_rows: int = 64) -> np.ndarray:
    if workers <= 1 or a.shape[0] <= threshold_rows:
        return matrix_sequential(a, b)

    ranges = _split_ranges(0, a.shape[0], threshold_rows)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        partials = list(ex.map(lambda rg: _matmul_rows(a, b, rg[0], rg[1]), ranges))

    out = np.empty((a.shape[0], b.shape[1]), dtype=np.float64)
    for start, block in partials:
        out[start:start + block.shape[0], :] = block
    return out


def compare_patterns(
    html_paths: list[str],
    array_data: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    workers_list: list[int],
    return_results: bool = False,
):

    results = {
        "patterns": {
            "html_tags": {
                "title": "HTML-теги",
                "seq": None,
                "map_reduce": {},
                "fork_join": {},
                "worker_pool": {},
                "validation": {},
                "best": {},
            },
            "array_stats": {
                "title": "Статистика масиву",
                "seq": None,
                "map_reduce": {},
                "fork_join": {},
                "worker_pool": {},
                "validation": {},
                "best": {},
            },
            "matrix_multiply": {
                "title": "Множення матриць",
                "seq": None,
                "map_reduce": {},
                "fork_join": {},
                "worker_pool": {},
                "validation": {},
                "best": {},
            },
        },
        "best_overall": {},
    }

    groups = [
        (
            "html_tags",
            "HTML-теги",
            lambda: html_sequential(html_paths),
            lambda w: html_map_reduce(html_paths, w),
            lambda w: html_fork_join(html_paths, w),
            lambda w: html_worker_pool(html_paths, w),
        ),
        (
            "array_stats",
            "Статистика масиву",
            lambda: array_sequential(array_data),
            lambda w: array_map_reduce(array_data, w),
            lambda w: array_fork_join(array_data, w),
            lambda w: array_worker_pool(array_data, w),
        ),
        (
            "matrix_multiply",
            "Множення матриць",
            lambda: matrix_sequential(a, b),
            lambda w: matrix_map_reduce(a, b, w),
            lambda w: matrix_fork_join(a, b, w),
            lambda w: matrix_worker_pool(a, b, w),
        ),
    ]

    for key, title, seq_fn, mr_fn, fj_fn, wp_fn in groups:
        print(f"\nЗадача: {title}")

        dt_seq, seq_res = _time_it(seq_fn)
        print(f"Послідовно: час, c: {dt_seq:.4f}")
        results["patterns"][key]["seq"] = float(dt_seq)

        validations = {
            "map_reduce": True,
            "fork_join": True,
            "worker_pool": True,
        }

        for w in workers_list:
            if w <= 1:
                continue

            dt_mr, mr_res = _time_it(mr_fn, w)
            ok_mr = _results_equal(seq_res, mr_res)
            validations["map_reduce"] = validations["map_reduce"] and ok_mr
            print(f"Map-Reduce ({w}): час, c: {dt_mr:.4f}; збіг: {'так' if ok_mr else 'ні'}")
            results["patterns"][key]["map_reduce"][int(w)] = float(dt_mr)

            dt_fj, fj_res = _time_it(fj_fn, w)
            ok_fj = _results_equal(seq_res, fj_res)
            validations["fork_join"] = validations["fork_join"] and ok_fj
            print(f"Fork-Join ({w}): час, c: {dt_fj:.4f}; збіг: {'так' if ok_fj else 'ні'}")
            results["patterns"][key]["fork_join"][int(w)] = float(dt_fj)

            dt_wp, wp_res = _time_it(wp_fn, w)
            ok_wp = _results_equal(seq_res, wp_res)
            validations["worker_pool"] = validations["worker_pool"] and ok_wp
            print(f"Worker Pool ({w}): час, c: {dt_wp:.4f}; збіг: {'так' if ok_wp else 'ні'}")
            results["patterns"][key]["worker_pool"][int(w)] = float(dt_wp)

        results["patterns"][key]["validation"] = validations

        best_map_reduce = _pick_best(dt_seq, results["patterns"][key]["map_reduce"])
        best_fork_join = _pick_best(dt_seq, results["patterns"][key]["fork_join"])
        best_worker_pool = _pick_best(dt_seq, results["patterns"][key]["worker_pool"])

        family_best = min(
            [
                ("map_reduce", best_map_reduce),
                ("fork_join", best_fork_join),
                ("worker_pool", best_worker_pool),
            ],
            key=lambda x: x[1]["time"],
        )

        results["patterns"][key]["best"] = {
            "pattern": family_best[0],
            "configuration": family_best[1]["label"],
            "time": family_best[1]["time"],
            "speedup_vs_sequential": family_best[1]["speedup_vs_sequential"],
        }

        best_text = results["patterns"][key]["best"]
        print(
            f"Найкращий підхід: {best_text['pattern']} ({best_text['configuration']}); "
            f"час, c: {best_text['time']:.4f}; "
            f"прискорення: {best_text['speedup_vs_sequential']:.4f}"
        )

    overall_candidates = []
    for key, item in results["patterns"].items():
        overall_candidates.append((key, item["best"]))

    best_overall_key, best_overall_value = min(overall_candidates, key=lambda x: x[1]["time"])
    results["best_overall"] = {
        "task": best_overall_key,
        "title": results["patterns"][best_overall_key]["title"],
        **best_overall_value,
    }

    print(
        "\nНайкращий підхід серед задач цього блоку: "
        f"{results['best_overall']['title']} -> "
        f"{results['best_overall']['pattern']} ({results['best_overall']['configuration']})"
    )

    if return_results:
        return results
