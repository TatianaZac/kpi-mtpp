from __future__ import annotations

import numpy as np
import csv
from collections import defaultdict
from queue import Queue
from threading import Lock, Thread
from time import perf_counter


def _time_it(fn, *args, **kwargs):
    t0 = perf_counter()
    res = fn(*args, **kwargs)
    return perf_counter() - t0, res


def _results_equal(left: dict, right: dict, *, atol: float = 1e-3, rtol: float = 1e-9) -> bool:
    if left.get("rows") != right.get("rows"):
        return False

    if not np.isclose(
        float(left.get("grand_total", 0.0)),
        float(right.get("grand_total", 0.0)),
        atol=atol,
        rtol=rtol,
    ):
        return False

    lt = left.get("totals", {})
    rt = right.get("totals", {})
    if lt.keys() != rt.keys():
        return False

    for key in lt:
        if not np.isclose(float(lt[key]), float(rt[key]), atol=atol, rtol=rtol):
            return False

    return True


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


def _convert_to_uah(tx: dict) -> dict:
    tx = dict(tx)
    amount = float(tx["amount"])
    rate = float(tx["rate_hint"])
    tx["amount_uah"] = amount * rate
    return tx


def _apply_cashback(tx: dict) -> dict:
    tx = dict(tx)
    amount_uah = float(tx["amount_uah"])
    status = tx["status_flag"]
    cashback_rate = 0.0
    if status == "vip":
        cashback_rate = 0.10
    elif status == "premium":
        cashback_rate = 0.20
    tx["cashback"] = amount_uah * cashback_rate
    tx["final_amount"] = amount_uah - tx["cashback"]
    return tx


def read_transactions(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _chunked_transactions(path: str, chunk_size: int):
    chunk = []
    for tx in read_transactions(path):
        chunk.append(tx)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def transactions_sequential(path: str) -> dict:
    totals = defaultdict(float)
    count = 0
    for tx in read_transactions(path):
        tx = _convert_to_uah(tx)
        tx = _apply_cashback(tx)
        totals[tx["product_type"]] += tx["final_amount"]
        count += 1
    return {
        "rows": count,
        "totals": dict(totals),
        "grand_total": float(sum(totals.values())),
    }


def transactions_pipeline(path: str, workers: int, chunk_size: int = 2000) -> dict:
    q1: Queue = Queue(maxsize=max(4, workers * 2))
    q2: Queue = Queue(maxsize=max(4, workers * 2))
    q3: Queue = Queue(maxsize=max(4, workers * 2))
    sentinel = object()

    totals = defaultdict(float)
    totals_lock = Lock()
    processed_rows = {"count": 0}

    def producer():
        for chunk in _chunked_transactions(path, chunk_size):
            q1.put(chunk)
        for _ in range(workers):
            q1.put(sentinel)

    def converter():
        while True:
            item = q1.get()
            if item is sentinel:
                q2.put(sentinel)
                q1.task_done()
                break
            out = [_convert_to_uah(tx) for tx in item]
            q2.put(out)
            q1.task_done()

    def cashbacker():
        while True:
            item = q2.get()
            if item is sentinel:
                q3.put(sentinel)
                q2.task_done()
                break
            out = [_apply_cashback(tx) for tx in item]
            q3.put(out)
            q2.task_done()

    def aggregator():
        stopped = 0
        while True:
            item = q3.get()
            if item is sentinel:
                stopped += 1
                q3.task_done()
                if stopped == workers:
                    break
                continue

            local = defaultdict(float)
            local_count = 0
            for tx in item:
                local[tx["product_type"]] += tx["final_amount"]
                local_count += 1

            with totals_lock:
                for k, v in local.items():
                    totals[k] += v
                processed_rows["count"] += local_count

            q3.task_done()

    threads = [Thread(target=producer)]
    threads += [Thread(target=converter) for _ in range(workers)]
    threads += [Thread(target=cashbacker) for _ in range(workers)]
    threads += [Thread(target=aggregator)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return {
        "rows": processed_rows["count"],
        "totals": dict(totals),
        "grand_total": float(sum(totals.values())),
    }


def _process_chunk(rows: list[dict]) -> tuple[dict, int]:
    local = defaultdict(float)
    count = 0
    for tx in rows:
        tx = _convert_to_uah(tx)
        tx = _apply_cashback(tx)
        local[tx["product_type"]] += tx["final_amount"]
        count += 1
    return dict(local), count


def transactions_producer_consumer(path: str, workers: int, chunk_size: int = 2000) -> dict:
    task_queue: Queue = Queue(maxsize=max(4, workers * 2))
    sentinel = object()

    totals = defaultdict(float)
    totals_lock = Lock()
    processed_rows = {"count": 0}

    def producer():
        for chunk in _chunked_transactions(path, chunk_size):
            task_queue.put(chunk)
        for _ in range(workers):
            task_queue.put(sentinel)

    def consumer():
        while True:
            item = task_queue.get()
            if item is sentinel:
                task_queue.task_done()
                break

            local, cnt = _process_chunk(item)

            with totals_lock:
                for k, v in local.items():
                    totals[k] += v
                processed_rows["count"] += cnt

            task_queue.task_done()

    threads = [Thread(target=producer)]
    threads += [Thread(target=consumer) for _ in range(workers)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return {
        "rows": processed_rows["count"],
        "totals": dict(totals),
        "grand_total": float(sum(totals.values())),
    }


def compare_transaction_patterns(path: str, workers_list: list[int], return_results: bool = False):
    print("\nPipeline / Producer-Consumer (транзакції)")
    results = {
        "transactions": {
            "title": "Фінансові транзакції",
            "seq": None,
            "pipeline": {},
            "producer_consumer": {},
            "validation": {},
            "best": {},
        }
    }

    dt_seq, seq_res = _time_it(transactions_sequential, path)
    print(f"Послідовно: час, c: {dt_seq:.4f}")
    results["transactions"]["seq"] = float(dt_seq)

    valid_pipeline = True
    valid_pc = True

    for w in workers_list:
        if w <= 1:
            continue

        dt_pipe, pipe_res = _time_it(transactions_pipeline, path, w)
        ok_pipe = _results_equal(seq_res, pipe_res)
        valid_pipeline = valid_pipeline and ok_pipe
        print(f"Pipeline ({w}): час, c: {dt_pipe:.4f}; збіг: {'так' if ok_pipe else 'ні'}")
        results["transactions"]["pipeline"][int(w)] = float(dt_pipe)

        dt_pc, pc_res = _time_it(transactions_producer_consumer, path, w)
        ok_pc = _results_equal(seq_res, pc_res)
        valid_pc = valid_pc and ok_pc
        print(f"Producer-Consumer ({w}): час, c: {dt_pc:.4f}; збіг: {'так' if ok_pc else 'ні'}")
        results["transactions"]["producer_consumer"][int(w)] = float(dt_pc)

    results["transactions"]["validation"] = {
        "pipeline": valid_pipeline,
        "producer_consumer": valid_pc,
    }

    best_pipeline = _pick_best(dt_seq, results["transactions"]["pipeline"])
    best_pc = _pick_best(dt_seq, results["transactions"]["producer_consumer"])

    family_best = min(
        [
            ("pipeline", best_pipeline),
            ("producer_consumer", best_pc),
        ],
        key=lambda x: x[1]["time"],
    )

    results["transactions"]["best"] = {
        "pattern": family_best[0],
        "configuration": family_best[1]["label"],
        "time": family_best[1]["time"],
        "speedup_vs_sequential": family_best[1]["speedup_vs_sequential"],
    }

    best = results["transactions"]["best"]
    print(
        f"Найкращий підхід: {best['pattern']} ({best['configuration']}); "
        f"час, c: {best['time']:.4f}; прискорення: {best['speedup_vs_sequential']:.4f}"
    )

    if return_results:
        return results
