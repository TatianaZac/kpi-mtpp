from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from utils import DEFAULT_SEED, make_rng


@dataclass
class Account:
    idx: int
    balance: int
    lock: threading.Lock


class UnsafeAccount:
    def __init__(self, idx: int, balance: int):
        self.idx = idx
        self.balance = balance


@dataclass
class TransferStats:
    elapsed: float
    initial_total: int
    final_total: int
    transferred_attempts: int
    negatives_detected: int
    successful_ops: int
    total_preserved: bool



def create_unsafe_accounts(count: int = 128, seed: int = DEFAULT_SEED) -> list[UnsafeAccount]:
    rng = make_rng(seed)
    return [UnsafeAccount(i, rng.randint(1_000, 10_000)) for i in range(count)]



def create_safe_accounts(count: int = 128, seed: int = DEFAULT_SEED) -> list[Account]:
    rng = make_rng(seed)
    return [Account(i, rng.randint(1_000, 10_000), threading.Lock()) for i in range(count)]



def total_money(accounts) -> int:
    return sum(acc.balance for acc in accounts)



def _unsafe_transfer(src: UnsafeAccount, dst: UnsafeAccount, amount: int) -> bool:
    if src.balance < amount:
        return False
    src_old = src.balance
    time.sleep(0.00001)
    src.balance = src_old - amount
    dst_old = dst.balance
    time.sleep(0.00001)
    dst.balance = dst_old + amount
    return True



def _safe_transfer_ordered(src: Account, dst: Account, amount: int) -> bool:
    if src.idx == dst.idx:
        return False
    first, second = (src, dst) if src.idx < dst.idx else (dst, src)
    with first.lock:
        with second.lock:
            if src.balance < amount:
                return False
            src.balance -= amount
            dst.balance += amount
            return True



def _safe_transfer_single_global(src: Account, dst: Account, amount: int, guard: threading.Lock) -> bool:
    with guard:
        if src.idx == dst.idx or src.balance < amount:
            return False
        src.balance -= amount
        dst.balance += amount
        return True



def sequential_transfers(accounts_count: int = 128, operations: int = 50_000, seed: int = DEFAULT_SEED) -> dict:
    rng = make_rng(seed)
    accounts = create_safe_accounts(accounts_count, seed)
    initial = total_money(accounts)
    started = perf_counter()
    success = 0

    for _ in range(operations):
        src_idx = rng.randrange(accounts_count)
        dst_idx = rng.randrange(accounts_count)
        if src_idx == dst_idx:
            continue
        amount = rng.randint(1, 200)
        src = accounts[src_idx]
        dst = accounts[dst_idx]
        if src.balance >= amount:
            src.balance -= amount
            dst.balance += amount
            success += 1

    elapsed = perf_counter() - started
    final_total = total_money(accounts)
    return {
        "title": "Послідовні перекази між рахунками",
        "time": elapsed,
        "initial_total": initial,
        "final_total": final_total,
        "successful_ops": success,
        "total_preserved": initial == final_total,
    }



def race_condition_demo(accounts_count: int = 128, operations: int = 80_000, workers: int = 200, seed: int = DEFAULT_SEED) -> TransferStats:
    accounts = create_unsafe_accounts(accounts_count, seed)
    initial = total_money(accounts)
    negatives_detected = 0
    success = 0
    attempts = 0
    attempts_lock = threading.Lock()
    success_lock = threading.Lock()
    neg_lock = threading.Lock()

    per_worker = max(1, operations // workers)

    def worker(worker_id: int) -> None:
        nonlocal attempts, success, negatives_detected
        rng = make_rng(seed + 10_000 + worker_id)
        local_success = 0
        local_attempts = 0
        local_negatives = 0
        for _ in range(per_worker):
            src_idx = rng.randrange(accounts_count)
            dst_idx = rng.randrange(accounts_count)
            if src_idx == dst_idx:
                continue
            amount = rng.randint(1, 200)
            local_attempts += 1
            done = _unsafe_transfer(accounts[src_idx], accounts[dst_idx], amount)
            if done:
                local_success += 1
            if accounts[src_idx].balance < 0 or accounts[dst_idx].balance < 0:
                local_negatives += 1
        with attempts_lock:
            attempts += local_attempts
        with success_lock:
            success += local_success
        with neg_lock:
            negatives_detected += local_negatives

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(workers):
            ex.submit(worker, i)
    elapsed = perf_counter() - started
    final_total = total_money(accounts)

    return TransferStats(
        elapsed=elapsed,
        initial_total=initial,
        final_total=final_total,
        transferred_attempts=attempts,
        negatives_detected=negatives_detected,
        successful_ops=success,
        total_preserved=(initial == final_total),
    )



def safe_transfers_demo(accounts_count: int = 128, operations: int = 80_000, workers: int = 200, seed: int = DEFAULT_SEED) -> TransferStats:
    accounts = create_safe_accounts(accounts_count, seed)
    initial = total_money(accounts)
    attempts = 0
    success = 0
    attempts_lock = threading.Lock()
    success_lock = threading.Lock()
    per_worker = max(1, operations // workers)

    def worker(worker_id: int) -> None:
        nonlocal attempts, success
        rng = make_rng(seed + 20_000 + worker_id)
        local_attempts = 0
        local_success = 0
        for _ in range(per_worker):
            src_idx = rng.randrange(accounts_count)
            dst_idx = rng.randrange(accounts_count)
            if src_idx == dst_idx:
                continue
            amount = rng.randint(1, 200)
            local_attempts += 1
            if _safe_transfer_ordered(accounts[src_idx], accounts[dst_idx], amount):
                local_success += 1
        with attempts_lock:
            attempts += local_attempts
        with success_lock:
            success += local_success

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(workers):
            ex.submit(worker, i)
    elapsed = perf_counter() - started
    final_total = total_money(accounts)

    return TransferStats(
        elapsed=elapsed,
        initial_total=initial,
        final_total=final_total,
        transferred_attempts=attempts,
        negatives_detected=0,
        successful_ops=success,
        total_preserved=(initial == final_total),
    )



def deadlock_demo_unsafe(timeout: float = 1.0, sleep_inside: float = 0.2) -> dict:
    accounts = create_safe_accounts(2, DEFAULT_SEED)
    a = accounts[0]
    b = accounts[1]
    barrier = threading.Barrier(2)

    def transfer_with_inverted_order(src: Account, dst: Account):
        with src.lock:
            barrier.wait()
            time.sleep(sleep_inside)
            with dst.lock:
                if src.balance >= 10:
                    src.balance -= 10
                    dst.balance += 10

    t1 = threading.Thread(target=transfer_with_inverted_order, args=(a, b), daemon=True)
    t2 = threading.Thread(target=transfer_with_inverted_order, args=(b, a), daemon=True)

    started = perf_counter()
    t1.start()
    t2.start()
    t1.join(timeout)
    t2.join(timeout)
    elapsed = perf_counter() - started
    deadlock_detected = t1.is_alive() and t2.is_alive()

    return {
        "title": "Deadlock без впорядкування lock-ів",
        "time": elapsed,
        "deadlock_detected": deadlock_detected,
        "threads_alive_after_timeout": int(t1.is_alive()) + int(t2.is_alive()),
        "explanation": "Потік 1 утримує lock A і чекає lock B, потік 2 утримує lock B і чекає lock A.",
    }



def deadlock_demo_safe(iterations: int = 5_000) -> dict:
    accounts = create_safe_accounts(2, DEFAULT_SEED)
    a = accounts[0]
    b = accounts[1]

    def worker(src: Account, dst: Account, loops: int):
        for _ in range(loops):
            _safe_transfer_ordered(src, dst, 1)

    t1 = threading.Thread(target=worker, args=(a, b, iterations))
    t2 = threading.Thread(target=worker, args=(b, a, iterations))

    started = perf_counter()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = perf_counter() - started

    return {
        "title": "Deadlock усунено глобальним порядком lock-ів",
        "time": elapsed,
        "deadlock_detected": False,
        "threads_alive_after_timeout": 0,
        "final_total": total_money(accounts),
    }



def compare_banking(workers_list: list[int], accounts_count: int = 128, operations: int = 80_000, seed: int = DEFAULT_SEED, return_results: bool = False):
    print("\nЗадача 1. Переказ коштів між банківськими рахунками")
    print(f"Рахунків: {accounts_count}, операцій: {operations}")

    seq = sequential_transfers(accounts_count=accounts_count, operations=operations, seed=seed)
    print(
        f"Послідовно: {seq['time']:.4f} c | сума {seq['initial_total']} -> {seq['final_total']} | "
        f"збереження суми: {seq['total_preserved']}"
    )

    unsafe_series = {}
    safe_series = {}
    unsafe_checks = {}
    safe_checks = {}

    for workers in workers_list:
        race = race_condition_demo(accounts_count=accounts_count, operations=operations, workers=workers, seed=seed)
        unsafe_series[str(workers)] = race.elapsed
        unsafe_checks[str(workers)] = {
            "initial_total": race.initial_total,
            "final_total": race.final_total,
            "total_preserved": race.total_preserved,
            "negatives_detected": race.negatives_detected,
            "successful_ops": race.successful_ops,
            "attempts": race.transferred_attempts,
        }
        print(
            f"Race/unsafe | workers={workers:4d}: {race.elapsed:.4f} c | "
            f"сума {race.initial_total} -> {race.final_total} | збереження: {race.total_preserved}"
        )

        safe = safe_transfers_demo(accounts_count=accounts_count, operations=operations, workers=workers, seed=seed)
        safe_series[str(workers)] = safe.elapsed
        safe_checks[str(workers)] = {
            "initial_total": safe.initial_total,
            "final_total": safe.final_total,
            "total_preserved": safe.total_preserved,
            "successful_ops": safe.successful_ops,
            "attempts": safe.transferred_attempts,
        }
        print(
            f"Race fixed   | workers={workers:4d}: {safe.elapsed:.4f} c | "
            f"сума {safe.initial_total} -> {safe.final_total} | збереження: {safe.total_preserved}"
        )

    deadlock_unsafe = deadlock_demo_unsafe()
    deadlock_safe = deadlock_demo_safe()
    print(
        f"Deadlock unsafe: виявлено={deadlock_unsafe['deadlock_detected']}, "
        f"живих потоків після timeout={deadlock_unsafe['threads_alive_after_timeout']}"
    )
    print(
        f"Deadlock fixed : виявлено={deadlock_safe['deadlock_detected']}, "
        f"час={deadlock_safe['time']:.4f} c"
    )

    results = {
        "banking": {
            "title": "Переказ коштів між банківськими рахунками",
            "sequential": seq,
            "unsafe_race": unsafe_series,
            "safe_race": safe_series,
            "unsafe_checks": unsafe_checks,
            "safe_checks": safe_checks,
            "deadlock_unsafe": deadlock_unsafe,
            "deadlock_safe": deadlock_safe,
            "best_safe_workers": min(safe_series, key=safe_series.get) if safe_series else None,
            "best_safe_time": min(safe_series.values()) if safe_series else None,
        }
    }
    if return_results:
        return results
    return None
