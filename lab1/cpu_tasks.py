# cpu_tasks.py
import math
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from time import perf_counter


def _time_it(fn, *args, **kwargs):
    # Допоміжна функція для вимірювання часу виконання
    t0 = perf_counter()
    res = fn(*args, **kwargs)
    return perf_counter() - t0, res


def pi_monte_carlo(iters: int, seed: int) -> float:
    # Обчислюю π методом Монте-Карло
    rnd = random.Random(seed)
    inside = 0
    for _ in range(iters):
        x = rnd.random()
        y = rnd.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / iters


def factorize(n: int) -> list[int]:
    # Факторизація через ділення
    factors = []
    x = n
    d = 2
    while d * d <= x:
        while x % d == 0:
            factors.append(d)
            x //= d
        d = 3 if d == 2 else d + 2
    if x > 1:
        factors.append(x)
    return factors


def primes_in_range(a: int, b: int) -> int:
    # Рахує кількість простих у діапазоні [a, b]
    def is_prime(k: int) -> bool:
        if k < 2:
            return False
        if k in (2, 3):
            return True
        if k % 2 == 0:
            return False
        r = int(math.isqrt(k))
        f = 3
        while f <= r:
            if k % f == 0:
                return False
            f += 2
        return True

    cnt = 0
    for x in range(a, b + 1):
        if is_prime(x):
            cnt += 1
    return cnt


def _run_sequential(tasks):
    # Послідовне виконання списку незалежних задач
    total = 0.0
    out = []
    for fn, args in tasks:
        dt, res = _time_it(fn, *args)
        total += dt
        out.append(res)
    return total, out


def _run_threaded(tasks, workers: int):
    # Паралельне виконання через пул потоків
    t0 = perf_counter()
    out = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {}
        for i, (fn, args) in enumerate(tasks):
            fut = ex.submit(fn, *args)
            fut_map[fut] = i

        for fut in as_completed(fut_map):
            i = fut_map[fut]
            out[i] = fut.result()

    return perf_counter() - t0, out


def _run_processed(tasks, workers: int):
    # Паралельне виконання через пул процесів
    t0 = perf_counter()
    out = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_map = {}
        for i, (fn, args) in enumerate(tasks):
            fut = ex.submit(fn, *args)
            fut_map[fut] = i

        for fut in as_completed(fut_map):
            i = fut_map[fut]
            out[i] = fut.result()

    return perf_counter() - t0, out


def cpu_demo(threads_list: list[int], return_results: bool = False):
    print("\n--- CPU-bound задачі ---")

    pi_tasks = [
        (pi_monte_carlo, (400_000, 1)),
        (pi_monte_carlo, (700_000, 2)),
        (pi_monte_carlo, (1_000_000, 3)),
        (pi_monte_carlo, (1_300_000, 4)),
    ]

    fac_tasks = [
        (factorize, (999_999_937,)),
        (factorize, (999_999_929,)),
        (factorize, (999_999_893,)),
        (factorize, (999_999_883,)),
    ]

    prime_tasks = [
        (primes_in_range, (200_000, 220_000)),
        (primes_in_range, (220_001, 240_000)),
        (primes_in_range, (240_001, 260_000)),
        (primes_in_range, (260_001, 280_000)),
    ]

    groups = [
        ("pi_monte_carlo", "π (Монте-Карло)", pi_tasks),
        ("factorization", "Факторизація", fac_tasks),
        ("primes", "Прості числа", prime_tasks),
    ]

    results = {"cpu": {}}

    for key, title, tasks in groups:
        print(f"\nЗадача: {title}")

        dt_seq, _ = _run_sequential(tasks)
        print(f"Послідовно: час, c: {dt_seq:.4f}")

        results["cpu"][key] = {
            "title": title,
            "seq": float(dt_seq),
            "threads": {},
            "procs": {},
        }

        for w in threads_list:
            if w <= 1:
                continue

            dt_thr, _ = _run_threaded(tasks, workers=w)
            print(f"Паралельно (потоків={w}): час, c: {dt_thr:.4f}")
            results["cpu"][key]["threads"][int(w)] = float(dt_thr)

            dt_proc, _ = _run_processed(tasks, workers=w)
            print(f"Паралельно (процесів={w}): час, c: {dt_proc:.4f}")
            results["cpu"][key]["procs"][int(w)] = float(dt_proc)

    if return_results:
        return results