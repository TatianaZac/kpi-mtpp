# io_task.py
import os
import random
import string
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter


def _time_it(fn, *args, **kwargs):
    # Допоміжна функція для вимірювання часу виконання
    t0 = perf_counter()
    res = fn(*args, **kwargs)
    return perf_counter() - t0, res


def generate_random_text(words: int, seed: int) -> str:
    # Генерує випадковий текст зі слів (для файлів)
    rnd = random.Random(seed)
    pool = string.ascii_lowercase
    out = []
    for _ in range(words):
        ln = rnd.randint(2, 10)
        w = "".join(rnd.choice(pool) for _ in range(ln))
        out.append(w)
    return " ".join(out)


def generate_dir_with_files(root: str, files_count: int = 600, max_depth: int = 3):
    # Створює директорію з піддиректоріями та txt файлами
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(42)
    for i in range(files_count):
        depth = rnd.randint(0, max_depth)
        cur = base
        for d in range(depth):
            cur = cur / f"dir_{rnd.randint(1, 15)}"
        cur.mkdir(parents=True, exist_ok=True)

        words = rnd.randint(80, 450)
        text = generate_random_text(words, seed=i + 100)

        fp = cur / f"file_{i:04d}.txt"
        fp.write_text(text, encoding="utf-8")


def list_text_files(root: str) -> list[Path]:
    # Рекурсивно збирає всі .txt файли
    base = Path(root)
    if not base.exists():
        return []
    return [p for p in base.rglob("*.txt") if p.is_file()]


def count_words_in_file(path: Path) -> int:
    # Рахує кількість слів у файлі
    data = path.read_text(encoding="utf-8", errors="ignore")
    return len(data.split())


def count_words_sequential(files: list[Path]) -> int:
    # Послідовний підрахунок слів у всіх файлах
    total = 0
    for f in files:
        total += count_words_in_file(f)
    return total


def count_words_threaded(files: list[Path], workers: int) -> int:
    # Паралельний підрахунок слів у файлах через пул потоків (I/O-bound)
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(count_words_in_file, f) for f in files]
        for fut in as_completed(futs):
            total += fut.result()
    return total


def count_words_processed(files: list[Path], workers: int) -> int:
    # Паралельний підрахунок слів у файлах через пул процесів
    total = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(count_words_in_file, f) for f in files]
        for fut in as_completed(futs):
            total += fut.result()
    return total

def io_demo(
    threads_list: list[int],
    root: str = "data_io",
    files_count: int = 600,
    force_regen: bool = False,
    return_results: bool = False,
):
    print("\n--- I/O-bound задача ---")
    files_count = min(1000, max(1, files_count))
    print(f"Файли: {files_count} шт. | директорія: {root}")

    base = Path(root)


    if force_regen and base.exists():
        for p in base.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted([p for p in base.rglob("*") if p.is_dir()], reverse=True):
            p.rmdir()
        base.rmdir()

    gen_time = 0.0
    generated = False
    if not base.exists() or len(list_text_files(root)) == 0:
        gen_time, _ = _time_it(generate_dir_with_files, root, files_count)
        generated = True

    files = list_text_files(root)
    print(f"Знайдено .txt файлів: {len(files)} | генерація зараз: {generated} | час генерації: {gen_time:.4f} c")

    dt, total_seq = _time_it(count_words_sequential, files)
    print(f"Послідовно: час, c: {dt:.4f} | слів: {total_seq}")

    results = {
        "io": {
            "title": "Count words in many files",
            "gen_time": float(gen_time),
            "seq": float(dt),
            "threads": {},
            "procs": {},
            "files": int(len(files)),
            "generated_now": bool(generated),
            "words": int(total_seq),
            "root": str(root),
        }
    }

    for w in threads_list:
        if w <= 1:
            continue
        dt2, total_thr = _time_it(count_words_threaded, files, w)
        print(f"Паралельно (потоків={w}): час, c: {dt2:.4f} | слів: {total_thr}")
        results["io"]["threads"][int(w)] = float(dt2)

        dt3, total_proc = _time_it(count_words_processed, files, w)
        print(f"Паралельно (процесів={w}): час, c: {dt3:.4f} | слів: {total_proc}")
        results["io"]["procs"][int(w)] = float(dt3)

    if return_results:
        return results