# benchmark.py
import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cpu_tasks import cpu_demo
from io_task import io_demo
from memory_task import memory_demo


def _now():
    return perf_counter()


def run_all():
    parser = argparse.ArgumentParser(
        description="ЛР1 МТПП: послідовно vs паралельно (потоки/процеси)"
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Кількість потоків/процесів для тестів")
    parser.add_argument("--matrix-size", type=int, default=10000,
                        help="Розмір матриці N для транспонування (NxN)")
    parser.add_argument("--files", type=int, default=600,
                        help="Кількість файлів для I/O задачі")
    parser.add_argument("--root", type=str, default="data_io",
                        help="Коренева директорія для I/O задачі")
    parser.add_argument("--outdir", type=str, default="figures",
                        help="Куди зберігати графіки/результати")


    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--with-time", action="store_true")

    args = parser.parse_args()

    print("Лабораторна робота №1 (МТПП)")
    print(f"Потоки для тестів: {args.threads}")

    t0 = _now()
    cpu_res = cpu_demo(args.threads, return_results=True)
    print(f"\nCPU-bound: загальний час блоку, c: {(_now() - t0):.4f}")

    t0 = _now()
    mem_res = memory_demo(args.threads, size=args.matrix_size, return_results=True)
    print(f"\nMemory-bound: загальний час блоку, c: {(_now() - t0):.4f}")

    t0 = _now()
    io_res = io_demo(args.threads, root=args.root, files_count=args.files, return_results=True)
    print(f"\nI/O-bound: загальний час блоку, c: {(_now() - t0):.4f}")

    results = {}
    results.update(cpu_res)
    results.update(mem_res)
    results.update(io_res)

    if args.no_plots:
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    workers_all = sorted(set([1] + [int(x) for x in args.threads if int(x) >= 1]))

    def _xy_for_series(seq_time: float, series: dict) -> tuple[list[int], list[float]]:
        """Повертає (x, y_times) з додаванням точки (1, seq_time)."""
        series = {int(k): float(v) for k, v in series.items()}
        x = [1] + [w for w in workers_all if w != 1 and w in series]
        y = [seq_time] + [series[w] for w in x[1:]]
        return x, y

    def _speedup(seq_time: float, y_times: list[float]) -> list[float]:
        return [seq_time / t for t in y_times]

    def save_combined_speedup(title: str, xlabel: str, lines: list[tuple[str, float, dict]], fname: str):
        """
        lines: [(label, seq_time, series_workers_to_time), ...]
        Малює один графік speedup з кількома лініями.
        """
        plt.figure()
        used_xticks = None

        for label, seq_time, series in lines:
            x, y = _xy_for_series(seq_time, series)
            sp = _speedup(seq_time, y)
            plt.plot(x, sp, marker="o", label=label)
            used_xticks = x  # вони однакові за змістом, беремо останні

        if used_xticks:
            plt.xticks(used_xticks, [str(v) for v in used_xticks])

        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xlabel(xlabel)
        plt.ylabel("Прискорення (T1 / Tp)")
        plt.title(title)
        plt.grid(True)
        plt.legend()

        out_path = (outdir / fname).resolve()
        print("Saving to:", out_path)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    def save_combined_time(title: str, xlabel: str, lines: list[tuple[str, float, dict]], fname: str):
        """
        lines: [(label, seq_time, series_workers_to_time), ...]
        Малює один графік часу з кількома лініями.
        """
        plt.figure()
        used_xticks = None

        for label, seq_time, series in lines:
            x, y = _xy_for_series(seq_time, series)
            plt.plot(x, y, marker="o", label=label)
            used_xticks = x

        if used_xticks:
            plt.xticks(used_xticks, [str(v) for v in used_xticks])

        plt.xlabel(xlabel)
        plt.ylabel("Час, сек")
        plt.title(title)
        plt.grid(True)
        plt.legend()

        out_path = (outdir / fname).resolve()
        print("Saving to:", out_path)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    # CPU: 2 файли (threads + procs), кожен містить 3 лінії задач
    cpu_threads_lines = []
    cpu_procs_lines = []
    for key, info in results["cpu"].items():
        seq = float(info["seq"])
        title = info["title"]
        thr = {int(k): float(v) for k, v in info["threads"].items()}
        cpu_threads_lines.append((title, seq, thr))

        procs = {int(k): float(v) for k, v in info.get("procs", {}).items()}
        if procs:
            cpu_procs_lines.append((title, seq, procs))

    if cpu_threads_lines:
        save_combined_speedup(
            "CPU-bound — прискорення (потоки): π / факторизація / прості числа",
            "Кількість потоків",
            cpu_threads_lines,
            "cpu_threads_speedup.png",
        )
        if args.with_time:
            save_combined_time(
                "CPU-bound — час (потоки): π / факторизація / прості числа",
                "Кількість потоків",
                cpu_threads_lines,
                "cpu_threads_time.png",
            )

    if cpu_procs_lines:
        save_combined_speedup(
            "CPU-bound — прискорення (процеси): π / факторизація / прості числа",
            "Кількість процесів",
            cpu_procs_lines,
            "cpu_procs_speedup.png",
        )
        if args.with_time:
            save_combined_time(
                "CPU-bound — час (процеси): π / факторизація / прості числа",
                "Кількість процесів",
                cpu_procs_lines,
                "cpu_procs_time.png",
            )

    # Memory: 1 файл (дві лінії threads vs procs)
    mem_seq = float(results["memory"]["seq"])
    mem_thr = {int(k): float(v) for k, v in results["memory"]["threads"].items()}
    mem_lines = [("Потоки", mem_seq, mem_thr)]

    mem_procs = {int(k): float(v) for k, v in results["memory"].get("procs", {}).items()}
    if mem_procs:
        mem_lines.append(("Процеси", mem_seq, mem_procs))

    save_combined_speedup(
        "Memory-bound — транспонування матриці: прискорення",
        "Кількість worker-ів",
        mem_lines,
        "memory_speedup.png",
    )
    if args.with_time:
        save_combined_time(
            "Memory-bound — транспонування матриці: час",
            "Кількість worker-ів",
            mem_lines,
            "memory_time.png",
        )

    # I/O: 1 файл (дві лінії threads vs procs)
    io_seq = float(results["io"]["seq"])
    io_thr = {int(k): float(v) for k, v in results["io"]["threads"].items()}
    io_lines = [("Потоки", io_seq, io_thr)]

    io_procs = {int(k): float(v) for k, v in results["io"].get("procs", {}).items()}
    if io_procs:
        io_lines.append(("Процеси", io_seq, io_procs))

    save_combined_speedup(
        "I/O-bound — підрахунок слів у файлах: прискорення",
        "Кількість worker-ів",
        io_lines,
        "io_speedup.png",
    )
    if args.with_time:
        save_combined_time(
            "I/O-bound — підрахунок слів у файлах: час",
            "Кількість worker-ів",
            io_lines,
            "io_time.png",
        )