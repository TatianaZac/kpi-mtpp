from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from banking_tasks import compare_banking
from ipc_tasks import compare_ipc
from utils import ensure_dir, save_json



def _save_banking_plots(results: dict, outdir: Path, workers_all: list[int]) -> None:
    item = results["banking"]
    seq_time = item["sequential"]["time"]
    unsafe = item["unsafe_race"]
    safe = item["safe_race"]

    xs = workers_all
    unsafe_y = [unsafe[str(x)] for x in xs]
    safe_y = [safe[str(x)] for x in xs]
    seq_y = [seq_time for _ in xs]

    plt.figure(figsize=(10, 5))
    plt.plot(xs, seq_y, marker="o", label="sequential")
    plt.plot(xs, unsafe_y, marker="o", label="unsafe race")
    plt.plot(xs, safe_y, marker="o", label="safe with locks")
    plt.xlabel("Кількість потоків")
    plt.ylabel("Час, с")
    plt.title("Переказ коштів між рахунками — час виконання")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "banking_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(xs, [1.0 for _ in xs], marker="o", label="sequential")
    plt.plot(xs, [seq_time / v for v in unsafe_y], marker="o", label="unsafe race")
    plt.plot(xs, [seq_time / v for v in safe_y], marker="o", label="safe with locks")
    plt.xlabel("Кількість потоків")
    plt.ylabel("Прискорення")
    plt.title("Переказ коштів між рахунками — прискорення")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "banking_speedup.png", dpi=150)
    plt.close()

    preserved_unsafe = [1 if item["unsafe_checks"][str(x)]["total_preserved"] else 0 for x in xs]
    preserved_safe = [1 if item["safe_checks"][str(x)]["total_preserved"] else 0 for x in xs]

    plt.figure(figsize=(10, 5))
    plt.plot(xs, preserved_unsafe, marker="o", label="unsafe race")
    plt.plot(xs, preserved_safe, marker="o", label="safe with locks")
    plt.yticks([0, 1], ["ні", "так"])
    plt.xlabel("Кількість потоків")
    plt.ylabel("Збереження загальної суми")
    plt.title("Race condition — чи зберігається загальна сума коштів")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "banking_consistency.png", dpi=150)
    plt.close()



def _save_ipc_plots(results: dict, outdir: Path) -> None:
    item = results["ipc"]
    labels = ["Pipe", "SharedMemory", "Socket+Node"]
    values = [
        item["pipe"]["avg_round_trip_ms"],
        item["shared_memory"]["avg_round_trip_ms"],
        item["socket_node"]["avg_round_trip_ms"],
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylabel("Середній round-trip, мс")
    plt.title("IPC — середній час передачі числа туди й назад")
    plt.tight_layout()
    plt.savefig(outdir / "ipc_roundtrip.png", dpi=150)
    plt.close()



def _build_summary(results: dict) -> dict:
    banking = results["banking"]
    ipc = results["ipc"]
    best_safe_workers = banking["best_safe_workers"]

    return {
        "banking": {
            "sequential_time": banking["sequential"]["time"],
            "best_safe_workers": best_safe_workers,
            "best_safe_time": banking["best_safe_time"],
            "deadlock_unsafe_detected": banking["deadlock_unsafe"]["deadlock_detected"],
            "deadlock_safe_detected": banking["deadlock_safe"]["deadlock_detected"],
        },
        "ipc": ipc["best"],
        "recommendation": {
            "race_condition_fix": "Персональні lock-и для рахунків + глобальний порядок захоплення lock-ів.",
            "deadlock_fix": "Однаковий порядок захоплення ресурсів у всіх потоках.",
            "ipc_fastest_method": ipc["best"]["method"],
        },
    }



def _print_summary(summary: dict) -> None:
    print("\nПІДСУМКОВИЙ ВИСНОВОК")
    print(
        f"Найкращий безпечний варіант для переказів: workers={summary['banking']['best_safe_workers']}, "
        f"час={summary['banking']['best_safe_time']:.4f} c"
    )
    print(
        f"Unsafe deadlock виявлено: {summary['banking']['deadlock_unsafe_detected']} | "
        f"Safe deadlock: {summary['banking']['deadlock_safe_detected']}"
    )
    print(
        f"Найшвидший IPC-метод: {summary['ipc']['title']} | "
        f"avg={summary['ipc']['avg_round_trip_ms']:.4f} ms"
    )



def run_all() -> None:
    parser = argparse.ArgumentParser(description="ЛР3 МТПП: race condition, deadlock та IPC")
    parser.add_argument("--threads", type=int, nargs="+", default=[50, 100, 250, 500, 1000, 1500], help="Кількість потоків для банківських переказів")
    parser.add_argument("--accounts", type=int, default=128, help="Кількість рахунків")
    parser.add_argument("--operations", type=int, default=80_000, help="Кількість операцій переказу")
    parser.add_argument("--pipe-rounds", type=int, default=1000, help="Кількість ітерацій для Pipe")
    parser.add_argument("--shm-rounds", type=int, default=1000, help="Кількість ітерацій для shared memory")
    parser.add_argument("--socket-rounds", type=int, default=1000, help="Кількість ітерацій для socket+Node")
    parser.add_argument("--outdir", type=str, default="results_lab3", help="Куди зберігати графіки та results.json")
    parser.add_argument("--node-script", type=str, default="node_helper.js", help="Шлях до Node.js helper")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    print("Лабораторна робота №3 (МТПП)")
    print(f"Потоки для тестів: {args.threads}")

    results = {}
    results.update(compare_banking(args.threads, accounts_count=args.accounts, operations=args.operations, return_results=True))
    results.update(compare_ipc(pipe_rounds=args.pipe_rounds, shm_rounds=args.shm_rounds, socket_rounds=args.socket_rounds, node_script=args.node_script, return_results=True))
    results["summary"] = _build_summary(results)

    outdir = ensure_dir(args.outdir)
    save_json(results, outdir / "results.json")
    _print_summary(results["summary"])

    if args.no_plots:
        return

    _save_banking_plots(results, outdir, args.threads)
    _save_ipc_plots(results, outdir)


if __name__ == "__main__":
    run_all()
