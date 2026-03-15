from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import (
    ensure_html_dataset,
    ensure_transactions_csv,
    generate_matrices,
    generate_number_array,
)
from pattern_tasks import compare_patterns
from transaction_pipeline import compare_transaction_patterns


def _now():
    return perf_counter()




def _plot_group(ax, title: str, seq: float, series: dict, workers_all: list[int], ylabel: str, speedup: bool = False):
    ax.set_title(title)
    ax.set_xlabel("Кількість потоків")
    ax.set_ylabel(ylabel)

    if speedup:
        seq_line = [1.0 for _ in workers_all]
        ax.plot(workers_all, seq_line, marker="o", label="sequential")
    else:
        seq_line = [seq for _ in workers_all]
        ax.plot(workers_all, seq_line, marker="o", label="sequential")

    for label, values in series.items():
        ys = []
        for w in workers_all:
            if w == 1:
                ys.append(1.0 if speedup else seq)
            else:
                v = values.get(str(w), values.get(w))
                if v is None:
                    ys.append(None)
                else:
                    ys.append((seq / v) if speedup else v)
        ax.plot(workers_all, ys, marker="o", label=label)

    ax.grid(True, alpha=0.3)
    ax.legend()


def _save_pattern_plots(results: dict, outdir: Path, workers_all: list[int]):
    for key, item in results["patterns"].items():
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        _plot_group(
            ax=ax,
            title=f"{item['title']} — час виконання",
            seq=item["seq"],
            series={
                "map_reduce": item["map_reduce"],
                "fork_join": item["fork_join"],
                "worker_pool": item["worker_pool"],
            },
            workers_all=workers_all,
            ylabel="Час, c",
            speedup=False,
        )
        fig.tight_layout()
        fig.savefig(outdir / f"{key}_time.png", dpi=150)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        _plot_group(
            ax=ax,
            title=f"{item['title']} — прискорення",
            seq=item["seq"],
            series={
                "map_reduce": item["map_reduce"],
                "fork_join": item["fork_join"],
                "worker_pool": item["worker_pool"],
            },
            workers_all=workers_all,
            ylabel="Прискорення",
            speedup=True,
        )
        fig.tight_layout()
        fig.savefig(outdir / f"{key}_speedup.png", dpi=150)
        plt.close(fig)


def _save_transactions_plots(results: dict, outdir: Path, workers_all: list[int]):
    item = results["transactions"]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    _plot_group(
        ax=ax,
        title=f"{item['title']} — час виконання",
        seq=item["seq"],
        series={
            "pipeline": item["pipeline"],
            "producer_consumer": item["producer_consumer"],
        },
        workers_all=workers_all,
        ylabel="Час, c",
        speedup=False,
    )
    fig.tight_layout()
    fig.savefig(outdir / "transactions_time.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    _plot_group(
        ax=ax,
        title=f"{item['title']} — прискорення",
        seq=item["seq"],
        series={
            "pipeline": item["pipeline"],
            "producer_consumer": item["producer_consumer"],
        },
        workers_all=workers_all,
        ylabel="Прискорення",
        speedup=True,
    )
    fig.tight_layout()
    fig.savefig(outdir / "transactions_speedup.png", dpi=150)
    plt.close(fig)


def _build_summary(results: dict) -> dict:
    summary = {
        "patterns": {},
        "transactions": results["transactions"]["best"],
        "overall_recommendation": {},
    }

    best_candidates = []

    for key, item in results.get("patterns", {}).items():
        summary["patterns"][key] = {
            "title": item["title"],
            "best": item["best"],
            "validation": item["validation"],
        }
        best_candidates.append((item["title"], item["best"]))

    tx_item = results["transactions"]
    best_candidates.append((tx_item["title"], tx_item["best"]))

    title, best = min(best_candidates, key=lambda x: x[1]["time"])
    summary["overall_recommendation"] = {
        "title": title,
        **best,
    }
    return summary


def _print_summary(summary: dict):
    print("\nВИСНОВОК ПРО НАЙКРАЩИЙ ПІДХІД")
    for item in summary["patterns"].values():
        best = item["best"]
        print(
            f"{item['title']}: {best['pattern']} ({best['configuration']}), "
            f"час {best['time']:.4f} c, прискорення {best['speedup_vs_sequential']:.4f}"
        )

    tx_best = summary["transactions"]
    print(
        f"Фінансові транзакції: {tx_best['pattern']} ({tx_best['configuration']}), "
        f"час {tx_best['time']:.4f} c, прискорення {tx_best['speedup_vs_sequential']:.4f}"
    )

    overall = summary["overall_recommendation"]
    print(
        f"Загалом найкращий за часом варіант у цьому запуску: {overall['title']} -> "
        f"{overall['pattern']} ({overall['configuration']})"
    )
    print("Універсального найкращого патерну немає: вибір залежить від типу задачі й накладних витрат.")


def run_all():
    parser = argparse.ArgumentParser(
        description="ЛР2 МТПП: порівняння патернів паралельного програмування"
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8], help="Кількість потоків для тестів")
    parser.add_argument("--html-files", type=int, default=1000, help="Кількість HTML-документів")
    parser.add_argument("--array-size", type=int, default=1_000_000, help="Розмір масиву для статистики")
    parser.add_argument("--matrix-size", type=int, default=1024, help="Розмір квадратних матриць NxN")
    parser.add_argument("--transactions", type=int, default=200_000, help="Кількість транзакцій у CSV")
    parser.add_argument("--data-root", type=str, default="data_lab2", help="Коренева директорія для наборів даних")
    parser.add_argument("--outdir", type=str, default="figures_lab2", help="Куди зберігати графіки та results.json")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    print(f"Потоки для тестів: {args.threads}")

    data_root = Path(args.data_root)
    html_dir = data_root / "html"
    tx_csv = data_root / "transactions" / "transactions.csv"

    html_paths = ensure_html_dataset(html_dir, files_count=args.html_files)
    array_data = generate_number_array(size=args.array_size)
    a, b = generate_matrices(size=args.matrix_size)
    tx_path = ensure_transactions_csv(tx_csv, rows=args.transactions)

    t0 = _now()
    pattern_res = compare_patterns(html_paths, array_data, a, b, args.threads, return_results=True)
    print(f"\nПатерни: загальний час блоку, c: {(_now() - t0):.4f}")

    t0 = _now()
    tx_res = compare_transaction_patterns(tx_path, args.threads, return_results=True)
    print(f"\nТранзакції: загальний час блоку, c: {(_now() - t0):.4f}")

    results = {}
    results.update(pattern_res)
    results.update(tx_res)
    results["summary"] = _build_summary(results)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(results["summary"])

    if args.no_plots:
        return

    workers_all = sorted(set([1] + [int(x) for x in args.threads if int(x) >= 1]))
    _save_pattern_plots(results, outdir, workers_all)
    _save_transactions_plots(results, outdir, workers_all)


if __name__ == "__main__":
    run_all()
