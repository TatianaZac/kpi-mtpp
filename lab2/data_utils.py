from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

HTML_TAGS = [
    "html", "head", "body", "title", "meta", "script", "style", "header",
    "main", "section", "article", "div", "span", "p", "a", "ul", "li",
    "nav", "footer", "img", "table", "tr", "td", "form", "input", "button",
]

CURRENCIES = ["UAH", "USD", "EUR", "GBP"]
PRODUCT_TYPES = ["books", "games", "music", "electronics", "home", "sport"]



def ensure_html_dataset(root: str | Path, files_count: int = 1000, seed: int = 42) -> list[str]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("*.html"))
    if len(existing) >= files_count:
        return [str(p) for p in existing[:files_count]]

    rnd = random.Random(seed)
    for i in range(len(existing), files_count):
        chunks = ["<!DOCTYPE html>", "<html>", "<head>"]
        chunks.append(f"<title>Document {i}</title>")
        if i % 3 == 0:
            chunks.append('<meta charset="utf-8">')
        if i % 5 == 0:
            chunks.append("<script>const x = 1;</script>")
        chunks.append("</head><body>")

        block_count = rnd.randint(10, 80)
        for j in range(block_count):
            tag = rnd.choice(HTML_TAGS[7:])
            if tag in {"img", "input", "meta"}:
                chunks.append(f"<{tag} data-i='{i}' data-j='{j}'>")
            else:
                chunks.append(f"<{tag}>item_{i}_{j}</{tag}>")
            if j % 7 == 0:
                chunks.append(f"<div><p>nested_{j}</p><span>value_{i}</span></div>")
            if j % 11 == 0:
                chunks.append("<ul><li>a</li><li>b</li><li>c</li></ul>")

        chunks.append("</body></html>")
        (root / f"doc_{i:04d}.html").write_text("\n".join(chunks), encoding="utf-8")

    return [str(p) for p in sorted(root.glob("*.html"))[:files_count]]



def generate_number_array(size: int = 1_000_000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=100.0, scale=25.0, size=size)
    expo = rng.exponential(scale=10.0, size=size)
    signs = rng.choice([-1.0, 1.0], size=size, p=[0.15, 0.85])
    arr = normal + signs * expo
    return arr.astype(np.float64)



def generate_matrices(size: int = 256, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.random((size, size), dtype=np.float64)
    b = rng.random((size, size), dtype=np.float64)
    return a, b



def ensure_transactions_csv(path: str | Path, rows: int = 200_000, seed: int = 42) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return str(path)

    rnd = random.Random(seed)
    rates = {"UAH": 1.0, "USD": 39.5, "EUR": 43.0, "GBP": 50.0}

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tx_id", "user_id", "amount", "currency", "date",
            "product_type", "status_flag", "rate_hint"
        ])
        for i in range(rows):
            currency = rnd.choice(CURRENCIES)
            amount = round(rnd.uniform(50.0, 5000.0), 2)
            month = rnd.randint(1, 12)
            day = rnd.randint(1, 28)
            writer.writerow([
                i + 1,
                rnd.randint(1, 25_000),
                amount,
                currency,
                f"2026-{month:02d}-{day:02d}",
                rnd.choice(PRODUCT_TYPES),
                rnd.choice(["regular", "vip", "premium"]),
                rates[currency],
            ])
    return str(path)
