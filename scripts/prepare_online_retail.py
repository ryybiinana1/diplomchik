from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


DATA_DIR = Path("data")
RAW_ZIP = DATA_DIR / "online_retail.zip"
RAW_XLSX = DATA_DIR / "online_retail.xlsx"
OUTPUT = DATA_DIR / "online_retail_canonical.csv"

DATA_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"


def ensure_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_XLSX.exists():
        print(f"Файл уже есть: {RAW_XLSX}")
        return

    print("Скачиваю датасет Online Retail...")
    r = requests.get(DATA_URL, timeout=120)
    r.raise_for_status()
    RAW_ZIP.write_bytes(r.content)

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not xlsx_names:
            raise FileNotFoundError("В архиве не найден .xlsx файл")

        with zf.open(xlsx_names[0]) as src, RAW_XLSX.open("wb") as dst:
            dst.write(src.read())

    print(f"Извлечён файл: {RAW_XLSX}")


def main():
    ensure_dataset()

    df = pd.read_excel(RAW_XLSX)

    df = df.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]).copy()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    # ВАЖНО: cancellation не удаляем, а сохраняем флаг
    df["is_cancellation"] = df["InvoiceNo"].astype(str).str.startswith("C")

    # Revenue для покупок как положительная сумма;
    # cancellation суммы тоже сохраняем по модулю через amount/feature logic позже
    df["Revenue"] = (df["Quantity"].abs() * df["UnitPrice"].abs()).astype(float)

    df["CustomerID"] = df["CustomerID"].astype(str)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    df.to_csv(OUTPUT, index=False)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()