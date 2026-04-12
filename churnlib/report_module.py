from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .economy_module import Scenario, expected_value, profit_curve, best_k


@dataclass(frozen=True)
class ReportConfig:
    out_dir: str
    top_k: int = 500
    value_col: str = "value_proxy"


def write_reports(df_test: pd.DataFrame, p: np.ndarray, scenarios: List[Scenario], cfg: ReportConfig) -> Dict[str, Any]:
    out = Path(cfg.out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    df = df_test.copy()
    df["p"] = np.asarray(p, dtype=float)

    V = df[cfg.value_col].astype(float).values
    base = next((s for s in scenarios if s.name == "base"), scenarios[0])
    df["EV_base"] = expected_value(df["p"].values, V, base)

    prio = df.sort_values("EV_base", ascending=False).head(cfg.top_k)
    prio_path = out / "tables" / "priority_list_topk.csv"
    prio.to_csv(prio_path, index=False)

    plt.figure(figsize=(8, 5))
    summary_rows = []
    base_best_k = None
    base_max_profit = None

    for sc in scenarios:
        curve = profit_curve(df["p"].values, V, sc)
        k, mx = best_k(curve)
        summary_rows.append({"scenario": sc.name, "best_k": k, "max_profit": mx})
        plt.plot(curve, label=f"{sc.name} (best_k={k})")

        if sc.name == "base":
            base_best_k = k
            base_max_profit = mx

    plt.title("Profit curve (cumulative EV for top-k)")
    plt.xlabel("k")
    plt.ylabel("Expected profit")
    plt.legend()
    plot_path = out / "plots" / "profit_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    summary_path = out / "tables" / "profit_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return {
        "priority_csv": str(prio_path),
        "profit_plot": str(plot_path),
        "profit_summary": str(summary_path),
        "base_best_k": int(base_best_k) if base_best_k is not None else None,
        "base_max_profit": float(base_max_profit) if base_max_profit is not None else None,
    }