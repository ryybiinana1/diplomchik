# churnlib/explain_module.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ExplainConfig:
    max_background: int = 2000
    max_explain: int = 500
    random_state: int = 42
    top_n_local: int = 3


def shap_explain_global(model, X: pd.DataFrame, cfg: ExplainConfig) -> Dict:
    import shap

    X_bg = X.sample(min(cfg.max_background, len(X)), random_state=cfg.random_state)
    X_ex = X.sample(min(cfg.max_explain, len(X)), random_state=cfg.random_state)

    explainer = shap.Explainer(model, X_bg)
    shap_values = explainer(X_ex)

    return {
        "explainer": explainer,
        "X_explain": X_ex,
        "shap_values": shap_values,
    }


def save_shap_artifacts(
    explain_res: Dict,
    out_dir: str | Path,
    top_risk_df: pd.DataFrame | None = None,
):
    import shap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_ex = explain_res["X_explain"]
    shap_values = explain_res["shap_values"]

    # global beeswarm
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # bar importance
    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # optional local waterfall for selected rows
    if top_risk_df is not None and len(top_risk_df) > 0:
        for i, idx in enumerate(top_risk_df.index[:3]):
            try:
                plt.figure()
                shap.plots.waterfall(shap_values[idx], show=False, max_display=15)
                plt.tight_layout()
                plt.savefig(out_dir / f"shap_local_{i+1}.png", dpi=150, bbox_inches="tight")
                plt.close()
            except Exception:
                pass

    return {
        "shap_beeswarm": str(out_dir / "shap_beeswarm.png"),
        "shap_bar": str(out_dir / "shap_bar.png"),
    }