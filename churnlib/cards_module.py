from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def write_datasheet(
    df,
    template: str,
    out_path: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "template": template,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "null_share": df.isna().mean().round(6).to_dict(),
        "duplicates_full_rows": int(df.duplicated().sum()),
        "extra": extra or {},
    }

    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"datasheet_path": str(out)}


def write_model_card(
    out_path: str,
    template: str | None = None,
    metrics: Dict[str, Any] | None = None,
    params: Dict[str, Any] | None = None,
    params_used: Dict[str, Any] | None = None,
    suitability: Dict[str, Any] | None = None,
    extra_feature_audit: List[Dict[str, Any]] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    if params is None:
        params = params_used or {}

    payload = {
        "template": template,
        "metrics": metrics or {},
        "params": params,
        "suitability": suitability or {},
        "extra_feature_audit": extra_feature_audit or [],
        "extra": kwargs,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"model_card_path": str(out)}