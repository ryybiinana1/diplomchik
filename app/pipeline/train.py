from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.pipeline.experiment import run_experiment_grid, run_single_experiment
from app.pipeline.profiling import profile_dataset
from churnlib.cards_module import write_datasheet
from churnlib.data_module import canonicalize_types
from churnlib.validation_module import basic_validate


def run_pipeline(
    input_csv: str,
    template: str,
    mapping: Dict[str, str],
    params: Dict[str, Any],
    out_dir: str,
    status_callback=None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if status_callback:
        status_callback(stage="loading_input", progress=5)

    df_raw = pd.read_csv(input_csv, encoding="utf-8-sig")

    if status_callback:
        status_callback(stage="profiling_and_validation", progress=10)

    quality_report = profile_dataset(df_raw, template)

    (out / "inputs").mkdir(exist_ok=True)
    (out / "inputs" / "quality_report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    df = df_raw.rename(columns={v: k for k, v in mapping.items() if v}).copy()

    if template == "transactions" and "amount" not in df.columns:
        if "unit_price" in df.columns and "quantity" in df.columns:
            up = pd.to_numeric(df["unit_price"], errors="coerce")
            q = pd.to_numeric(df["quantity"], errors="coerce")
            df["amount"] = up * q

    basic_validate(df, template=template)
    df = canonicalize_types(df, template)

    write_datasheet(df_raw, template, out / "datasheet.md")

    df.to_parquet(out / "inputs" / "canonical_input.parquet", index=False)

    params_local = dict(params)
    params_local["mapping_used"] = mapping
    params_local["input_source_columns"] = list(df_raw.columns)
    params_local["input_source_dtypes"] = {col: str(df_raw[col].dtype) for col in df_raw.columns}

    has_grid = any(
        key in params_local
        for key in ["horizon_days_grid", "history_days_grid", "step_days_grid", "model_kind_grid"]
    )

    if status_callback:
        status_callback(stage="experiment_grid", progress=25)

    if has_grid:
        grid_res = run_experiment_grid(
            df=df,
            template=template,
            mapping_used=mapping,
            params=params_local,
            out_dir=out / "grid_runs",
            status_callback=status_callback,
        )
        best_result = grid_res["best_result"]
        if status_callback:
            status_callback(stage="finalizing_artifacts", progress=90)

        return {
            **best_result,
            "template": template,
            "mode": "grid_search",
            "quality_report_path": str(out / "inputs" / "quality_report.json"),
            "n_experiments": grid_res["n_experiments"],
            "selection_metric": grid_res["selection_metric"],
            "all_results_csv": grid_res["all_results_csv"],
        }

    res = run_single_experiment(df, template, mapping, params_local, out / "single_run")
    res["quality_report_path"] = str(out / "inputs" / "quality_report.json")

    if status_callback:
        status_callback(stage="finalizing_artifacts", progress=90)

    return res
