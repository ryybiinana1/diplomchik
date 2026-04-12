from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import sklearn

from churnlib.drift_module import compute_feature_psi
from churnlib.explain_module import ExplainConfig, shap_explain_global, save_shap_artifacts
from churnlib.data_module import canonicalize_types, build_snapshots, build_latest_snapshot, SnapshotConfig
from churnlib.model_module import train_time_cv, walk_forward_backtest, TrainConfig
from churnlib.calibration_module import calibrate, save_calibration_plot, save_calibration_summary, CalibrationConfig
from churnlib.report_module import write_reports, ReportConfig
from churnlib.economy_module import DEFAULT_SCENARIOS, expected_value
from churnlib.cards_module import write_model_card, write_datasheet
from churnlib.validation_module import basic_validate, profile_dataset


def _prepare_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()

    for col in X.columns:
        if str(X[col].dtype).startswith("datetime"):
            X[col] = pd.to_datetime(X[col]).view("int64") // 10**9

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def get_model_param_grid(params: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    default_grid = {
        "logreg": [
            {"C": 0.1},
            {"C": 1.0},
            {"C": 5.0},
        ],
        "sklearn_gbdt": [
            {"max_iter": 300, "learning_rate": 0.05},
            {"max_iter": 500, "learning_rate": 0.03},
        ],
        "random_forest": [
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5},
            {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 3},
        ],
        "lightgbm": [
            {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50},
            {"learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 20},
        ],
        "catboost": [
            {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 10},
            {"learning_rate": 0.03, "depth": 8, "l2_leaf_reg": 3},
        ],
    }
    return params.get("model_param_grid", default_grid)


def save_model_bundle(
    out_dir: Path,
    model,
    calibrator,
    feature_cols: List[str],
    mapping: Dict[str, str],
    template: str,
    params: Dict[str, Any],
) -> Dict[str, str]:
    bundle_dir = out_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, bundle_dir / "model.pkl")
    joblib.dump(calibrator, bundle_dir / "calibrator.pkl")

    with open(bundle_dir / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    with open(bundle_dir / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    with open(bundle_dir / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "template": template,
                "params": params,
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "bundle_dir": str(bundle_dir),
        "model": str(bundle_dir / "model.pkl"),
        "calibrator": str(bundle_dir / "calibrator.pkl"),
        "feature_cols": str(bundle_dir / "feature_cols.json"),
        "mapping": str(bundle_dir / "mapping.json"),
        "pipeline_config": str(bundle_dir / "pipeline_config.json"),
    }


def load_model_bundle(bundle_dir: str | Path) -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)

    model = joblib.load(bundle_dir / "model.pkl")
    calibrator = joblib.load(bundle_dir / "calibrator.pkl")

    with open(bundle_dir / "feature_cols.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    with open(bundle_dir / "mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    with open(bundle_dir / "pipeline_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    return {
        "model": model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "mapping": mapping,
        "config": config,
    }


def choose_metric(result: Dict[str, Any], metric_name: str) -> float:
    if metric_name == "base_max_profit":
        return float(result["business_metrics"]["base_max_profit"])

    if metric_name in result["test_metrics_cal"]:
        value = float(result["test_metrics_cal"][metric_name])
        if metric_name == "brier":
            return -value
        return value

    if metric_name in result["cv_metrics_mean"]:
        value = float(result["cv_metrics_mean"][metric_name])
        if metric_name == "brier":
            return -value
        return value

    if metric_name in result.get("walk_forward", {}).get("mean_metrics", {}):
        value = float(result["walk_forward"]["mean_metrics"][metric_name])
        if metric_name == "brier":
            return -value
        return value

    raise ValueError(f"Unknown metric for selection: {metric_name}")


def run_single_experiment(
    df: pd.DataFrame,
    template: str,
    params: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_cfg = SnapshotConfig(
        template=template,
        horizon_days=int(params.get("horizon_days", 30)),
        history_days=int(params.get("history_days", 180)),
        step_days=int(params.get("step_days", 30)),
        min_events_in_history=int(params.get("min_events_in_history", 1)),
        min_lifetime_days=int(params.get("min_lifetime_days", 0)),
        max_recency_days=params.get("max_recency_days", None),
    )

    snaps = build_snapshots(df, snap_cfg)
    if snaps.empty:
        raise ValueError(
            f"Empty snapshots for horizon={snap_cfg.horizon_days}, "
            f"history={snap_cfg.history_days}, step={snap_cfg.step_days}"
        )

    snaps.to_parquet(out_dir / "snapshots.parquet", index=False)

    target_rate = float(snaps["target"].mean()) if len(snaps) else None
    class_counts = snaps["target"].value_counts(dropna=False).to_dict()

    feature_cols = [c for c in snaps.columns if c not in ["entity_id", "anchor_time", "target"]]

    train_cfg = TrainConfig(
        model_kind=str(params.get("model_kind", "lightgbm")),
        use_class_weight=bool(params.get("use_class_weight", False)),
        model_params=params.get("model_params", {}),
    )

    train_res = train_time_cv(snaps, feature_cols, train_cfg)

    backtest_res = walk_forward_backtest(snaps, feature_cols, train_cfg)
    backtest_res["folds_df"].to_csv(out_dir / "walk_forward_folds.csv", index=False)

    cal_cfg = CalibrationConfig(
        method=str(params.get("calibration", "sigmoid")),
        cv=3,
    )

    cal_res = calibrate(
        train_res["model"],
        _prepare_feature_matrix(train_res["train_df"], feature_cols),
        train_res["train_df"]["target"].astype(int).values,
        _prepare_feature_matrix(train_res["test_df"], feature_cols),
        train_res["test_df"]["target"].astype(int).values,
        cal_cfg,
    )

    save_calibration_plot(
        train_res["y_test"],
        cal_res["p_raw"],
        cal_res["p_cal"],
        str(out_dir / "plots_calibration.png"),
    )

    save_calibration_summary(
        cal_res,
        str(out_dir / "calibration_summary.json"),
    )

    rep_paths = write_reports(
        train_res["test_df"],
        cal_res["p_cal"],
        DEFAULT_SCENARIOS,
        ReportConfig(out_dir=str(out_dir)),
    )

    psi_path = None
    try:
        psi_df = compute_feature_psi(train_res["train_df"], train_res["test_df"], feature_cols)
        psi_path = out_dir / "feature_psi.csv"
        psi_df.to_csv(psi_path, index=False)
    except Exception:
        psi_path = None

    explain_paths: Dict[str, Any] = {}
    try:
        X_test_explain = _prepare_feature_matrix(train_res["test_df"], feature_cols)

        explain_res = shap_explain_global(
            train_res["model"],
            X_test_explain,
            ExplainConfig(),
        )

        explain_paths = save_shap_artifacts(
            explain_res,
            out_dir / "explain",
        )
    except Exception as e:
        explain_paths = {"shap_error": str(e)}

    joblib.dump(train_res["model"], out_dir / "model.pkl")
    joblib.dump(cal_res["calibrator"], out_dir / "calibrator.pkl")

    train_res["train_df"].to_parquet(out_dir / "train_df.parquet", index=False)
    train_res["test_df"].to_parquet(out_dir / "test_df.parquet", index=False)

    write_model_card(
        template=template,
        horizon_days=snap_cfg.horizon_days,
        history_days=snap_cfg.history_days,
        metrics_raw=train_res["test_metrics"],
        metrics_cal=cal_res["metrics"],
        out_path=out_dir / "model_card.md",
    )

    bundle_paths = save_model_bundle(
        out_dir=out_dir,
        model=train_res["model"],
        calibrator=cal_res["calibrator"],
        feature_cols=feature_cols,
        mapping=params.get("mapping_used", {}),
        template=template,
        params=params,
    )

    return {
        "template": template,
        "n_snapshots": int(len(snaps)),
        "feature_count": int(len(feature_cols)),
        "class_balance": {
            "target_rate": target_rate,
            "class_counts": class_counts,
            "pos_weight": train_res.get("pos_weight"),
        },
        "params_used": {
            "horizon_days": snap_cfg.horizon_days,
            "history_days": snap_cfg.history_days,
            "step_days": snap_cfg.step_days,
            "min_events_in_history": snap_cfg.min_events_in_history,
            "min_lifetime_days": snap_cfg.min_lifetime_days,
            "max_recency_days": snap_cfg.max_recency_days,
            "model_kind": str(params.get("model_kind", "lightgbm")),
            "model_params": params.get("model_params", {}),
            "use_class_weight": bool(params.get("use_class_weight", False)),
            "calibration": str(params.get("calibration", "sigmoid")),
        },
        "walk_forward": {
            "mean_metrics": backtest_res["mean_metrics"],
            "std_metrics": backtest_res["std_metrics"],
        },
        "cv_metrics_mean": train_res["cv_metrics_mean"],
        "cv_metrics_std": train_res["cv_metrics_std"],
        "test_metrics_raw": train_res["test_metrics"],
        "test_metrics_cal": cal_res["metrics"],
        "calibration": {
            "method": cal_res["method"],
            "delta": cal_res["delta"],
        },
        "business_metrics": {
            "base_best_k": rep_paths.get("base_best_k"),
            "base_max_profit": rep_paths.get("base_max_profit"),
        },
        "artifacts": {
            **rep_paths,
            "model_card": str(out_dir / "model_card.md"),
            "snapshots": str(out_dir / "snapshots.parquet"),
            "train_df": str(out_dir / "train_df.parquet"),
            "test_df": str(out_dir / "test_df.parquet"),
            "model": str(out_dir / "model.pkl"),
            "calibrator": str(out_dir / "calibrator.pkl"),
            "calibration_summary": str(out_dir / "calibration_summary.json"),
            "calibration_plot": str(out_dir / "plots_calibration.png"),
            "walk_forward_folds": str(out_dir / "walk_forward_folds.csv"),
            "feature_psi": str(psi_path) if psi_path else None,
            "explain": explain_paths,
            "bundle": bundle_paths,
        },
    }


def run_experiment_grid(
    df: pd.DataFrame,
    template: str,
    params: Dict[str, Any],
    out_dir: Path,
    status_callback=None,
) -> Dict[str, Any]:
    horizon_grid = params.get("horizon_days_grid", [params.get("horizon_days", 60)])
    history_grid = params.get("history_days_grid", [params.get("history_days", 180)])
    step_grid = params.get("step_days_grid", [params.get("step_days", 30)])
    model_kind_grid = params.get("model_kind_grid", [params.get("model_kind", "lightgbm")])
    calibration_grid = params.get("calibration_grid", [params.get("calibration", "sigmoid")])

    model_param_grid = get_model_param_grid(params)
    selection_metric = str(params.get("selection_metric", "pr_auc"))

    all_param_combinations = []
    for h in horizon_grid:
        for w in history_grid:
            for s in step_grid:
                for model_kind in model_kind_grid:
                    for model_params in model_param_grid.get(model_kind, [{}]):
                        for calibration in calibration_grid:
                            all_param_combinations.append(
                                (h, w, s, model_kind, model_params, calibration)
                            )

    total_experiments = len(all_param_combinations)

    all_results: List[Dict[str, Any]] = []
    best_result = None
    best_score = None

    for idx, (h, w, s, model_kind, model_params, calibration) in enumerate(all_param_combinations, start=1):
        progress = 30 + int(50 * idx / max(total_experiments, 1))

        if status_callback:
            status_callback(
                stage="model_selection",
                progress=progress,
                extra={
                    "current_experiment": {
                        "index": idx,
                        "total": total_experiments,
                        "model_kind": model_kind,
                        "horizon_days": int(h),
                        "history_days": int(w),
                        "step_days": int(s),
                        "calibration": calibration,
                    }
                }
            )

        exp_params = dict(params)
        exp_params["horizon_days"] = int(h)
        exp_params["history_days"] = int(w)
        exp_params["step_days"] = int(s)
        exp_params["model_kind"] = model_kind
        exp_params["model_params"] = model_params
        exp_params["calibration"] = calibration

        exp_name = (
            f"h{h}_w{w}_s{s}_{model_kind}_"
            f"{abs(hash(json.dumps(model_params, sort_keys=True))) % 10**8}_{calibration}"
        )
        exp_dir = out_dir / "experiments" / exp_name

        try:
            res = run_single_experiment(df, template, exp_params, exp_dir)
            score = choose_metric(res, selection_metric)

            row = {
                "experiment": exp_name,
                "horizon_days": int(h),
                "history_days": int(w),
                "step_days": int(s),
                "model_kind": model_kind,
                "model_params": json.dumps(model_params, ensure_ascii=False, sort_keys=True),
                "calibration": calibration,
                "n_snapshots": res["n_snapshots"],

                "cv_roc_auc_mean": res["cv_metrics_mean"]["roc_auc"],
                "cv_pr_auc_mean": res["cv_metrics_mean"]["pr_auc"],
                "cv_brier_mean": res["cv_metrics_mean"]["brier"],

                "cv_roc_auc_std": res["cv_metrics_std"]["roc_auc"],
                "cv_pr_auc_std": res["cv_metrics_std"]["pr_auc"],
                "cv_brier_std": res["cv_metrics_std"]["brier"],

                "wf_roc_auc_mean": res["walk_forward"]["mean_metrics"]["roc_auc"],
                "wf_pr_auc_mean": res["walk_forward"]["mean_metrics"]["pr_auc"],
                "wf_brier_mean": res["walk_forward"]["mean_metrics"]["brier"],

                "wf_roc_auc_std": res["walk_forward"]["std_metrics"]["roc_auc"],
                "wf_pr_auc_std": res["walk_forward"]["std_metrics"]["pr_auc"],
                "wf_brier_std": res["walk_forward"]["std_metrics"]["brier"],

                "test_raw_roc_auc": res["test_metrics_raw"]["roc_auc"],
                "test_raw_pr_auc": res["test_metrics_raw"]["pr_auc"],
                "test_raw_brier": res["test_metrics_raw"]["brier"],

                "test_cal_roc_auc": res["test_metrics_cal"]["roc_auc"],
                "test_cal_pr_auc": res["test_metrics_cal"]["pr_auc"],
                "test_cal_brier": res["test_metrics_cal"]["brier"],

                "base_best_k": res["business_metrics"]["base_best_k"],
                "base_max_profit": res["business_metrics"]["base_max_profit"],

                "selection_score": score,
                "status": "ok",
            }
            all_results.append(row)

            if best_score is None or score > best_score:
                best_score = score
                best_result = res

            if status_callback:
                best_payload = None
                if best_result:
                    best_payload = {
                        "model_kind": best_result["params_used"]["model_kind"],
                        "pr_auc": best_result["test_metrics_cal"]["pr_auc"],
                        "brier": best_result["test_metrics_cal"]["brier"],
                    }

                status_callback(
                    stage="model_selection",
                    progress=progress,
                    extra={
                        "last_finished": {
                            "model_kind": model_kind,
                            "pr_auc": res["test_metrics_cal"]["pr_auc"],
                            "brier": res["test_metrics_cal"]["brier"],
                        },
                        "best_so_far": best_payload,
                    }
                )

        except Exception as e:
            all_results.append({
                "experiment": exp_name,
                "horizon_days": int(h),
                "history_days": int(w),
                "step_days": int(s),
                "model_kind": model_kind,
                "model_params": json.dumps(model_params, ensure_ascii=False, sort_keys=True),
                "calibration": calibration,
                "status": "failed",
                "error": str(e),
            })

    if not all_results:
        raise ValueError("No experiments were executed.")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "experiment_results.csv", index=False)

    if best_result is None:
        raise ValueError("All experiments failed. Check experiment_results.csv")

    return {
        "mode": "grid_search",
        "selection_metric": selection_metric,
        "best_result": best_result,
        "all_results_csv": str(out_dir / "experiment_results.csv"),
        "n_experiments": int(len(results_df)),
    }


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

    df_raw = pd.read_csv(input_csv)

    if status_callback:
        status_callback(stage="profiling_and_validation", progress=10)

    quality_report = profile_dataset(df_raw, template)
    (out / "inputs").mkdir(exist_ok=True)
    with open(out / "inputs" / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    df = df_raw.rename(columns={v: k for k, v in mapping.items() if v}).copy()

    basic_validate(df, template=template)

    df = canonicalize_types(df, template)

    write_datasheet(df_raw, template, out / "datasheet.md")
    df.to_parquet(out / "inputs" / "canonical_input.parquet", index=False)

    params_local = dict(params)
    params_local["mapping_used"] = mapping

    if status_callback:
        status_callback(stage="experiment_grid", progress=30)

    has_grid = any(
        key in params_local
        for key in ["horizon_days_grid", "history_days_grid", "step_days_grid", "model_kind_grid"]
    )

    if has_grid:
        grid_res = run_experiment_grid(
            df,
            template,
            params_local,
            out,
            status_callback=status_callback,
        )
        best_result = grid_res["best_result"]

        if status_callback:
            status_callback(stage="finalizing_artifacts", progress=90)

        return {
            "template": template,
            "mode": "grid_search",
            "quality_report_path": str(out / "inputs" / "quality_report.json"),
            "n_experiments": grid_res["n_experiments"],
            "selection_metric": grid_res["selection_metric"],
            "all_results_csv": grid_res["all_results_csv"],
            **best_result,
        }

    result = run_single_experiment(df, template, params_local, out / "single_run")
    result["quality_report_path"] = str(out / "inputs" / "quality_report.json")

    if status_callback:
        status_callback(stage="finalizing_artifacts", progress=90)

    return result


def run_scoring_pipeline(
    input_csv: str,
    bundle_dir: str,
    out_dir: str,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle(bundle_dir)
    mapping = bundle["mapping"]
    template = bundle["config"]["template"]
    params = bundle["config"]["params"]
    feature_cols = bundle["feature_cols"]

    df_raw = pd.read_csv(input_csv)

    df = df_raw.rename(columns={v: k for k, v in mapping.items() if v}).copy()

    basic_validate(df, template=template)
    df = canonicalize_types(df, template)

    snap_cfg = SnapshotConfig(
        template=template,
        horizon_days=int(params.get("horizon_days", 30)),
        history_days=int(params.get("history_days", 180)),
        step_days=int(params.get("step_days", 30)),
        min_events_in_history=int(params.get("min_events_in_history", 1)),
        min_lifetime_days=int(params.get("min_lifetime_days", 0)),
        max_recency_days=params.get("max_recency_days", None),
    )

    snaps = build_latest_snapshot(df, snap_cfg)
    if snaps.empty:
        raise ValueError("No snapshot rows for scoring")

    X = _prepare_feature_matrix(snaps, feature_cols)

    model = bundle["model"]
    calibrator = bundle["calibrator"]

    p_raw = model.predict_proba(X)[:, 1]
    p_cal = calibrator.predict_proba(X)[:, 1]

    scored = snaps.copy()
    scored["p_raw"] = p_raw
    scored["p_calibrated"] = p_cal

    try:
        scored["risk_segment"] = pd.qcut(
            scored["p_calibrated"],
            q=4,
            labels=["low", "medium", "high", "critical"],
            duplicates="drop",
        )
    except Exception:
        scored["risk_segment"] = "unknown"

    base = next(s for s in DEFAULT_SCENARIOS if s.name == "base")
    V = scored["value_proxy"].astype(float).values
    scored["EV"] = expected_value(scored["p_calibrated"].values, V, base)

    out_csv = out / "scored_clients.csv"
    scored.to_csv(out_csv, index=False)

    return {
        "n_rows": int(len(scored)),
        "output_csv": str(out_csv),
        "columns": list(scored.columns),
    }



'''import json
import os
from io import BytesIO
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Churn/Retention Self-Serve",
    layout="wide",
)

st.title("Churn / Retention Self-Serve")

def api_post(url: str, data=None, files=None, timeout=180):
    return requests.post(url, data=data, files=files, timeout=timeout)


def api_get(url: str, timeout=60):
    return requests.get(url, timeout=timeout)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def read_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(raw_bytes))


def detect_id_like_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for col in df.columns:
        name = str(col).lower()

        if any(x in name for x in [
            "id", "customer", "client", "account", "subject",
            "invoice", "order", "transaction", "no"
        ]):
            out.append(col)
            continue

        try:
            nunique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
            if nunique_ratio > 0.95:
                out.append(col)
        except Exception:
            pass

    return list(dict.fromkeys(out))


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    id_like = set(detect_id_like_columns(df))

    for col in df.columns:
        if col in id_like:
            continue

        name = str(col).lower()

        if any(x in name for x in ["date", "time", "timestamp", "datetime"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                valid_share = parsed.notna().mean()
                if valid_share > 0.7:
                    years = parsed.dropna().dt.year
                    if not years.empty and years.between(2000, 2100).mean() > 0.8:
                        candidates.append(col)
                        continue
            except Exception:
                pass

        if df[col].dtype == "object":
            try:
                sample = df[col].dropna().astype(str).head(500)
                parsed = pd.to_datetime(sample, errors="coerce")
                valid_share = parsed.notna().mean()

                if valid_share > 0.8:
                    years = parsed.dropna().dt.year
                    if not years.empty and years.between(2000, 2100).mean() > 0.8:
                        candidates.append(col)
            except Exception:
                pass

    return list(dict.fromkeys(candidates))


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    out = []
    id_like = set(detect_id_like_columns(df))

    for col in df.columns:
        if col in id_like:
            continue

        try:
            s = pd.to_numeric(df[col], errors="coerce")
            valid_share = s.notna().mean()
            if valid_share < 0.8:
                continue

            nunique_ratio = s.nunique(dropna=True) / max(len(s.dropna()), 1)
            if nunique_ratio > 0.95:
                continue

            out.append(col)
        except Exception:
            pass

    return out


def render_quality_overview(quality: Dict[str, Any]):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк", quality.get("n_rows"))
    c2.metric("Колонок", quality.get("n_cols"))
    c3.metric("Дубликаты", quality.get("duplicates_full_rows"))
    c4.metric("Warnings", len(quality.get("warnings", [])))

    st.subheader("Пропуски по колонкам")
    null_df = pd.DataFrame({
        "column": list(quality.get("nulls_by_column", {}).keys()),
        "nulls": list(quality.get("nulls_by_column", {}).values()),
        "null_share": list(quality.get("null_share_by_column", {}).values()),
    })
    if not null_df.empty:
        st.dataframe(null_df, use_container_width=True)
    else:
        st.info("Нет данных о пропусках.")

    if quality.get("date_ranges"):
        st.subheader("Диапазоны дат")
        st.json(quality["date_ranges"])

    if quality.get("warnings"):
        st.subheader("Warnings")
        for w in quality["warnings"]:
            st.warning(w)


def render_transactions_overview(df: pd.DataFrame):
    st.subheader("Готовность transactional-данных")

    id_candidates = detect_id_like_columns(df)
    date_candidates = detect_date_columns(df)
    num_candidates = detect_numeric_columns(df)

    customer_col = next((c for c in df.columns if "customer" in c.lower()), id_candidates[0] if id_candidates else None)
    tx_col = next((c for c in df.columns if "invoice" in c.lower() or "transaction" in c.lower()), None)
    date_col = date_candidates[0] if date_candidates else None
    amount_col = next(
        (c for c in num_candidates if any(x in c.lower() for x in ["amount", "revenue", "price"])),
        num_candidates[0] if num_candidates else None
    )

    unique_customers = int(df[customer_col].nunique()) if customer_col and customer_col in df.columns else None
    unique_tx = int(df[tx_col].nunique()) if tx_col and tx_col in df.columns else None

    median_tx_per_customer = None
    share_customers_2plus = None
    if customer_col and tx_col:
        tx_per_cust = df.groupby(customer_col)[tx_col].nunique()
        median_tx_per_customer = float(tx_per_cust.median()) if not tx_per_cust.empty else None
        share_customers_2plus = float((tx_per_cust >= 2).mean()) if not tx_per_cust.empty else None

    date_range_str = "—"
    if date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not parsed.empty:
            date_range_str = f"{parsed.min().date()} — {parsed.max().date()}"

    c1, c2, c3 = st.columns(3)
    c1.metric("Строк", len(df))
    c2.metric("Клиентов", unique_customers if unique_customers is not None else "—")
    c3.metric("Транзакций", unique_tx if unique_tx is not None else "—")

    c4, c5, c6 = st.columns(3)
    c4.metric("Период", date_range_str)
    c5.metric("Median tx/client", round(median_tx_per_customer, 2) if median_tx_per_customer is not None else "—")
    c6.metric("Клиентов с 2+ покупками", f"{share_customers_2plus * 100:.1f}%" if share_customers_2plus is not None else "—")

    left, right = st.columns(2)

    with left:
        st.subheader("Транзакции по месяцам")
        if date_col:
            parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not parsed.empty:
                ts = parsed.dt.to_period("M").astype(str).value_counts().sort_index()
                st.bar_chart(ts)
            else:
                st.info("Не удалось построить график по времени.")
        else:
            st.info("Дата-колонка не распознана.")

    with right:
        st.subheader("Выручка по месяцам")
        if date_col and amount_col:
            tmp = df[[date_col, amount_col]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
            tmp = tmp.dropna()
            if not tmp.empty:
                tmp["month"] = tmp[date_col].dt.to_period("M").astype(str)
                monthly = tmp.groupby("month")[amount_col].sum().sort_index()
                st.bar_chart(monthly)
            else:
                st.info("Не удалось построить график выручки.")
        else:
            st.info("Не удалось определить колонки даты и суммы.")

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Покупок на клиента")
        if customer_col and tx_col:
            tx_per_cust = df.groupby(customer_col)[tx_col].nunique()
            counts, edges = np.histogram(tx_per_cust.values, bins=20)
            hist_df = pd.DataFrame({"bin_left": edges[:-1], "count": counts})
            st.bar_chart(hist_df.set_index("bin_left")["count"])
        else:
            st.info("Не удалось определить customer_id / transaction_id.")

    with right2:
        st.subheader("Выручка на клиента")
        if customer_col and amount_col:
            tmp = df[[customer_col, amount_col]].copy()
            tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
            tmp = tmp.dropna()
            if not tmp.empty:
                rev_per_cust = tmp.groupby(customer_col)[amount_col].sum()
                counts, edges = np.histogram(rev_per_cust.values, bins=20)
                hist_df = pd.DataFrame({"bin_left": edges[:-1], "count": counts})
                st.bar_chart(hist_df.set_index("bin_left")["count"])
            else:
                st.info("Нет валидных значений для выручки на клиента.")
        else:
            st.info("Не удалось определить customer_id / amount.")


def render_subscriptions_overview(df: pd.DataFrame):
    st.subheader("Готовность subscription-данных")
    st.dataframe(df.head(20), use_container_width=True)


def render_events_overview(df: pd.DataFrame):
    st.subheader("Готовность event-данных")
    st.dataframe(df.head(20), use_container_width=True)


def get_unused_columns(all_columns: List[str], mapping_result: Dict[str, Any]) -> List[str]:
    used = {v for v in mapping_result.values() if v}
    return [c for c in all_columns if c not in used]


def risk_segment_preview(df: pd.DataFrame):
    if "risk_segment" in df.columns:
        st.subheader("Распределение risk_segment")
        st.bar_chart(df["risk_segment"].astype(str).value_counts())


CONTRACTS = {
    "transactions": {
        "required": {
            "customer_id": "Идентификатор клиента",
            "transaction_id": "Идентификатор транзакции/чека",
            "event_time": "Дата и время транзакции",
            "amount": "Денежная сумма транзакции",
        },
        "optional": {
            "item_id": "Идентификатор товара",
            "quantity": "Количество",
            "country": "Страна/регион",
            "unit_price": "Цена единицы",
            "is_cancellation": "Флаг отмены/возврата",
            "promo_code": "Промокод",
        }
    },
    "subscriptions": {
        "required": {
            "account_id": "Идентификатор аккаунта",
            "period_start": "Начало периода",
            "period_end": "Конец периода",
            "mrr": "Регулярная выручка",
            "subscription_status": "Статус подписки",
        },
        "optional": {
            "churn_date": "Дата оттока",
            "plan_name": "Название тарифа",
            "seats_purchased": "Куплено мест",
            "seats_used": "Использовано мест",
        }
    },
    "events": {
        "required": {
            "subject_id": "Идентификатор пользователя/субъекта",
            "event_time": "Дата и время события",
            "event_name": "Название события",
        },
        "optional": {
            "account_id": "Идентификатор аккаунта",
            "platform": "Платформа/устройство",
        }
    }
}


# =========================
# session init
# =========================

if "working_csv_bytes" not in st.session_state:
    st.session_state["working_csv_bytes"] = None

if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None

if "inspect_data" not in st.session_state:
    st.session_state["inspect_data"] = None

if "mapping_result" not in st.session_state:
    st.session_state["mapping_result"] = {}

if "extra_feature_columns" not in st.session_state:
    st.session_state["extra_feature_columns"] = []

if "job_info" not in st.session_state:
    st.session_state["job_info"] = {}

if "job_result" not in st.session_state:
    st.session_state["job_result"] = None

if "job_status" not in st.session_state:
    st.session_state["job_status"] = None

if "models_cache" not in st.session_state:
    st.session_state["models_cache"] = None

if "score_result" not in st.session_state:
    st.session_state["score_result"] = None


# =========================
# tabs
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Готовность данных",
    "2. Контракт и схема",
    "3. Обучение",
    "4. Мониторинг и результаты",
    "5. Модели и прогноз",
])


# =========================
# TAB 1
# =========================

with tab1:
    st.header("Готовность данных")

    template = st.selectbox(
        "Шаблон данных",
        ["transactions", "subscriptions", "events"],
        key="template",
    )

    uploaded = st.file_uploader(
        "Загрузите CSV",
        type=["csv"],
        key="uploaded_file",
    )

    if uploaded is not None:
        current_bytes = uploaded.getvalue()
        current_name = uploaded.name

        if (
            st.session_state["working_csv_bytes"] is None
            or st.session_state["uploaded_file_name"] != current_name
        ):
            st.session_state["working_csv_bytes"] = current_bytes
            st.session_state["uploaded_file_name"] = current_name
            st.session_state["inspect_data"] = None
            st.session_state["mapping_result"] = {}

    if st.session_state["working_csv_bytes"] is not None:
        raw_df = read_csv_bytes(st.session_state["working_csv_bytes"])

        dup_count = int(raw_df.duplicated().sum())
        if dup_count > 0:
            st.warning(f"Найдено полных дубликатов строк: {dup_count}")
            if st.button("Удалить полные дубликаты для текущей сессии"):
                raw_df = raw_df.drop_duplicates().copy()
                st.session_state["working_csv_bytes"] = df_to_csv_bytes(raw_df)
                st.session_state["inspect_data"] = None
                st.session_state["mapping_result"] = {}
                st.success("Полные дубликаты удалены из рабочей версии датасета.")

        try:
            if template == "transactions":
                render_transactions_overview(raw_df)
            elif template == "subscriptions":
                render_subscriptions_overview(raw_df)
            else:
                render_events_overview(raw_df)
        except Exception as e:
            st.error(f"Ошибка при построении обзора данных: {e}")

        st.subheader("Preview")
        st.dataframe(raw_df.head(20), use_container_width=True)

        if st.button("Подготовить quality report и suggested mapping"):
            files = {
                "file": (
                    st.session_state["uploaded_file_name"],
                    st.session_state["working_csv_bytes"],
                    "text/csv",
                )
            }
            resp = api_post(
                f"{API_URL}/inspect",
                data={"template": template},
                files=files,
                timeout=180,
            )
            if resp.status_code == 200:
                st.session_state["inspect_data"] = resp.json()
                st.success("Файл проанализирован.")
            else:
                st.error(resp.text)

    if st.session_state["inspect_data"]:
        st.divider()
        st.subheader("Quality report")
        try:
            render_quality_overview(st.session_state["inspect_data"]["quality_report"])
        except Exception as e:
            st.error(f"Ошибка при отображении quality report: {e}")


# =========================
# TAB 2
# =========================

with tab2:
    st.header("Контракт и схема")

    inspect_data = st.session_state.get("inspect_data")
    current_template = st.session_state.get("template", "transactions")

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Обязательные поля")
        req_df = pd.DataFrame({
            "canonical_field": list(CONTRACTS[current_template]["required"].keys()),
            "description": list(CONTRACTS[current_template]["required"].values()),
        })
        st.dataframe(req_df, use_container_width=True)

        st.subheader("Рекомендованные поля")
        opt_df = pd.DataFrame({
            "canonical_field": list(CONTRACTS[current_template]["optional"].keys()),
            "description": list(CONTRACTS[current_template]["optional"].values()),
        })
        st.dataframe(opt_df, use_container_width=True)

        st.info(
            "Дополнительные пользовательские колонки можно сохранить в схеме проекта, "
            "но они не будут автоматически включены в модель без отдельной проверки на утечки и пригодность."
        )

    with right:
        if not inspect_data:
            st.info("Сначала проанализируйте файл на первой странице.")
        else:
            columns = inspect_data["columns"]
            suggested = inspect_data["suggested_mapping"]

            st.subheader("Сопоставление колонок")
            mapping_result = {}

            for canon, suggested_col in suggested.items():
                options = ["-- не выбрано --"] + columns
                default_idx = options.index(suggested_col) if suggested_col in options else 0
                selected = st.selectbox(
                    canon,
                    options,
                    index=default_idx,
                    key=f"map_{canon}",
                )
                mapping_result[canon] = None if selected == "-- не выбрано --" else selected

            st.session_state["mapping_result"] = mapping_result

            st.divider()
            st.subheader("Дополнительные колонки")
            unused_columns = get_unused_columns(columns, mapping_result)

            extra_feature_columns = st.multiselect(
                "Сохранить дополнительные колонки в схеме проекта",
                options=unused_columns,
                default=st.session_state.get("extra_feature_columns", []),
                key="extra_columns_select",
            )
            st.session_state["extra_feature_columns"] = extra_feature_columns

            st.subheader("Итоговая схема")
            st.json({
                "mapping": mapping_result,
                "extra_feature_columns": extra_feature_columns,
            })


# =========================
# TAB 3
# =========================

with tab3:
    st.header("Обучение")

    inspect_data = st.session_state.get("inspect_data")
    mapping_result = st.session_state.get("mapping_result", {})

    if inspect_data is None or not mapping_result:
        st.info("Сначала загрузите файл и заполните схему.")
    else:
        st.subheader("Полный автоматический анализ")
        st.caption(
            "Система сама выполнит полный перебор разумных горизонтов, моделей и калибровок, "
            "сравнит качество, устойчивость и бизнес-эффект, а затем выберет лучший вариант."
        )

        model_name = st.text_input(
            "Название проекта / модели",
            value="retention_model_full_analysis",
            key="model_name",
        )

        params = {
            "horizon_days_grid": [30, 60, 90],
            "history_days_grid": [180, 365],
            "step_days_grid": [30],
            "model_kind_grid": ["logreg", "sklearn_gbdt", "lightgbm", "catboost"],
            "calibration_grid": ["sigmoid", "isotonic"],
            "selection_metric": "pr_auc",
            "use_class_weight": True,
            "model_name": model_name,
            "extra_feature_columns": st.session_state.get("extra_feature_columns", []),
        }

        with st.expander("Дополнительные настройки"):
            params["horizon_days_grid"] = st.multiselect(
                "Горизонты прогноза",
                [30, 60, 90, 120],
                default=params["horizon_days_grid"],
            )
            params["history_days_grid"] = st.multiselect(
                "Окно истории",
                [90, 180, 365],
                default=params["history_days_grid"],
            )
            params["step_days_grid"] = st.multiselect(
                "Шаг anchor",
                [7, 14, 30],
                default=params["step_days_grid"],
            )

        if st.button("Запустить полный анализ"):
            if st.session_state["working_csv_bytes"] is None:
                st.error("Нет рабочего CSV. Перезагрузите файл на первой странице.")
            else:
                files = {
                    "file": (
                        st.session_state["uploaded_file_name"],
                        st.session_state["working_csv_bytes"],
                        "text/csv",
                    )
                }

                payload = {
                    "template": st.session_state["template"],
                    "mapping_json": json.dumps(mapping_result, ensure_ascii=False),
                    "params_json": json.dumps(params, ensure_ascii=False),
                }

                resp = api_post(
                    f"{API_URL}/jobs",
                    data=payload,
                    files=files,
                    timeout=180,
                )

                if resp.status_code == 200:
                    st.session_state["job_info"] = resp.json()
                    st.success(resp.json())
                else:
                    st.error(resp.text)

        if st.session_state.get("job_info"):
            st.subheader("Текущая job")
            st.json(st.session_state["job_info"])


# =========================
# TAB 4
# =========================

with tab4:
    st.header("Мониторинг и результаты")

    default_job_id = st.session_state.get("job_info", {}).get("job_id", "")
    job_id = st.text_input("job_id", value=default_job_id, key="result_job_id")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Проверить статус"):
            if not job_id:
                st.warning("Введите job_id")
            else:
                resp = api_get(f"{API_URL}/jobs/{job_id}")
                if resp.status_code == 200:
                    status_data = resp.json()
                    st.session_state["job_status"] = status_data
                else:
                    st.error(resp.text)

    with c2:
        if st.button("Загрузить результат"):
            if not job_id:
                st.warning("Введите job_id")
            else:
                resp = api_get(f"{API_URL}/jobs/{job_id}/result")
                if resp.status_code == 200:
                    st.session_state["job_result"] = resp.json()
                else:
                    st.error(resp.text)

    if st.session_state.get("job_status"):
        status_data = st.session_state["job_status"]

        st.subheader("Статус обучения")

        c1, c2, c3 = st.columns(3)
        c1.metric("Status", status_data.get("status", "—"))
        c2.metric("Stage", status_data.get("stage", "—"))
        c3.metric("Progress", f"{status_data.get('progress', 0)}%")

        progress = status_data.get("progress")
        if progress is not None:
            st.progress(int(progress))

        current_exp = status_data.get("current_experiment")
        if current_exp:
            st.subheader("Текущий эксперимент")
            st.json(current_exp)

        best_so_far = status_data.get("best_so_far")
        if best_so_far:
            st.subheader("Лучший результат на текущий момент")
            st.json(best_so_far)

        last_finished = status_data.get("last_finished")
        if last_finished:
            st.subheader("Последний завершённый эксперимент")
            st.json(last_finished)

    result = st.session_state.get("job_result")
    if result:
        st.divider()
        st.subheader("Победившая конфигурация")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Template", result.get("template", "—"))
        c2.metric("Snapshots", result.get("n_snapshots", "—"))
        c3.metric("Features", result.get("feature_count", "—"))
        c4.metric("Best k", result.get("business_metrics", {}).get("base_best_k", "—"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("PR-AUC", round(result.get("test_metrics_cal", {}).get("pr_auc", 0), 4))
        c6.metric("ROC-AUC", round(result.get("test_metrics_cal", {}).get("roc_auc", 0), 4))
        c7.metric("Brier", round(result.get("test_metrics_cal", {}).get("brier", 0), 4))
        c8.metric("MaxProfit", round(result.get("business_metrics", {}).get("base_max_profit", 0), 2))

        st.subheader("Параметры победителя")
        st.json(result.get("params_used", {}))

        st.subheader("Калибровка")
        st.json(result.get("calibration", {}))

        st.subheader("Walk-forward")
        st.json(result.get("walk_forward", {}))

        st.subheader("Артефакты")
        st.json(result.get("artifacts", {}))

        download_job_id = st.text_input("job_id для скачивания ZIP", value=job_id, key="download_job_id")
        if st.button("Скачать report.zip"):
            r = api_get(f"{API_URL}/jobs/{download_job_id}/download", timeout=120)
            if r.status_code == 200:
                st.download_button(
                    "Download report.zip",
                    data=r.content,
                    file_name="report.zip",
                )
            else:
                st.error(r.text)


# =========================
# TAB 5
# =========================

with tab5:
    st.header("Модели и прогноз")

    if st.button("Обновить список моделей"):
        resp = api_get(f"{API_URL}/models")
        if resp.status_code == 200:
            st.session_state["models_cache"] = resp.json()["models"]
        else:
            st.error(resp.text)

    models = st.session_state.get("models_cache")
    if models:
        models_df = pd.DataFrame(models)
        st.subheader("Реестр моделей")
        st.dataframe(models_df, use_container_width=True)

        model_options = {
            f"{m['job_id']} | {m.get('template')} | {m.get('params_used', {}).get('model_kind', '')}": m
            for m in models
        }

        selected_label = st.selectbox(
            "Выберите модель",
            list(model_options.keys()),
        )
        selected_model = model_options[selected_label]

        st.subheader("Детали модели")
        st.json(selected_model)

        st.divider()
        st.subheader("Прогноз на новых данных")

        score_file = st.file_uploader(
            "Загрузите CSV для scoring",
            type=["csv"],
            key="score_file",
        )

        if score_file is not None and st.button("Запустить прогноз"):
            files = {"file": (score_file.name, score_file.getvalue(), "text/csv")}
            resp = api_post(
                f"{API_URL}/score",
                data={"bundle_dir": selected_model["bundle_dir"]},
                files=files,
                timeout=180,
            )

            if resp.status_code == 200:
                score_result = resp.json()
                st.session_state["score_result"] = score_result
            else:
                st.error(resp.text)

    score_result = st.session_state.get("score_result")
    if score_result:
        st.subheader("Результат прогноза")
        st.json({
            "n_rows": score_result.get("n_rows"),
            "output_csv": score_result.get("output_csv"),
        })

        preview_df = pd.DataFrame(score_result.get("preview", []))
        if not preview_df.empty:
            st.dataframe(preview_df, use_container_width=True)
            risk_segment_preview(preview_df)

'''







