# app/pipeline.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

import joblib
import numpy as np
import pandas as pd


from app.mapping import sanitize_extra_feature_columns
from churnlib.calibration_module import CalibrationConfig, calibrate, save_calibration_plot, save_calibration_summary
from churnlib.cards_module import write_datasheet, write_model_card
from churnlib.data_module import SnapshotConfig, build_latest_snapshot, build_snapshots, canonicalize_types
from churnlib.drift_module import compute_feature_psi
from churnlib.economy_module import DEFAULT_SCENARIOS
from churnlib.explain_module import ExplainConfig, shap_explain_global, save_shap_artifacts
from churnlib.model_module import TrainConfig, train_time_cv, walk_forward_backtest, prepare_X
from churnlib.report_module import ReportConfig, write_docx_report, write_reports
from churnlib.validation_module import basic_validate, assess_suitability

def _build_quality_payload(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    fpr, tpr, _ = roc_curve(y_true, p)
    precision, recall, _ = precision_recall_curve(y_true, p)

    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred).tolist()

    return {
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
        },
        "pr_curve": {
            "precision": [float(x) for x in precision],
            "recall": [float(x) for x in recall],
        },
        "confusion_matrix": cm,
        "threshold": float(threshold),
        "target_rate": float(np.mean(y_true)),
    }

def assess_training_readiness(df: pd.DataFrame, template: str) -> Dict[str, Any]:
    verdict = "ready"
    reasons = []

    if template == "transactions":
        date_col = None
        for col in df.columns:
            if "date" in str(col).lower() or "time" in str(col).lower():
                date_col = col
                break

        if date_col is None:
            verdict = "partial"
            reasons.append("Не найдена колонка с датой.")

    if not reasons:
        reasons.append("Явных стоп-факторов не найдено.")

    return {
        "verdict": verdict,
        "reasons": reasons,
    }

    
def _detect_date_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        s = df[col]
        try:
            parsed = pd.to_datetime(s, errors="coerce")
            valid_share = parsed.notna().mean()
            if valid_share > 0.7:
                years = parsed.dropna().dt.year
                if not years.empty and years.between(2000, 2100).mean() > 0.8:
                    out.append(col)
            elif any(x in str(col).lower() for x in ["date", "time", "timestamp", "datetime"]):
                out.append(col)
        except Exception:
            pass
    return list(dict.fromkeys(out))


def _detect_id_like_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        name = str(col).lower()
        if any(x in name for x in ["id", "customer", "client", "account", "subject", "invoice", "order", "transaction", "no"]):
            out.append(col)
            continue
        try:
            nunique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
            if nunique_ratio > 0.95:
                out.append(col)
        except Exception:
            pass
    return list(dict.fromkeys(out))


def _detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    out = []
    id_like = set(_detect_id_like_columns(df))
    for col in df.columns:
        if col in id_like:
            continue
        try:
            s = pd.to_numeric(df[col], errors="coerce")
            valid_share = s.notna().mean()
            if valid_share >= 0.7:
                out.append(col)
        except Exception:
            pass
    return out


def profile_dataset(df: pd.DataFrame, template: str) -> Dict[str, Any]:
    nulls = df.isna().sum().to_dict()
    null_share = (df.isna().mean().round(6)).to_dict()

    date_candidates = _detect_date_columns(df)
    id_candidates = _detect_id_like_columns(df)
    num_candidates = _detect_numeric_columns(df)

    date_ranges: Dict[str, Dict[str, str]] = {}
    for col in date_candidates:
        parsed = pd.to_datetime(df[col], errors="coerce")
        parsed = parsed.dropna()
        if not parsed.empty:
            date_ranges[col] = {
                "min": str(parsed.min()),
                "max": str(parsed.max()),
                "non_null_share": f"{parsed.notna().mean():.3f}",
            }

    warnings = []
    if len(df) < 1000:
        warnings.append("Мало строк: для устойчивого обучения желательно больше наблюдений.")
    if df.duplicated().sum() > 0:
        warnings.append("Есть полные дубликаты строк.")
    if max(null_share.values(), default=0) > 0.5:
        warnings.append("Есть колонки с большим количеством пропусков.")

    diagnostics = {
        "date_candidates": date_candidates,
        "id_candidates": id_candidates,
        "numeric_candidates": num_candidates,
    }

    readiness = assess_training_readiness(df, template)

    return {
        "template": template,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "duplicates_full_rows": int(df.duplicated().sum()),
        "nulls_by_column": nulls,
        "null_share_by_column": null_share,
        "date_ranges": date_ranges,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "readiness": readiness,
    }

def _prepare_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # Используем ту же логику, что и churnlib.model_module.prepare_X
    return prepare_X(df, feature_cols)


def save_model_bundle(
    bundle_dir: Path,
    model,
    calibrator,
    feature_cols: List[str],
    mapping: Dict[str, str],
    template: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, bundle_dir / "model.joblib")
    joblib.dump(calibrator, bundle_dir / "calibrator.joblib")

    (bundle_dir / "feature_cols.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (bundle_dir / "mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (bundle_dir / "config.json").write_text(
        json.dumps(
            {"template": template, "params": params},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"bundle_dir": str(bundle_dir)}


def load_model_bundle(bundle_dir: str) -> Dict[str, Any]:
    d = Path(bundle_dir)
    model = joblib.load(d / "model.joblib")
    calibrator = joblib.load(d / "calibrator.joblib")
    feature_cols = json.loads((d / "feature_cols.json").read_text(encoding="utf-8"))
    mapping = json.loads((d / "mapping.json").read_text(encoding="utf-8"))
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    return {
        "model": model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "mapping": mapping,
        "config": config,
    }


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _grid_values(params: Dict[str, Any], key: str, default: List[Any]) -> List[Any]:
    v = params.get(key, default)
    return _as_list(v) if not isinstance(v, str) else [v]


def choose_metric(result: Dict[str, Any], metric_name: str) -> float:
    """
    Для brier — меньше лучше, поэтому возвращаем -brier.
    Для остальных — больше лучше.
    """
    if metric_name == "base_max_profit":
        return float(result.get("business_metrics", {}).get("base_max_profit", -1e18))

    # calibration test metrics
    tm = result.get("test_metrics_cal", {})
    if metric_name in tm:
        v = float(tm[metric_name])
        return -v if metric_name == "brier" else v

    # cv metrics
    cv = result.get("cv_metrics_mean", {})
    if metric_name in cv:
        v = float(cv[metric_name])
        return -v if metric_name == "brier" else v

    wf = result.get("walk_forward", {}).get("mean_metrics", {})
    if metric_name in wf:
        v = float(wf[metric_name])
        return -v if metric_name == "brier" else v

    raise ValueError(f"Unknown metric for selection: {metric_name}")


def run_single_experiment(
    df: pd.DataFrame,
    template: str,
    mapping_used: Dict[str, str],
    params: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # extra features (guarded)
    extra_cols_raw = params.get("extra_feature_cols") or params.get("extra_feature_columns") or []
    extra_cols_final, extra_audit = sanitize_extra_feature_columns(
        df=df,
        template=template,
        mapping=mapping_used,
        extra_cols=list(extra_cols_raw) if isinstance(extra_cols_raw, list) else [],
        max_cols=int(params.get("max_extra_feature_cols", 20)),
    )

    snap_cfg = SnapshotConfig(
        template=template,
        horizon_days=int(params.get("horizon_days", 30)),
        history_days=int(params.get("history_days", 180)),
        step_days=int(params.get("step_days", 30)),
        min_events_in_history=int(params.get("min_events_in_history", 1)),
        min_lifetime_days=int(params.get("min_lifetime_days", 0)),
        max_recency_days=params.get("max_recency_days", None),
        tz=params.get("tz", None),
        extra_feature_cols=extra_cols_final,
    )

    snaps = build_snapshots(df, snap_cfg)
    if snaps.empty:
        raise ValueError(
            f"Empty snapshots for horizon={snap_cfg.horizon_days}, "
            f"history={snap_cfg.history_days}, step={snap_cfg.step_days}"
        )

    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    snaps.to_parquet(out_dir / "snapshots.parquet", index=False)

    target_rate = float(snaps["target"].mean()) if len(snaps) else None
    class_counts = snaps["target"].value_counts(dropna=False).to_dict()

    feature_cols = [c for c in snaps.columns if c not in ["entity_id", "anchor_time", "target"]]

    # train
    train_cfg = TrainConfig(
        model_kind=str(params.get("model_kind", "lightgbm")),
        use_class_weight=bool(params.get("use_class_weight", False)),
        model_params=params.get("model_params", {}),
    )
    train_res = train_time_cv(snaps, feature_cols, train_cfg)
    backtest_res = walk_forward_backtest(snaps, feature_cols, train_cfg)
    backtest_res["folds_df"].to_csv(out_dir / "tables" / "walk_forward_folds.csv", index=False)

    # calibration
    cal_cfg = CalibrationConfig(method=str(params.get("calibration", "sigmoid")), cv=3)

    X_train = _prepare_feature_matrix(train_res["train_df"], feature_cols)
    X_test = _prepare_feature_matrix(train_res["test_df"], feature_cols)
    y_train = train_res["train_df"]["target"].astype(int).values
    y_test = train_res["test_df"]["target"].astype(int).values

    cal_res = calibrate(train_res["model"], X_train, y_train, X_test, y_test, cal_cfg)

    save_calibration_plot(
        y_true=y_test,
        p_raw=cal_res["p_raw"],
        p_cal=cal_res["p_cal"],
        path=str(out_dir / "plots" / "calibration_curve.png"),
    )
    save_calibration_summary(cal_res, str(out_dir / "tables" / "calibration_summary.json"))

    # quality payload for UI
    quality_payload = _build_quality_payload(
        y_true=y_test,
        p=np.asarray(cal_res["p_cal"], dtype=float),
        threshold=float(params.get("decision_threshold", 0.5)),
    )

    # scored test set
    test_scored = train_res["test_df"].copy()
    test_scored["p_raw"] = np.asarray(cal_res["p_raw"], dtype=float)
    test_scored["p_cal"] = np.asarray(cal_res["p_cal"], dtype=float)
    test_scored["pred"] = (
        test_scored["p_cal"] >= float(params.get("decision_threshold", 0.5))
    ).astype(int)
    test_scored_path = out_dir / "tables" / "test_scored.csv"
    test_scored.to_csv(test_scored_path, index=False)

    # drift PSI
    psi_path = None
    try:
        psi_df = compute_feature_psi(train_res["train_df"], train_res["test_df"], feature_cols)
        psi_path = out_dir / "tables" / "feature_psi.csv"
        psi_df.to_csv(psi_path, index=False)
    except Exception:
        psi_path = None

    # business report
    rep_cfg = ReportConfig(out_dir=str(out_dir), top_k=int(params.get("top_k_priority", 500)))
    business_rep = write_reports(
        df_test=train_res["test_df"],
        p=np.asarray(cal_res["p_cal"], dtype=float),
        scenarios=list(DEFAULT_SCENARIOS),
        cfg=rep_cfg,
    )

    # SHAP
    shap_artifacts = {}
    if bool(params.get("enable_shap", False)):
        try:
            ex_cfg = ExplainConfig(
                max_background=int(params.get("shap_background", 2000)),
                max_explain=int(params.get("shap_explain", 500)),
                random_state=42,
                top_n_local=3,
            )
            X_all = _prepare_feature_matrix(snaps, feature_cols)
            explain_res = shap_explain_global(train_res["model"], X_all, ex_cfg)

            top_risk = train_res["test_df"].copy()
            top_risk["p_cal"] = cal_res["p_cal"]
            top_risk = top_risk.sort_values("p_cal", ascending=False).head(50)

            shap_artifacts = save_shap_artifacts(
                explain_res,
                out_dir / "plots",
                top_risk_df=top_risk,
            )
        except Exception:
            shap_artifacts = {}

    # bundle
    bundle_info = save_model_bundle(
        bundle_dir=out_dir / "bundle",
        model=train_res["model"],
        calibrator=cal_res["calibrator"],
        feature_cols=feature_cols,
        mapping=mapping_used,
        template=template,
        params={
            **params,
            "extra_feature_cols": extra_cols_final,
        },
    )

    suitability = assess_suitability(df, template, params=params)

    # docx report
    docx_path = str(out_dir / "report.docx")
    docx_info = {}
    try:
        docx_info = write_docx_report(
            out_path=docx_path,
            template=template,
            params_used={
                **params,
                "extra_feature_cols": extra_cols_final,
            },
            suitability=suitability,
            metrics={
                "target_rate": target_rate,
                "class_counts": class_counts,
                "cv_metrics_mean": train_res.get("cv_metrics_mean", {}),
                "cv_metrics_std": train_res.get("cv_metrics_std", {}),
                "test_metrics_raw": cal_res.get("raw_metrics", {}),
                "test_metrics_cal": cal_res.get("metrics", {}),
                "walk_forward_mean": backtest_res.get("mean_metrics", {}),
                "walk_forward_std": backtest_res.get("std_metrics", {}),
                "business_metrics": {
                    "base_best_k": business_rep.get("base_best_k"),
                    "base_max_profit": business_rep.get("base_max_profit"),
                },
            },
            artifact_paths={
                "profit_plot": business_rep.get("profit_plot"),
                "calibration_plot": str(out_dir / "plots" / "calibration_curve.png"),
                "shap_beeswarm": shap_artifacts.get("shap_beeswarm"),
                "shap_bar": shap_artifacts.get("shap_bar"),
            },
            extra_feature_audit=extra_audit,
        )
    except Exception:
        docx_info = {}

    try:
        write_model_card(
            template=template,
            params_used={
                **params,
                "extra_feature_cols": extra_cols_final,
            },
            suitability=suitability,
            metrics={
                "target_rate": target_rate,
                "class_counts": class_counts,
                "test_metrics_raw": cal_res.get("raw_metrics", {}),
                "test_metrics_cal": cal_res.get("metrics", {}),
                "cv_metrics_mean": train_res.get("cv_metrics_mean", {}),
                "cv_metrics_std": train_res.get("cv_metrics_std", {}),
                "walk_forward": backtest_res.get("mean_metrics", {}),
                "business_metrics": {
                    "base_best_k": business_rep.get("base_best_k"),
                    "base_max_profit": business_rep.get("base_max_profit"),
                },
            },
            extra_feature_audit=extra_audit,
            out_path=str(out_dir / "model_card.md"),
        )
    except Exception:
        pass

    try:
        write_datasheet(
            df=snaps,
            template=template,
            out_path=str(out_dir / "datasheet.json"),
            extra={
                "mapping_used": mapping_used,
                "extra_feature_cols": extra_cols_final,
                "feature_cols": feature_cols,
                "target_rate": target_rate,
            },
        )
    except Exception:
        pass

    return {
        "template": template,
        "mode": "single",
        "params_used": {
            **params,
            "extra_feature_cols": extra_cols_final,
        },
        "target_rate": target_rate,
        "class_counts": class_counts,
        "feature_count": int(len(feature_cols)),
        "cv_metrics_mean": train_res.get("cv_metrics_mean", {}),
        "cv_metrics_std": train_res.get("cv_metrics_std", {}),
        "test_metrics_raw": cal_res.get("raw_metrics", {}),
        "test_metrics_cal": cal_res.get("metrics", {}),
        "quality_payload": quality_payload,
        "walk_forward": {
            "mean_metrics": backtest_res.get("mean_metrics", {}),
            "std_metrics": backtest_res.get("std_metrics", {}),
            "folds_csv": str(out_dir / "tables" / "walk_forward_folds.csv"),
        },
        "business_metrics": {
            "base_best_k": business_rep.get("base_best_k"),
            "base_max_profit": business_rep.get("base_max_profit"),
        },
        "suitability": suitability,
        "extra_feature_audit": extra_audit,
        "artifacts": {
            "bundle": bundle_info,
            "profit_plot": business_rep.get("profit_plot"),
            "profit_summary": business_rep.get("profit_summary"),
            "priority_csv": business_rep.get("priority_csv"),
            "report_docx": docx_info.get("docx_path"),
            "calibration_plot": str(out_dir / "plots" / "calibration_curve.png"),
            "test_scored_csv": str(test_scored_path),
            "feature_psi_csv": str(psi_path) if psi_path else None,
            "shap_beeswarm": shap_artifacts.get("shap_beeswarm"),
            "shap_bar": shap_artifacts.get("shap_bar"),
        },
    }


def run_experiment_grid(
    df: pd.DataFrame,
    template: str,
    mapping_used: Dict[str, str],
    params: Dict[str, Any],
    out_dir: Path,
    status_callback=None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = _grid_values(params, "horizon_days_grid", [30, 60, 90])
    histories = _grid_values(params, "history_days_grid", [180, 365])
    steps = _grid_values(params, "step_days_grid", [30])
    models = _grid_values(params, "model_kind_grid", ["lightgbm", "logreg"])
    calibrations = _grid_values(params, "calibration_grid", [params.get("calibration", "sigmoid")])

    selection_metric = str(params.get("selection_metric", "pr_auc"))

    all_rows: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    best_score: float = -1e18

    total = max(len(horizons) * len(histories) * len(steps) * len(models) * len(calibrations), 1)
    done = 0

    for h in horizons:
        for hist in histories:
            for s in steps:
                for mk in models:
                    for cal in calibrations:
                        done += 1
                        exp_name = f"h{int(h)}_hist{int(hist)}_step{int(s)}_{mk}_{cal}"
                        exp_dir = out_dir / exp_name

                        p_local = dict(params)
                        p_local.update(
                            {
                                "horizon_days": int(h),
                                "history_days": int(hist),
                                "step_days": int(s),
                                "model_kind": mk,
                                "calibration": cal,
                            }
                        )

                        if status_callback:
                            status_callback(stage="grid_search", progress=int(20 + 70 * done / total), extra={"experiment": exp_name})

                        try:
                            res = run_single_experiment(df, template, mapping_used, p_local, exp_dir)
                            score = choose_metric(res, selection_metric)
                            all_rows.append(
                                {
                                    "experiment": exp_name,
                                    "status": "ok",
                                    "selection_metric": selection_metric,
                                    "score": score,
                                    "horizon_days": int(h),
                                    "history_days": int(hist),
                                    "step_days": int(s),
                                    "model_kind": mk,
                                    "calibration": cal,
                                    "test_metrics_cal": json.dumps(res.get("test_metrics_cal", {}), ensure_ascii=False),
                                    "business_metrics": json.dumps(res.get("business_metrics", {}), ensure_ascii=False),
                                    "bundle_dir": res.get("artifacts", {}).get("bundle", {}).get("bundle_dir"),
                                }
                            )
                            if score > best_score:
                                best_score = score
                                best_result = res
                        except Exception as e:
                            all_rows.append(
                                {
                                    "experiment": exp_name,
                                    "status": "failed",
                                    "error": str(e),
                                    "horizon_days": int(h),
                                    "history_days": int(hist),
                                    "step_days": int(s),
                                    "model_kind": mk,
                                    "calibration": cal,
                                    "selection_metric": selection_metric,
                                    "score": None,
                                }
                            )

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_dir / "experiment_results.csv", index=False)

    if best_result is None:
        raise ValueError("All experiments failed. Check experiment_results.csv")

    return {
        "mode": "grid_search",
        "selection_metric": selection_metric,
        "n_experiments": int(len(results_df)),
        "all_results_csv": str(out_dir / "experiment_results.csv"),
        "best_result": best_result,
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
    (out / "inputs" / "quality_report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # переименование колонок
    df = df_raw.rename(columns={v: k for k, v in mapping.items() if v}).copy()

    # validate + canonicalize types
    basic_validate(df, template=template)
    df = canonicalize_types(df, template)

    # datasheet по сырью
    write_datasheet(df_raw, template, out / "datasheet.md")

    # сохраняем canonical input
    df.to_parquet(out / "inputs" / "canonical_input.parquet", index=False)

    params_local = dict(params)
    params_local["mapping_used"] = mapping

    # определяем режим
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
            "template": template,
            "mode": "grid_search",
            "quality_report_path": str(out / "inputs" / "quality_report.json"),
            "n_experiments": grid_res["n_experiments"],
            "selection_metric": grid_res["selection_metric"],
            "all_results_csv": grid_res["all_results_csv"],
            **best_result,
        }

    # single run
    res = run_single_experiment(df, template, mapping, params_local, out / "single_run")
    res["quality_report_path"] = str(out / "inputs" / "quality_report.json")

    if status_callback:
        status_callback(stage="finalizing_artifacts", progress=90)

    return res


def _attach_rule_based_reasons(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Простые объяснения "на человеческом языке" для клиента.
    Мы не делаем SHAP на фронте: только доп. колонки reason_1..reason_3.
    """
    def pick_reasons(row: pd.Series) -> List[str]:
        reasons: List[str] = []
        # самые типичные сигналы
        if "recency_days" in row and pd.notna(row["recency_days"]) and float(row["recency_days"]) > 60:
            reasons.append("Давно не было активности/покупок")
        if "frequency_tx" in row and pd.notna(row["frequency_tx"]) and float(row["frequency_tx"]) <= 1:
            reasons.append("Редкая активность (мало покупок в истории)")
        if "monetary" in row and pd.notna(row["monetary"]) and float(row["monetary"]) <= 0:
            reasons.append("Низкая выручка/сумма покупок")
        # fallback
        if not reasons:
            reasons.append("Риск повышен по совокупности поведения")
        return reasons[:3]

    rr = scored.apply(pick_reasons, axis=1)
    scored = scored.copy()
    scored["reason_1"] = rr.apply(lambda x: x[0] if len(x) > 0 else "")
    scored["reason_2"] = rr.apply(lambda x: x[1] if len(x) > 1 else "")
    scored["reason_3"] = rr.apply(lambda x: x[2] if len(x) > 2 else "")
    return scored


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
        template=str(template),
        horizon_days=int(params.get("horizon_days", 30)),
        history_days=int(params.get("history_days", 180)),
        step_days=int(params.get("step_days", 30)),
        min_events_in_history=int(params.get("min_events_in_history", 1)),
        min_lifetime_days=int(params.get("min_lifetime_days", 0)),
        max_recency_days=params.get("max_recency_days", None),
        tz=params.get("tz", None),
        extra_feature_cols=params.get("extra_feature_cols", []),
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

    # сегменты риска
    try:
        scored["risk_segment"] = pd.qcut(
            scored["p_calibrated"], q=4, labels=["low", "medium", "high", "critical"], duplicates="drop"
        ).astype(str)
    except Exception:
        scored["risk_segment"] = "unknown"

    # простой EV (используем сценарий base)
    base = next(s for s in DEFAULT_SCENARIOS if s.name == "base")
    V = scored["value_proxy"].astype(float).values
    scored["EV"] = V * scored["p_calibrated"].values * float(base.gain_if_save)

    scored = _attach_rule_based_reasons(scored)

    out_csv = out / "scored_clients.csv"
    scored.to_csv(out_csv, index=False)

    return {
        "template": template,
        "bundle_dir": bundle_dir,
        "output_csv": str(out_csv),
        "n_scored": int(len(scored)),
    }
