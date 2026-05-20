from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.mapping import sanitize_extra_feature_columns
from app.pipeline.bundle import save_model_bundle
from app.pipeline.features import prepare_feature_matrix
from app.pipeline.quality import build_quality_payload, choose_metric
from churnlib.calibration_module import CalibrationConfig, calibrate, save_calibration_plot, save_calibration_summary
from churnlib.cards_module import write_datasheet, write_model_card
from churnlib.data_module import SnapshotConfig, build_snapshots, canonicalize_types
from churnlib.drift_module import compute_feature_psi
from churnlib.economy_module import build_scenarios, scenario_from_params
from churnlib.explain_module import ExplainConfig, shap_explain_global, save_shap_artifacts
from churnlib.model_module import TrainConfig, train_time_cv, walk_forward_backtest
from churnlib.report_module import ReportConfig, write_docx_report, write_reports, write_training_html_report
from churnlib.validation_module import assess_suitability


def _normalize_extra_feature_config(
    df: pd.DataFrame,
    template: str,
    mapping_used: Dict[str, str],
    params: Dict[str, Any],
) -> tuple[List[str], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    raw_cfg = params.get("extra_feature_config") or {}
    raw_cols = params.get("extra_feature_cols") or params.get("extra_feature_columns") or []

    extra_cols_final, extra_audit = sanitize_extra_feature_columns(
        df=df,
        template=template,
        mapping=mapping_used,
        extra_cols=list(raw_cols) if isinstance(raw_cols, list) else [],
        max_cols=int(params.get("max_extra_feature_cols", 20)),
    )

    cfg_out: Dict[str, Dict[str, Any]] = {}
    for col in extra_cols_final:
        base_cfg = raw_cfg.get(col) if isinstance(raw_cfg, dict) else {}
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        top_values = (
            series.fillna("NA")
            .astype(str)
            .value_counts(dropna=False)
            .head(min(max(nunique, 1), 8))
            .index.astype(str)
            .tolist()
        )
        encoding = str(base_cfg.get("encoding") or "label")
        if nunique <= 12 and encoding not in {"onehot", "label"}:
            encoding = "onehot"
        if nunique > 12 and encoding == "onehot":
            top_values = top_values[:8]
        cfg_out[col] = {
            "keep": bool(base_cfg.get("keep", True)),
            "kind": str(base_cfg.get("kind") or ""),
            "encoding": encoding,
            "recommended_encoding": str(base_cfg.get("recommended_encoding") or encoding),
            "top_values": list(top_values),
            "nunique": nunique,
        }
    return extra_cols_final, cfg_out, extra_audit


def _build_training_schema(
    df: pd.DataFrame,
    mapping_used: Dict[str, str],
    params: Dict[str, Any],
    extra_feature_config: Dict[str, Dict[str, Any]],
    feature_cols: List[str],
) -> Dict[str, Any]:
    source_columns = list(params.get("input_source_columns") or df.columns)
    required_source_columns = [v for v in mapping_used.values() if v]
    expected_source_columns = list(dict.fromkeys(required_source_columns + list(extra_feature_config.keys())))
    source_dtypes = dict(params.get("input_source_dtypes") or {})
    return {
        "source_columns": source_columns,
        "required_source_columns": required_source_columns,
        "expected_source_columns": expected_source_columns,
        "source_dtypes": source_dtypes,
        "mapping": mapping_used,
        "extra_feature_config": extra_feature_config,
        "feature_cols": feature_cols,
        "template": params.get("template"),
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


def _merge_quality_class_metrics(
    test_metrics_cal: Dict[str, Any], quality_payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Добавляет precision/recall/f1 из quality_payload в отчётные метрики (для отбора и UI)."""
    out = dict(test_metrics_cal or {})
    qp = quality_payload or {}
    for k in ("precision", "recall", "f1"):
        if k in qp:
            out[k] = qp[k]
    return out


def run_single_experiment(
    df: pd.DataFrame,
    template: str,
    mapping_used: Dict[str, str],
    params: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lightweight_candidate = bool(params.get("lightweight_candidate", False))
    extra_cols_final, extra_feature_config, extra_audit = _normalize_extra_feature_config(
        df=df,
        template=template,
        mapping_used=mapping_used,
        params=params,
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
        extra_feature_config=extra_feature_config,
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

    train_cfg = TrainConfig(
        model_kind=str(params.get("model_kind", "lightgbm")),
        n_splits=int(params.get("n_splits", 2 if lightweight_candidate else 4)),
        use_class_weight=bool(params.get("use_class_weight", False)),
        model_params=params.get("model_params", {}),
    )
    train_res = train_time_cv(snaps, feature_cols, train_cfg)
    backtest_res = {"mean_metrics": {}, "std_metrics": {}, "folds_df": pd.DataFrame()}
    if not lightweight_candidate:
        backtest_res = walk_forward_backtest(snaps, feature_cols, train_cfg)
        backtest_res["folds_df"].to_csv(out_dir / "tables" / "walk_forward_folds.csv", index=False)

    cal_cfg = CalibrationConfig(method=str(params.get("calibration", "sigmoid")), cv=int(params.get("calibration_cv", 3)))

    X_train = prepare_feature_matrix(train_res["train_df"], feature_cols)
    X_test = prepare_feature_matrix(train_res["test_df"], feature_cols)
    y_train = train_res["train_df"]["target"].astype(int).values
    y_test = train_res["test_df"]["target"].astype(int).values

    if lightweight_candidate:
        p_raw = train_res["model"].predict_proba(X_test)[:, 1]
        cal_res = {
            "calibrator": train_res["model"],
            "p_raw": p_raw,
            "p_cal": p_raw,
            "raw_metrics": train_res.get("test_metrics", {}),
            "metrics": train_res.get("test_metrics", {}),
            "delta": {},
            "method": "none",
        }
    else:
        cal_res = calibrate(train_res["model"], X_train, y_train, X_test, y_test, cal_cfg)

        save_calibration_plot(
            y_true=y_test,
            p_raw=cal_res["p_raw"],
            p_cal=cal_res["p_cal"],
            path=str(out_dir / "plots" / "calibration_curve.png"),
        )
        save_calibration_summary(cal_res, str(out_dir / "tables" / "calibration_summary.json"))

    quality_payload = build_quality_payload(
        y_true=y_test,
        p=np.asarray(cal_res["p_cal"], dtype=float),
        threshold=float(params.get("decision_threshold", 0.5)),
    )
    test_metrics_cal = _merge_quality_class_metrics(cal_res.get("metrics", {}), quality_payload)

    test_scored = train_res["test_df"].copy()
    test_scored["p_raw"] = np.asarray(cal_res["p_raw"], dtype=float)
    test_scored["p_cal"] = np.asarray(cal_res["p_cal"], dtype=float)
    test_scored["pred"] = (
        test_scored["p_cal"] >= float(params.get("decision_threshold", 0.5))
    ).astype(int)
    test_scored_path = out_dir / "tables" / "test_scored.csv"
    test_scored.to_csv(test_scored_path, index=False)

    psi_path = None
    if not lightweight_candidate:
        try:
            psi_df = compute_feature_psi(train_res["train_df"], train_res["test_df"], feature_cols)
            psi_path = out_dir / "tables" / "feature_psi.csv"
            psi_df.to_csv(psi_path, index=False)
        except Exception:
            psi_path = None

    rep_cfg = ReportConfig(out_dir=str(out_dir), top_k=int(params.get("top_k_priority", 500)))
    business_scenario = scenario_from_params(params)
    business_rep = write_reports(
        df_test=train_res["test_df"],
        p=np.asarray(cal_res["p_cal"], dtype=float),
        scenarios=build_scenarios(params),
        cfg=rep_cfg,
    )

    shap_artifacts = {}
    if bool(params.get("enable_shap", False)):
        try:
            ex_cfg = ExplainConfig(
                max_background=int(params.get("shap_background", 2000)),
                max_explain=int(params.get("shap_explain", 500)),
                random_state=42,
                top_n_local=3,
            )
            X_all = prepare_feature_matrix(snaps, feature_cols)
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
            "extra_feature_config": extra_feature_config,
        },
        training_schema=_build_training_schema(
            df=df,
            mapping_used=mapping_used,
            params={**params, "template": template},
            extra_feature_config=extra_feature_config,
            feature_cols=feature_cols,
        ),
    )

    suitability = assess_suitability(df, template, params=params)

    docx_path = str(out_dir / "report.docx")
    docx_info = {}
    if not lightweight_candidate:
        try:
            docx_info = write_docx_report(
                out_path=docx_path,
                template=template,
                params_used={
                    **params,
                    "extra_feature_cols": extra_cols_final,
                    "extra_feature_config": extra_feature_config,
                },
                suitability=suitability,
                metrics={
                    "target_rate": target_rate,
                    "class_counts": class_counts,
                    "cv_metrics_mean": train_res.get("cv_metrics_mean", {}),
                    "cv_metrics_std": train_res.get("cv_metrics_std", {}),
                    "test_metrics_raw": cal_res.get("raw_metrics", {}),
                    "test_metrics_cal": test_metrics_cal,
                    "walk_forward_mean": backtest_res.get("mean_metrics", {}),
                    "walk_forward_std": backtest_res.get("std_metrics", {}),
                    "business_metrics": {
                        "base_best_k": business_rep.get("base_best_k"),
                        "base_max_profit": business_rep.get("base_max_profit"),
                        "scenario": {
                            "margin": business_scenario.margin,
                            "cost": business_scenario.cost,
                            "success": business_scenario.success,
                        },
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

    html_report_info = {}
    if not lightweight_candidate:
        try:
            html_report_info = write_training_html_report(
                out_path=str(out_dir / "training_report.html"),
                template=template,
                params_used={
                    **params,
                    "extra_feature_cols": extra_cols_final,
                    "extra_feature_config": extra_feature_config,
                },
                suitability=suitability,
                metrics={
                    "target_rate": target_rate,
                    "class_counts": class_counts,
                    "cv_metrics_mean": train_res.get("cv_metrics_mean", {}),
                    "cv_metrics_std": train_res.get("cv_metrics_std", {}),
                    "test_metrics_raw": cal_res.get("raw_metrics", {}),
                    "test_metrics_cal": test_metrics_cal,
                    "walk_forward_mean": backtest_res.get("mean_metrics", {}),
                    "walk_forward_std": backtest_res.get("std_metrics", {}),
                    "business_metrics": {
                        "base_best_k": business_rep.get("base_best_k"),
                        "base_max_profit": business_rep.get("base_max_profit"),
                        "scenario": {
                            "margin": business_scenario.margin,
                            "cost": business_scenario.cost,
                            "success": business_scenario.success,
                        },
                    },
                },
                artifact_paths={
                    "profit_plot": business_rep.get("profit_plot"),
                    "calibration_plot": str(out_dir / "plots" / "calibration_curve.png"),
                    "shap_beeswarm": shap_artifacts.get("shap_beeswarm"),
                    "shap_bar": shap_artifacts.get("shap_bar"),
                },
                mode="single",
            )
        except Exception:
            html_report_info = {}

    if not lightweight_candidate:
        try:
            write_model_card(
                template=template,
                params_used={
                    **params,
                    "extra_feature_cols": extra_cols_final,
                    "extra_feature_config": extra_feature_config,
                },
                suitability=suitability,
                metrics={
                    "target_rate": target_rate,
                    "class_counts": class_counts,
                    "test_metrics_raw": cal_res.get("raw_metrics", {}),
                    "test_metrics_cal": test_metrics_cal,
                    "cv_metrics_mean": train_res.get("cv_metrics_mean", {}),
                    "cv_metrics_std": train_res.get("cv_metrics_std", {}),
                    "walk_forward": backtest_res.get("mean_metrics", {}),
                    "business_metrics": {
                        "base_best_k": business_rep.get("base_best_k"),
                        "base_max_profit": business_rep.get("base_max_profit"),
                        "scenario": {
                            "margin": business_scenario.margin,
                            "cost": business_scenario.cost,
                            "success": business_scenario.success,
                        },
                    },
                },
                extra_feature_audit=extra_audit,
                out_path=str(out_dir / "model_card.md"),
            )
        except Exception:
            pass

    if not lightweight_candidate:
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
            "extra_feature_config": extra_feature_config,
        },
        "target_rate": target_rate,
        "class_counts": class_counts,
        "feature_count": int(len(feature_cols)),
        "cv_metrics_mean": train_res.get("cv_metrics_mean", {}),
        "cv_metrics_std": train_res.get("cv_metrics_std", {}),
        "test_metrics_raw": cal_res.get("raw_metrics", {}),
        "test_metrics_cal": test_metrics_cal,
        "quality_payload": quality_payload,
        "walk_forward": {
            "mean_metrics": backtest_res.get("mean_metrics", {}),
            "std_metrics": backtest_res.get("std_metrics", {}),
            "folds_csv": str(out_dir / "tables" / "walk_forward_folds.csv"),
        },
        "business_metrics": {
            "base_best_k": business_rep.get("base_best_k"),
            "base_max_profit": business_rep.get("base_max_profit"),
            "scenario": {
                "margin": business_scenario.margin,
                "cost": business_scenario.cost,
                "success": business_scenario.success,
            },
        },
        "suitability": suitability,
        "extra_feature_audit": extra_audit,
        "training_schema": _build_training_schema(
            df=df,
            mapping_used=mapping_used,
            params={**params, "template": template},
            extra_feature_config=extra_feature_config,
            feature_cols=feature_cols,
        ),
        "artifacts": {
            "bundle": bundle_info,
            "bundle_dir": bundle_info.get("bundle_dir"),
            "profit_plot": business_rep.get("profit_plot"),
            "profit_summary": business_rep.get("profit_summary"),
            "priority_csv": business_rep.get("priority_csv"),
            "report_docx": docx_info.get("docx_path"),
            "training_report_docx": docx_info.get("docx_path"),
            "training_report_html": html_report_info.get("html_path"),
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
    best_params_local: Optional[Dict[str, Any]] = None
    best_exp_dir: Optional[Path] = None
    best_exp_name: Optional[str] = None
    want_shap = bool(params.get("enable_shap", False))

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
                        # Для скорости в grid-search отключаем SHAP на каждом кандидате.
                        # SHAP посчитаем один раз только для лучшей конфигурации после отбора.
                        p_local["enable_shap"] = False
                        p_local["lightweight_candidate"] = True
                        p_local["n_splits"] = int(params.get("candidate_n_splits", 2))

                        if status_callback:
                            status_callback(
                                stage="grid_search",
                                progress=int(20 + 70 * done / total),
                                extra={"experiment": exp_name},
                            )

                        try:
                            res = run_single_experiment(df, template, mapping_used, p_local, exp_dir)
                            score = choose_metric(res, selection_metric)
                            cm = (res.get("quality_payload") or {}).get("confusion_matrix") or []
                            tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else None
                            fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else None
                            fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else None
                            tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else None
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
                                    "roc_auc": res.get("test_metrics_cal", {}).get("roc_auc"),
                                    "pr_auc": res.get("test_metrics_cal", {}).get("pr_auc"),
                                    "brier": res.get("test_metrics_cal", {}).get("brier"),
                                    "precision": res.get("test_metrics_cal", {}).get("precision"),
                                    "recall": res.get("test_metrics_cal", {}).get("recall"),
                                    "f1": res.get("test_metrics_cal", {}).get("f1"),
                                    "base_best_k": res.get("business_metrics", {}).get("base_best_k"),
                                    "base_max_profit": res.get("business_metrics", {}).get("base_max_profit"),
                                    "cm_tn": tn,
                                    "cm_fp": fp,
                                    "cm_fn": fn,
                                    "cm_tp": tp,
                                    "test_metrics_cal": json.dumps(res.get("test_metrics_cal", {}), ensure_ascii=False),
                                    "business_metrics": json.dumps(res.get("business_metrics", {}), ensure_ascii=False),
                                    "bundle_dir": res.get("artifacts", {}).get("bundle", {}).get("bundle_dir"),
                                    "weights_model": str(
                                        (Path(res.get("artifacts", {}).get("bundle", {}).get("bundle_dir", "")) / "model.joblib")
                                    ),
                                    "weights_calibrator": str(
                                        (Path(res.get("artifacts", {}).get("bundle", {}).get("bundle_dir", "")) / "calibrator.joblib")
                                    ),
                                }
                            )
                            if score > best_score:
                                best_score = score
                                best_result = res
                                best_params_local = dict(p_local)
                                best_exp_dir = exp_dir
                                best_exp_name = exp_name
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

    # Полный прогон делаем только для победителя: отчёт, калибровка, walk-forward и SHAP.
    if best_params_local is not None and best_exp_dir is not None:
        try:
            if status_callback:
                status_callback(
                    stage="best_model_interpretation",
                    progress=93,
                    extra={"experiment": best_exp_name or "best_experiment"},
                )
            p_best = dict(best_params_local)
            p_best["enable_shap"] = want_shap
            p_best.pop("lightweight_candidate", None)
            p_best.pop("n_splits", None)
            best_result = run_single_experiment(df, template, mapping_used, p_best, best_exp_dir)
        except Exception:
            # Если полный прогон упал, оставляем быстрый результат, чтобы не ронять весь job.
            pass

    return {
        "mode": "grid_search",
        "selection_metric": selection_metric,
        "n_experiments": int(len(results_df)),
        "all_results_csv": str(out_dir / "experiment_results.csv"),
        "best_result": best_result,
    }
