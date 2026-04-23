from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient
from ui.components.charts import plot_experiment_scores, render_bar_chart, render_line_chart
from ui.poll_rerun import schedule_autorefresh
from ui.components.nav import page_nav
from ui.components.status_cards import metric_card_row
from ui.state import get_state
from ui.steps import METRIC_LABELS, STEP_3_TRAIN, STEP_5_FORECAST, format_model_kind


def _safe_metric(x) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "—"


def _parse_metrics_blob(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except Exception:
            return {}
    return {}


def page():
    st.title("Качество и сравнение моделей")
    st.caption(
        "Метрики итоговой модели, кривые ошибок и при сравнении нескольких вариантов — таблица всех прогонов. "
        "Итоговая модель совпадает с лучшим вариантом по выбранной метрике."
    )

    api = ApiClient.from_env()
    state = get_state()

    if not state.job_id:
        st.info("Сначала запустите обучение на шаге 3.")
        page_nav(STEP_3_TRAIN, STEP_5_FORECAST)
        return

    try:
        train_status = api.job_status(state.job_id)
    except Exception as e:
        st.error(f"Не удалось получить статус обучения: {e}")
        page_nav(STEP_3_TRAIN, STEP_5_FORECAST)
        return

    if train_status.get("status") == "failed":
        st.error(f"Обучение завершилось с ошибкой: {train_status.get('error')}")
        page_nav(STEP_3_TRAIN, STEP_5_FORECAST)
        return

    if train_status.get("status") != "done":
        st.warning(
            "Здесь появятся метрики после завершения обучения. Страница сама обновляется, пока задание в работе."
        )
        schedule_autorefresh(2500, key="model_quality_wait_poll")
        if st.button("Обновить сейчас", key="mq_refresh"):
            st.rerun()
        page_nav(STEP_3_TRAIN, STEP_5_FORECAST)
        return

    if state.job_result is None:
        try:
            state.job_result = api.job_result(state.job_id)
        except Exception as e:
            st.error(f"Не удалось загрузить результат: {e}")
            page_nav(STEP_3_TRAIN, STEP_5_FORECAST)
            return

    result = state.job_result or {}
    quality = result.get("quality_payload") or {}
    test_metrics: dict = dict(
        result.get("test_metrics_cal") or result.get("test_metrics_raw") or result.get("test_metrics") or {}
    )
    for k in ("precision", "recall", "f1"):
        if k in quality:
            test_metrics[k] = quality[k]
    params_used = result.get("params_used", {})

    artifacts = result.get("artifacts") or {}
    best_bundle = artifacts.get("bundle_dir") or (artifacts.get("bundle") or {}).get("bundle_dir")

    comparison_df = pd.DataFrame()
    if result.get("mode") == "grid_search":
        st.markdown("### Сравнение вариантов")
        try:
            exp = api.job_experiments(state.job_id)
            rows = exp.get("rows") or []
            sm = exp.get("selection_metric") or result.get("selection_metric")
            if sm:
                st.write(f"**Метрика для выбора лучшего варианта:** {METRIC_LABELS.get(str(sm), sm)}")

            if rows:
                edf = pd.DataFrame(rows)
                metrics_df = edf["test_metrics_cal"].apply(_parse_metrics_blob).apply(pd.Series) if "test_metrics_cal" in edf.columns else pd.DataFrame()
                comparison_df = pd.concat([edf.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)
                if "model_kind" in comparison_df.columns:
                    comparison_df["model_kind_label"] = comparison_df["model_kind"].map(format_model_kind)
                if "bundle_dir" in comparison_df.columns:
                    comparison_df["is_winner"] = comparison_df["bundle_dir"].map(
                        lambda x: "Да" if str(x) == str(best_bundle) else ""
                    )
                display_cols = [
                    c
                    for c in [
                        "experiment",
                        "status",
                        "model_kind_label",
                        "horizon_days",
                        "history_days",
                        "step_days",
                        "calibration",
                        "score",
                        "pr_auc",
                        "roc_auc",
                        "f1",
                        "precision",
                        "recall",
                        "is_winner",
                    ]
                    if c in comparison_df.columns
                ]
                sub = comparison_df[display_cols].copy()
                sub = sub.rename(
                    columns={
                        "experiment": "Вариант",
                        "status": "Статус",
                        "model_kind_label": "Алгоритм",
                        "horizon_days": "Горизонт, дн.",
                        "history_days": "История, дн.",
                        "step_days": "Шаг, дн.",
                        "calibration": "Калибровка",
                        "score": "Оценка отбора",
                        "pr_auc": "PR-AUC",
                        "roc_auc": "ROC-AUC",
                        "f1": "F1",
                        "precision": "Precision",
                        "recall": "Recall",
                        "is_winner": "Лучшая модель",
                    }
                )
                if "Статус" in sub.columns:
                    sub["Статус"] = sub["Статус"].map({"ok": "Успех", "failed": "Ошибка"}).fillna(sub["Статус"])

                only_success = st.checkbox("Показывать только успешные варианты", value=True)
                filtered = sub[sub["Статус"] == "Успех"].copy() if only_success and "Статус" in sub.columns else sub
                st.dataframe(filtered, width="stretch", hide_index=True)
                if "Оценка отбора" in filtered.columns:
                    chart_df = filtered.rename(columns={"Вариант": "label", "Оценка отбора": "score"})
                    plot_experiment_scores(chart_df, "label", "score", "Лидерборд по метрике отбора")
            else:
                st.info("Таблица сравнения пуста.")
        except Exception as e:
            st.warning(f"Не удалось загрузить таблицу сравнения: {e}")

    if not test_metrics:
        st.warning("Числовые метрики для итоговой модели пока недоступны.")
        with st.expander("Подробности ответа сервера"):
            st.json(result)
        page_nav(STEP_3_TRAIN, STEP_5_FORECAST)
        return

    st.markdown("### Лучшая итоговая модель")
    metric_card_row(
        [
            ("Точность (Precision)", _safe_metric(test_metrics.get("precision"))),
            ("Полнота (Recall)", _safe_metric(test_metrics.get("recall"))),
            ("F1", _safe_metric(test_metrics.get("f1"))),
            ("ROC-AUC", _safe_metric(test_metrics.get("roc_auc"))),
            ("PR-AUC", _safe_metric(test_metrics.get("pr_auc"))),
        ]
    )

    st.markdown("### Параметры итоговой модели")
    st.write(f"**Название:** {params_used.get('model_name', '—')}")
    st.write(f"**Тип данных:** {result.get('template', '—')}")
    mk = params_used.get("model_kind")
    st.write(f"**Алгоритм:** {format_model_kind(str(mk))}")
    st.write(f"**Горизонт (дней):** {params_used.get('horizon_days', '—')}")
    st.write(f"**Окно истории (дней):** {params_used.get('history_days', '—')}")
    if result.get("mode") == "grid_search":
        st.caption("Это лучший вариант среди всех протестированных моделей по выбранной метрике отбора.")

    roc = quality.get("roc_curve", {})
    pr = quality.get("pr_curve", {})
    cm = quality.get("confusion_matrix")
    class_counts = result.get("class_counts") or {}
    target_rate = result.get("target_rate")
    business_metrics = result.get("business_metrics") or {}
    walk_forward = result.get("walk_forward") or {}

    tab_metrics, tab_curves, tab_data, tab_econ, tab_stability = st.tabs(
        ["Метрики", "Кривые", "Target и данные", "Экономика", "Стабильность"]
    )

    with tab_metrics:
        summary_rows = [
            {"Метрика": "PR-AUC", "Значение": _safe_metric(test_metrics.get("pr_auc"))},
            {"Метрика": "ROC-AUC", "Значение": _safe_metric(test_metrics.get("roc_auc"))},
            {"Метрика": "Precision", "Значение": _safe_metric(test_metrics.get("precision"))},
            {"Метрика": "Recall", "Значение": _safe_metric(test_metrics.get("recall"))},
            {"Метрика": "F1", "Значение": _safe_metric(test_metrics.get("f1"))},
        ]
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    with tab_data:
        c1, c2 = st.columns(2)
        c1.metric("Доля класса 1 (target_rate)", _safe_metric(target_rate))
        c2.metric("Число признаков", str(result.get("feature_count", "—")))
        st.caption(
            "Здесь распределение классов уже интерпретируемо корректно: оно рассчитано не по сырому CSV, "
            "а по snapshot-датасету, на котором реально обучалась модель."
        )

        if target_rate is not None:
            try:
                rate = float(target_rate)
                if rate <= 0.1 or rate >= 0.9:
                    st.warning(
                        "Класс 1 встречается очень редко или очень часто. "
                        "Это важный сигнал о возможном дисбалансе обучающей выборки."
                    )
                else:
                    st.success("Сильного дисбаланса классов на обучающей выборке не видно.")
            except Exception:
                pass

        if isinstance(class_counts, dict) and class_counts:
            cc_df = pd.DataFrame(
                [{"Класс": str(k), "Количество": int(v)} for k, v in class_counts.items()]
            )
            render_bar_chart(
                cc_df,
                x_col="Класс",
                y_col="Количество",
                title="Распределение классов на обучающей выборке",
                x_title="Класс target",
                y_title="Количество snapshot-объектов",
            )

    with tab_curves:
        left, right = st.columns(2)
        with left:
            st.markdown("#### ROC")
            if roc.get("fpr") and roc.get("tpr"):
                roc_df = pd.DataFrame({"FPR": roc["fpr"], "TPR": roc["tpr"]})
                render_line_chart(
                    roc_df,
                    x_col="FPR",
                    y_col="TPR",
                    title="ROC-кривая",
                    x_title="False Positive Rate",
                    y_title="True Positive Rate",
                )
            else:
                st.info("Кривая ROC недоступна.")

        with right:
            st.markdown("#### Precision-Recall")
            if pr.get("recall") and pr.get("precision"):
                pr_df = pd.DataFrame({"Recall": pr["recall"], "Precision": pr["precision"]})
                render_line_chart(
                    pr_df,
                    x_col="Recall",
                    y_col="Precision",
                    title="PR-кривая",
                    x_title="Recall",
                    y_title="Precision",
                )
            else:
                st.info("Кривая PR недоступна.")

        if cm and len(cm) == 2:
            st.markdown("#### Матрица ошибок")
            cm_df = pd.DataFrame(cm, index=["Факт: класс 0", "Факт: класс 1"], columns=["Прогноз: 0", "Прогноз: 1"])
            st.dataframe(cm_df, width="stretch")

    with tab_stability:
        psi_path = artifacts.get("feature_psi_csv")
        if psi_path and Path(str(psi_path)).exists():
            try:
                psi_df = pd.read_csv(str(psi_path))
                if not psi_df.empty:
                    st.markdown("#### Стабильность признаков (PSI, топ)")
                    if {"feature", "psi"}.issubset(set(psi_df.columns)):
                        top_psi = psi_df.sort_values("psi", ascending=False).head(20)
                        render_bar_chart(
                            top_psi,
                            x_col="feature",
                            y_col="psi",
                            title="Топ признаков по PSI",
                            x_title="Признак",
                            y_title="PSI",
                            horizontal=True,
                            height=420,
                        )
                    else:
                        st.dataframe(psi_df.head(20), width="stretch", hide_index=True)
            except Exception as e:
                st.info(f"Не удалось прочитать feature_psi.csv: {e}")

        folds_csv = walk_forward.get("folds_csv")
        if folds_csv and Path(str(folds_csv)).exists():
            try:
                folds_df = pd.read_csv(str(folds_csv))
                if not folds_df.empty:
                    st.markdown("#### Walk-forward по фолдам")
                    st.dataframe(folds_df, width="stretch", hide_index=True)
            except Exception as e:
                st.info(f"Не удалось прочитать файл walk-forward: {e}")

    with tab_econ:
        c1, c2 = st.columns(2)
        c1.metric("Оптимальный top-k", str(business_metrics.get("base_best_k", "—")))
        bm_profit = business_metrics.get("base_max_profit")
        c2.metric("Макс. ожидаемая прибыль", "—" if bm_profit is None else f"{float(bm_profit):,.2f}".replace(",", " "))

        profit_plot = artifacts.get("profit_plot")
        if profit_plot and Path(str(profit_plot)).exists():
            st.markdown("#### Кривая экономического эффекта")
            st.image(str(profit_plot), width="stretch")
        else:
            st.info("График экономического эффекта недоступен.")

        profit_summary = artifacts.get("profit_summary")
        if profit_summary and Path(str(profit_summary)).exists():
            try:
                ps_df = pd.read_csv(str(profit_summary))
                if not ps_df.empty:
                    st.markdown("#### Сводка по сценариям")
                    st.dataframe(ps_df, width="stretch", hide_index=True)
            except Exception as e:
                st.info(f"Не удалось прочитать profit_summary.csv: {e}")

    st.success(
        "Эта модель сохранена как итоговая и доступна на шаге «Прогноз» для оценки новых клиентов."
    )

    page_nav(STEP_3_TRAIN, STEP_5_FORECAST)
