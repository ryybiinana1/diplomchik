from __future__ import annotations

from io import BytesIO
import json

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient
from ui.components.charts import render_bar_chart, render_line_chart
from ui.components.layout import render_page_header, section_card
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


def _safe_money(x) -> str:
    try:
        return f"{float(x):,.2f}".replace(",", " ")
    except Exception:
        return "—"


def _metric_float(x) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _quality_verdict(metrics: dict, business_metrics: dict) -> tuple[str, str]:
    pr_auc = _metric_float(metrics.get("pr_auc"))
    roc_auc = _metric_float(metrics.get("roc_auc"))
    f1 = _metric_float(metrics.get("f1"))
    max_profit = _metric_float(business_metrics.get("base_max_profit"))

    if pr_auc is not None and roc_auc is not None and pr_auc >= 0.65 and roc_auc >= 0.75:
        return (
            "Модель выглядит достаточно сильной для практического пилота.",
            "Качество по ключевым метрикам уверенное, поэтому можно переходить к приоритизации клиентов и проверке эффекта на кампании удержания.",
        )
    if (pr_auc is not None and pr_auc >= 0.5) or (roc_auc is not None and roc_auc >= 0.7) or (f1 is not None and f1 >= 0.45):
        profit_tail = (
            f" Базовый сценарий даёт ожидаемый эффект до {_safe_money(max_profit)}."
            if max_profit is not None
            else ""
        )
        return (
            "Модель можно использовать как рабочий baseline.",
            "Есть сигнал для отбора клиентов, но решение лучше принимать вместе с экономикой, а не только по вероятности оттока." + profit_tail,
        )
    return (
        "Модель стоит использовать осторожно.",
        "Метрики пока скорее подходят для исследовательского режима: сначала лучше проверить данные, сценарий удержания и устойчивость эффекта на пилоте.",
    )


def _parse_metrics_blob(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except Exception:
            return {}
    return {}


def _read_job_csv(api: ApiClient, job_id: str, artifact_key: str) -> pd.DataFrame:
    data = api.download_job_artifact(job_id, artifact_key)
    return pd.read_csv(BytesIO(data))


def _priority_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ("p", "p_cal", "p_raw", "EV_base", "value_proxy"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    display_cols = [
        c
        for c in ("entity_id", "customer_id", "client_id", "account_id", "user_id", "p", "EV_base", "value_proxy")
        if c in out.columns
    ]
    if not display_cols:
        display_cols = list(out.columns[: min(8, len(out.columns))])
    return out[display_cols].rename(
        columns={
            "entity_id": "Объект",
            "customer_id": "Клиент",
            "client_id": "Клиент",
            "account_id": "Аккаунт",
            "user_id": "Пользователь",
            "p": "Вероятность оттока",
            "EV_base": "Ожидаемый эффект",
            "value_proxy": "Ценность",
        }
    )


def page():
    render_page_header(
        "Качество и сравнение моделей",
        "Здесь собраны метрики итоговой модели, кривые ошибок и при сравнении нескольких вариантов таблица всех прогонов.",
        eyebrow="Шаг 4",
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
    business_metrics = result.get("business_metrics") or {}

    artifacts = result.get("artifacts") or {}
    best_bundle = artifacts.get("bundle_dir") or (artifacts.get("bundle") or {}).get("bundle_dir")

    comparison_df = pd.DataFrame()
    comparison_table = pd.DataFrame()
    comparison_selection_label = None
    comparison_error = None
    winner_summary = None
    if result.get("mode") == "grid_search":
        try:
            exp = api.job_experiments(state.job_id)
            rows = exp.get("rows") or []
            sm = exp.get("selection_metric") or result.get("selection_metric")
            if sm:
                comparison_selection_label = METRIC_LABELS.get(str(sm), sm)

            if rows:
                edf = pd.DataFrame(rows)
                metrics_df = (
                    edf["test_metrics_cal"].apply(_parse_metrics_blob).apply(pd.Series)
                    if "test_metrics_cal" in edf.columns
                    else pd.DataFrame()
                )
                if not metrics_df.empty:
                    metrics_df = metrics_df[[c for c in metrics_df.columns if c not in edf.columns]]
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
                comparison_table = sub.copy()
                successful = comparison_df[comparison_df["status"] == "ok"].copy() if "status" in comparison_df.columns else comparison_df
                if not successful.empty:
                    winner = successful[successful["bundle_dir"].astype(str) == str(best_bundle)].head(1)
                    if winner.empty:
                        winner = successful.sort_values("score", ascending=False).head(1)
                    if not winner.empty:
                        winner_summary = winner.iloc[0].to_dict()
        except Exception as e:
            comparison_error = str(e)

    if not test_metrics:
        st.warning("Числовые метрики для итоговой модели пока недоступны.")
        with st.expander("Подробности ответа сервера"):
            st.json(result)
        page_nav(STEP_3_TRAIN, STEP_5_FORECAST)
        return

    verdict_title, verdict_note = _quality_verdict(test_metrics, business_metrics)

    with section_card("Управленческий вывод", "Короткая выжимка для принятия решения перед переходом к прогнозу."):
        top_k = business_metrics.get("base_best_k")
        max_profit = business_metrics.get("base_max_profit")
        metric_card_row(
            [
                ("PR-AUC", _safe_metric(test_metrics.get("pr_auc"))),
                ("ROC-AUC", _safe_metric(test_metrics.get("roc_auc"))),
                ("Рекомендуемый top-k", str(top_k) if top_k is not None else "—"),
                ("Макс. эффект", _safe_money(max_profit)),
            ]
        )
        st.write(f"**Вывод:** {verdict_title}")
        st.caption(verdict_note)

    with section_card("Лучшая итоговая модель", "Ключевые метрики уже откалиброванной итоговой модели."):
        metric_card_row(
            [
                ("Точность (Precision)", _safe_metric(test_metrics.get("precision"))),
                ("Полнота (Recall)", _safe_metric(test_metrics.get("recall"))),
                ("F1", _safe_metric(test_metrics.get("f1"))),
                ("ROC-AUC", _safe_metric(test_metrics.get("roc_auc"))),
                ("PR-AUC", _safe_metric(test_metrics.get("pr_auc"))),
            ]
        )

    with section_card("Параметры итоговой модели", "Эта сводка помогает быстро понять, что именно было выбрано системой."):
        st.write(f"**Название:** {params_used.get('model_name', '—')}")
        st.write(f"**Тип данных:** {result.get('template', '—')}")
        mk = params_used.get("model_kind")
        st.write(f"**Алгоритм:** {format_model_kind(str(mk))}")
        st.write(f"**Горизонт (дней):** {params_used.get('horizon_days', '—')}")
        st.write(f"**Окно истории (дней):** {params_used.get('history_days', '—')}")
        if result.get("mode") == "grid_search":
            st.caption("Это лучший вариант среди всех протестированных моделей по выбранной метрике отбора.")

    if winner_summary:
        with section_card("Итог сравнения", "Короткая выжимка по лучшему варианту среди всех успешно обученных моделей."):
            metric_card_row(
                [
                    ("Успешных вариантов", str(int((comparison_df["status"] == "ok").sum())) if "status" in comparison_df.columns else "—"),
                    ("Метрика отбора", _safe_metric(winner_summary.get("score"))),
                    ("PR-AUC лучшего", _safe_metric(winner_summary.get("pr_auc"))),
                    ("ROC-AUC лучшего", _safe_metric(winner_summary.get("roc_auc"))),
                ]
            )
            st.write(f"**Алгоритм лучшего варианта:** {format_model_kind(str(winner_summary.get('model_kind')))}")

    if result.get("mode") == "grid_search":
        with section_card("Все протестированные варианты", "Полная таблица сравнения нужна скорее для аналитика, чем для руководителя."):
            if comparison_selection_label:
                st.write(f"**Метрика выбора лучшего варианта:** {comparison_selection_label}")
            if comparison_error:
                st.warning(f"Не удалось загрузить таблицу сравнения: {comparison_error}")
            elif comparison_table.empty:
                st.info("Таблица сравнения пуста.")
            else:
                filtered = (
                    comparison_table[comparison_table["Статус"] == "Успех"].copy()
                    if "Статус" in comparison_table.columns
                    else comparison_table
                )
                with st.expander("Открыть подробную таблицу сравнения", expanded=False):
                    st.dataframe(filtered, width="stretch", hide_index=True)

    roc = quality.get("roc_curve", {})
    pr = quality.get("pr_curve", {})
    cm = quality.get("confusion_matrix")
    class_counts = result.get("class_counts") or {}
    target_rate = result.get("target_rate")
    business_scenario = business_metrics.get("scenario") or {}
    walk_forward = result.get("walk_forward") or {}

    tab_metrics, tab_econ, tab_curves, tab_data, tab_stability = st.tabs(
        ["Метрики", "Экономика", "Кривые", "Target и данные", "Стабильность"]
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
        if walk_forward.get("folds_csv"):
            try:
                folds_df = _read_job_csv(api, state.job_id, "walk_forward_folds")
                if not folds_df.empty:
                    st.markdown("#### Walk-forward по фолдам")
                    st.caption(
                        "Walk-forward показывает, насколько качество модели устойчиво на последовательных временных разрезах."
                    )
                    st.dataframe(folds_df, width="stretch", hide_index=True)
            except Exception as e:
                st.info(f"Не удалось прочитать файл walk-forward: {e}")
        else:
            st.info("Дополнительная проверка устойчивости для этого запуска недоступна.")

    with tab_econ:
        c1, c2 = st.columns(2)
        c1.metric("Оптимальный top-k", str(business_metrics.get("base_best_k", "—")))
        bm_profit = business_metrics.get("base_max_profit")
        c2.metric("Макс. ожидаемая прибыль", "—" if bm_profit is None else f"{float(bm_profit):,.2f}".replace(",", " "))
        if business_scenario:
            st.caption(
                "Базовый сценарий удержания: "
                f"margin={float(business_scenario.get('margin', 0)):.2f}, "
                f"cost={float(business_scenario.get('cost', 0)):.2f}, "
                f"success={float(business_scenario.get('success', 0)):.2f}"
            )
            st.caption(
                "Это центральный сценарий для расчёта экономики. На графике и в сводке дополнительно сравниваются "
                "осторожный и оптимистичный сценарии, которые строятся от этих же базовых допущений."
            )

        if artifacts.get("profit_plot"):
            try:
                st.markdown("#### Кривая экономического эффекта")
                st.image(api.download_job_artifact(state.job_id, "profit_plot"), width="stretch")
            except Exception as e:
                st.info(f"График экономического эффекта недоступен: {e}")
        else:
            st.info("График экономического эффекта недоступен.")

        if artifacts.get("profit_summary"):
            try:
                ps_df = _read_job_csv(api, state.job_id, "profit_summary")
                if not ps_df.empty:
                    st.markdown("#### Сводка по сценариям")
                    st.caption(
                        "`best_k` — сколько клиентов выгоднее всего взять в кампанию; "
                        "`max_profit` — максимальный ожидаемый эффект при таком размере кампании."
                    )
                    st.dataframe(ps_df, width="stretch", hide_index=True)
            except Exception as e:
                st.info(f"Не удалось прочитать profit_summary.csv: {e}")

        if artifacts.get("priority_csv"):
            try:
                prio_df = _read_job_csv(api, state.job_id, "priority_csv")
                if not prio_df.empty:
                    st.markdown("#### Топ клиентов, которых стоит удерживать")
                    st.dataframe(_priority_table(prio_df.head(25)), width="stretch", hide_index=True)
            except Exception as e:
                st.info(f"Не удалось прочитать priority_list_topk.csv: {e}")

    page_nav(STEP_3_TRAIN, STEP_5_FORECAST)
