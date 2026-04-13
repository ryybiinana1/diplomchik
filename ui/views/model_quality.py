from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient
from ui.components.nav import page_nav
from ui.state import get_state


def page():
    st.title("Качество модели")
    st.caption("Шаг 4: посмотреть, насколько хорошо обучилась модель, прежде чем переходить к прогнозу.")

    api = ApiClient.from_env()
    state = get_state()

    if not getattr(state, "job_id", None):
        st.info("Сначала запусти обучение на странице «Обучение».")
        page_nav("Обучение", "Прогноз")
        return

    if getattr(state, "job_result", None) is None:
        try:
            state.job_result = api.job_result(state.job_id)
        except Exception as e:
            st.error(f"Не удалось получить результат job: {e}")
            page_nav("Обучение", "Прогноз")
            return

    result = state.job_result or {}

    test_metrics = result.get("test_metrics_cal") or result.get("test_metrics") or {}
    quality = result.get("quality_payload", {})

    if not test_metrics:
        st.warning("Метрики качества пока недоступны.")
        st.json(result)
        page_nav("Обучение", "Прогноз")
        return

    st.subheader("Основные метрики")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")
    c2.metric("PR-AUC", f"{test_metrics.get('pr_auc', 0):.3f}")
    c3.metric("F1", f"{test_metrics.get('f1', 0):.3f}")
    c4.metric("Precision", f"{test_metrics.get('precision', 0):.3f}")
    c5.metric("Recall", f"{test_metrics.get('recall', 0):.3f}")

    c6, c7, c8 = st.columns(3)
    c6.metric("Brier", f"{test_metrics.get('brier', 0):.3f}")
    c7.metric("Test size", result.get("walk_forward", {}).get("folds_csv", "—"))
    c8.metric("Target rate", f"{quality.get('target_rate', 0):.3f}")

    roc = quality.get("roc_curve", {})
    pr = quality.get("pr_curve", {})

    left, right = st.columns(2)

    with left:
        st.markdown("### ROC curve")
        if roc.get("fpr") and roc.get("tpr"):
            roc_df = pd.DataFrame({
                "FPR": roc["fpr"],
                "TPR": roc["tpr"],
            }).set_index("FPR")
            st.line_chart(roc_df)
        else:
            st.info("ROC curve пока недоступна.")

    with right:
        st.markdown("### Precision-Recall curve")
        if pr.get("recall") and pr.get("precision"):
            pr_df = pd.DataFrame({
                "Recall": pr["recall"],
                "Precision": pr["precision"],
            }).set_index("Recall")
            st.line_chart(pr_df)
        else:
            st.info("PR curve пока недоступна.")

    st.markdown("### Матрица ошибок")
    cm = quality.get("confusion_matrix")
    if cm and len(cm) == 2:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )
        st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("Матрица ошибок пока недоступна.")

    st.markdown("### Как это интерпретировать")
    st.info(
        "ROC-AUC показывает, насколько хорошо модель отделяет клиентов с риском оттока от остальных. "
        "PR-AUC особенно полезен, если класс оттока редкий. "
        "Матрица ошибок показывает, сколько объектов модель классифицировала правильно и где ошиблась."
    )

    page_nav("Обучение", "Прогноз")