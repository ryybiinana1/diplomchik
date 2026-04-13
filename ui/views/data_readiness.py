from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components.nav import page_nav
from ui.api_client import ApiClient
from ui.components.charts import hist_per_entity, monthly_counts
from ui.components.status_cards import verdict_card
from ui.state import get_state, read_csv_cached


def page():
    st.title("Готовность данных")
    st.caption(
        "Шаг 1: загрузка файла и первичная диагностика. "
        "Никакого ML — только понятно, подходят ли данные."
    )

    api = ApiClient.from_env()
    state = get_state()

    # contracts
    if state.contracts is None:
        try:
            state.contracts = api.contracts()
        except Exception as e:
            st.error(f"Не удалось получить contracts с API: {e}")
            return

    contracts = state.contracts["contracts"]

    # A) upload
    up = st.file_uploader("Загрузите CSV", type=["csv"])
    if up is not None:
        state.file_name = up.name
        state.file_bytes = up.getvalue()
        state.inspect = None  # reset downstream

    if not state.file_bytes:
        st.info("Загрузите CSV, чтобы продолжить.")
        return

    # B) choose template
    label_map = {
        "transactions": contracts["transactions"]["label"],
        "subscriptions": contracts["subscriptions"]["label"],
        "events": contracts["events"]["label"],
    }

    template_options = ["transactions", "subscriptions", "events"]
    current_template = state.template if state.template in template_options else "transactions"

    template = st.radio(
        "Тип данных",
        options=template_options,
        format_func=lambda k: label_map.get(k, k),
        index=template_options.index(current_template),
        horizontal=True,
    )
    state.template = template

    # load df locally for quick stats/charts
    try:
        df = read_csv_cached(state.file_bytes)
    except Exception as e:
        st.error(f"Не удалось прочитать CSV локально: {e}")
        return

    # C) quick stats
    st.subheader("Краткая сводка")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк", f"{len(df):,}".replace(",", " "))
    c2.metric("Колонок", df.shape[1])
    c3.metric("Дубликаты строк", int(df.duplicated().sum()))
    c4.metric("Доля пропусков (ср.)", f"{df.isna().mean().mean():.1%}")

    # D) backend inspect
    if state.inspect is None:
        with st.spinner("Анализируем файл (inspect)..."):
            try:
                state.inspect = api.inspect(state.file_bytes, template=template)
            except Exception as e:
                st.error(f"/inspect не сработал: {e}")
                return

    payload = state.inspect
    quality = payload.get("quality_report", {})
    readiness = quality.get("readiness", {})
    diagnostics = quality.get("diagnostics", {})
    date_ranges = quality.get("date_ranges", {})

    # D1) period + detected columns
    st.markdown("### Период данных и ключевые показатели")

    if date_ranges:
        st.json(date_ranges)
    else:
        st.warning(
            "Не удалось автоматически определить период данных. "
            "Проверь, есть ли в датасете колонка с датой."
        )

    diag_df = pd.DataFrame(
        {
            "date_candidates": pd.Series(diagnostics.get("date_candidates", []), dtype="object"),
            "id_candidates": pd.Series(diagnostics.get("id_candidates", []), dtype="object"),
            "numeric_candidates": pd.Series(diagnostics.get("numeric_candidates", []), dtype="object"),
        }
    )
    if not diag_df.empty:
        st.dataframe(diag_df, use_container_width=True)

    # E) charts
    st.subheader("Графики, которые реально помогают понять пригодность")

    if template == "transactions":
        time_col = "event_time" if "event_time" in df.columns else None
        entity_col = "customer_id" if "customer_id" in df.columns else None

        if time_col:
            monthly_counts(df, time_col=time_col, title="Количество транзакций по месяцам")
            monthly_counts(
                df,
                time_col=time_col,
                value_col="amount" if "amount" in df.columns else None,
                title="Сумма по месяцам (если amount доступен)",
            )
        else:
            st.info("Для графика по времени нужна колонка event_time.")

        if entity_col:
            hist_per_entity(
                df,
                entity_col=entity_col,
                title="Сколько транзакций на клиента (распределение)",
            )
        else:
            st.info("Для распределения по клиентам нужна колонка customer_id.")

    elif template == "subscriptions":
        if "period_end" in df.columns:
            monthly_counts(df, time_col="period_end", title="Количество периодов по месяцам")
        else:
            st.info("Для графика по подпискам нужна колонка period_end.")

        if "account_id" in df.columns:
            hist_per_entity(
                df,
                entity_col="account_id",
                title="Сколько записей на аккаунт (распределение)",
            )
        else:
            st.info("Для распределения по аккаунтам нужна колонка account_id.")

    else:  # events
        if "event_time" in df.columns:
            monthly_counts(df, time_col="event_time", title="Количество событий по месяцам")
        else:
            st.info("Для графика по событиям нужна колонка event_time.")

        if "subject_id" in df.columns:
            hist_per_entity(
                df,
                entity_col="subject_id",
                title="Сколько событий на пользователя (распределение)",
            )
        else:
            st.info("Для распределения по пользователям нужна колонка subject_id.")

    # F) verdict
    st.subheader("Вердикт по данным")

    reasons = []
    verdict = readiness.get("verdict", "ready")

    if readiness.get("reasons"):
        reasons.extend(readiness["reasons"])

    span = quality.get("time_span_days")
    if span is not None and template == "transactions" and span < 120:
        if verdict == "ready":
            verdict = "partial"
        reasons.append(
            f"История короткая: {span} дней. Для устойчивой churn-модели часто нужно больше (например 120–180+)."
        )

    tx_info = quality.get("tx_per_customer", {})
    s2 = tx_info.get("share_customers_2plus_tx")
    if template == "transactions" and s2 is not None and s2 < 0.15:
        if verdict == "ready":
            verdict = "partial"
        reasons.append(f"Мало повторных покупок: доля клиентов с ≥2 покупками = {s2:.1%}.")

    if not reasons:
        reasons.append("Базовые проверки не нашли явных стоп-факторов.")

    verdict_card(verdict, reasons)

    # G) preview
    st.subheader("Превью данных")
    preview = payload.get("preview", [])
    if preview:
        st.dataframe(preview, use_container_width=True)
    else:
        st.info("Превью пока не пришло с backend.")

    st.info(
        "Следующий шаг: сопоставить колонки с обязательными полями "
        "и при желании выбрать дополнительные признаки."
    )

    page_nav(None, "Сопоставление")