from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient
from ui.components.charts import (
    hist_per_entity,
    monthly_average_value,
    monthly_counts,
    monthly_unique_entities,
    plot_category_profile,
    plot_daily_counts,
    plot_missing_by_column,
    plot_numeric_bins,
    plot_weekly_counts,
    top_entities_by_value,
)
from ui.components.dataset_helpers import (
    audit_extra_column,
    pick_best_date_col,
    pick_best_id_col,
    profile_extra_column,
)
from ui.components.layout import render_page_header, section_card
from ui.components.nav import page_nav
from ui.state import get_state, read_csv_cached
from ui.steps import STEP_1_DATA, STEP_3_TRAIN


def _file_sig(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _mapped_column(mapping: dict, canonical: str | None) -> str | None:
    if not canonical:
        return None
    v = mapping.get(canonical)
    return v if v else None


def _fmt_int(value: int | float | None) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}".replace(",", " ")
    except Exception:
        return "—"


def _fmt_money(value: float | int | None) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.2f}".replace(",", " ")
    except Exception:
        return "—"


def _build_business_frame(
    df: pd.DataFrame,
    *,
    template: str,
    mapping: dict,
) -> tuple[pd.DataFrame, str | None, str | None]:
    dfx = df.copy()

    if template == "transactions":
        amount_src = _mapped_column(mapping, "amount")
        if amount_src and amount_src in dfx.columns:
            dfx["_business_value"] = pd.to_numeric(dfx[amount_src], errors="coerce")
            return dfx, "_business_value", "Выручка"

        up_c = _mapped_column(mapping, "unit_price")
        q_c = _mapped_column(mapping, "quantity")
        if up_c and q_c and up_c in dfx.columns and q_c in dfx.columns:
            dfx["_business_value"] = pd.to_numeric(dfx[up_c], errors="coerce") * pd.to_numeric(
                dfx[q_c], errors="coerce"
            )
            return dfx, "_business_value", "Выручка"

        return dfx, None, None

    if template == "subscriptions":
        mrr_src = _mapped_column(mapping, "mrr")
        if mrr_src and mrr_src in dfx.columns:
            dfx["_business_value"] = pd.to_numeric(dfx[mrr_src], errors="coerce")
            return dfx, "_business_value", "Платёж / MRR"
        return dfx, None, None

    if template == "events":
        dfx["_business_value"] = 1.0
        return dfx, "_business_value", "Число событий"

    return dfx, None, None


def _metric_rows(metrics: list[tuple[str, str]], per_row: int = 4) -> None:
    for start in range(0, len(metrics), per_row):
        chunk = metrics[start : start + per_row]
        cols = st.columns(len(chunk))
        for col, (label, value) in zip(cols, chunk):
            col.metric(label, value)


def _period_text(df: pd.DataFrame, time_col: str | None) -> str | None:
    if not time_col or time_col not in df.columns:
        return None
    s = pd.to_datetime(df[time_col], errors="coerce").dropna()
    if s.empty:
        return None
    return f"{s.min():%d.%m.%Y} - {s.max():%d.%m.%Y}"


def _entity_label(template: str) -> str:
    return {
        "transactions": "клиентов",
        "subscriptions": "аккаунтов",
        "events": "пользователей",
    }.get(template, "объектов")


def _extra_feature_table(df: pd.DataFrame, state) -> pd.DataFrame:
    rows = []
    for col in state.extra_feature_cols or []:
        verdict, reason = audit_extra_column(df, col)
        cfg = (state.extra_feature_config or {}).get(col, {})
        rows.append(
            {
                "Колонка": col,
                "Тип": profile_extra_column(df, col)["kind"],
                "Оценка": {"allow": "Можно", "review": "Осторожно", "block": "Не рекомендуется"}.get(
                    verdict, verdict
                ),
                "Режим": "One-hot" if cfg.get("encoding") == "onehot" else "Компактно",
                "Комментарий": reason,
            }
        )
    return pd.DataFrame(rows)


def page():
    render_page_header(
        "Анализ датасета",
        "Проверьте, что файл читается корректно и содержит полезную историю по клиентам. "
        "Ниже отдельно показаны бизнес-обзор и техническая диагностика качества данных.",
        eyebrow="Шаг 2",
    )

    api = ApiClient.from_env()
    state = get_state()

    if state.contracts is None:
        try:
            state.contracts = api.contracts()
        except Exception as e:
            st.error(f"Не удалось загрузить описание полей: {e}")
            page_nav(STEP_1_DATA, STEP_3_TRAIN, next_disabled_reason="Описание полей пока недоступно.")
            return

    if not state.working_file_bytes or not state.mapping:
        st.warning("Сначала загрузите файл и сохраните сопоставление колонок на шаге 1.")
        page_nav(STEP_1_DATA, STEP_3_TRAIN)
        return

    contract = state.contracts["contracts"][state.template] if state.contracts else None
    if contract:
        missing = [f for f in contract["required"] if not state.mapping.get(f)]
        if missing:
            st.error("Не заполнены обязательные поля. Вернитесь на шаг 1.")
            page_nav(STEP_1_DATA, STEP_3_TRAIN)
            return

    df = read_csv_cached(state.working_file_bytes)
    data_sig = _file_sig(state.working_file_bytes)
    if st.session_state.get("_inspect_data_sig") != data_sig:
        state.inspect = None

    if state.inspect is None:
        with st.spinner("Загружаем отчёт с сервера…"):
            try:
                state.inspect = api.inspect(state.working_file_bytes, template=state.template)
                st.session_state["_inspect_data_sig"] = data_sig
            except Exception as e:
                st.error(f"Не удалось получить аналитику: {e}")
                page_nav(STEP_1_DATA, STEP_3_TRAIN, next_disabled_reason="Сервер пока не смог построить аналитику.")
                return

    payload = state.inspect or {}
    quality = payload.get("quality_report", {})
    diagnostics = quality.get("diagnostics", {})
    date_ranges = quality.get("date_ranges", {})

    dup_df = int(df.duplicated().sum())
    miss_mean = float(df.isna().mean().mean())

    entity_canon = (contract or {}).get("entity_id_field")
    time_canon = (contract or {}).get("time_field")
    time_src = _mapped_column(state.mapping, time_canon) or pick_best_date_col(df, diagnostics)
    entity_src = _mapped_column(state.mapping, entity_canon) or pick_best_id_col(df, diagnostics)
    event_name_src = _mapped_column(state.mapping, "event_name") if state.template == "events" else None
    dfx, business_value_col, business_value_label = _build_business_frame(
        df,
        template=state.template,
        mapping=state.mapping,
    )
    period_text = _period_text(dfx, time_src)

    with section_card("Краткая сводка", "Быстрое резюме по объёму файла и ключевым показателям."):
        summary_metrics: list[tuple[str, str]] = [
            ("Строк в файле", _fmt_int(len(dfx))),
            (
                f"Уникальных {_entity_label(state.template)}",
                _fmt_int(dfx[entity_src].nunique()) if entity_src and entity_src in dfx.columns else "—",
            ),
            ("Полных дубликатов", _fmt_int(dup_df)),
            ("Средняя доля пропусков", f"{miss_mean:.1%}"),
        ]
        if business_value_col and business_value_col in dfx.columns:
            total_value = pd.to_numeric(dfx[business_value_col], errors="coerce").sum(min_count=1)
            avg_value = pd.to_numeric(dfx[business_value_col], errors="coerce").mean()
            summary_metrics.append((f"Суммарная {business_value_label.lower()}", _fmt_money(total_value)))
            if state.template == "transactions":
                summary_metrics.append(("Средний чек по строке", _fmt_money(avg_value)))
            elif state.template == "subscriptions":
                summary_metrics.append(("Средний платёж по строке", _fmt_money(avg_value)))

        _metric_rows(summary_metrics, per_row=4)

        if period_text:
            st.caption(
                f"Период данных: **{period_text}**. Для анализа используются время **{time_src or '—'}** и объект **{entity_src or '—'}**."
            )
        else:
            st.caption(f"Для анализа используются время **{time_src or '—'}** и объект **{entity_src or '—'}**.")

    with section_card(
        "Бизнес-обзор",
        "Агрегаты по времени и по клиентам помогают быстро понять, насколько история подходит для дипломного кейса.",
    ):
        if time_src and entity_src:
            left, right = st.columns(2)
            with left:
                if business_value_col:
                    title = {
                        "transactions": "Выручка по месяцам",
                        "subscriptions": "Платежи / MRR по месяцам",
                        "events": "Число событий по месяцам",
                    }.get(state.template, "Суммарное значение по месяцам")
                    monthly_counts(dfx, time_col=time_src, value_col=business_value_col, title=title)
                else:
                    monthly_counts(dfx, time_col=time_src, title="Интенсивность записей по месяцам")
            with right:
                monthly_unique_entities(
                    dfx,
                    time_col=time_src,
                    entity_col=entity_src,
                    title=f"Уникальные {_entity_label(state.template)} по месяцам",
                )

            if business_value_col:
                left, right = st.columns(2)
                with left:
                    top_title = {
                        "transactions": "Топ клиентов по суммарной выручке",
                        "subscriptions": "Топ аккаунтов по сумме платежей",
                        "events": "Топ пользователей по числу событий",
                    }.get(state.template, "Топ объектов по суммарному значению")
                    top_entities_by_value(
                        dfx,
                        entity_col=entity_src,
                        value_col=business_value_col,
                        title=top_title,
                        top_n=10,
                    )
                with right:
                    if state.template == "transactions":
                        monthly_average_value(
                            dfx,
                            time_col=time_src,
                            value_col=business_value_col,
                            title="Средний чек по месяцам",
                        )
                    elif state.template == "subscriptions":
                        monthly_average_value(
                            dfx,
                            time_col=time_src,
                            value_col=business_value_col,
                            title="Средний платёж по месяцам",
                        )
                    else:
                        monthly_counts(
                            dfx,
                            time_col=time_src,
                            title="Интенсивность событий по месяцам",
                        )
        else:
            st.info("Чтобы собрать бизнес-обзор, укажите колонку времени и ID объекта на шаге 1.")

    with section_card(
        "Качество данных",
        "Этот блок отвечает на технический вопрос: можно ли доверять файлу для дальнейшего обучения.",
    ):
        q_left, q_right = st.columns(2)
        with q_left:
            plot_missing_by_column(dfx, top_n=15)
        with q_right:
            if entity_src:
                hist_per_entity(
                    dfx,
                    entity_col=entity_src,
                    title=f"Сколько строк истории приходится на один объект ({entity_src})",
                )
                st.caption(
                    "Это технический график: он показывает, у скольких клиентов есть 1, 2, 3 и более строк истории."
                )
            else:
                st.info("Не удалось определить колонку объекта. Укажите ID на шаге 1.")

        if time_src:
            monthly_counts(dfx, time_col=time_src, title=f"Интенсивность загрузки данных по месяцам ({time_src})")
            st.caption(
                "График ниже показывает именно число строк в файле по времени и помогает заметить провалы загрузки."
            )
        else:
            st.info("Не удалось выбрать колонку с датой/временем — укажите её в сопоставлении на шаге 1.")

        with st.expander("Показать технические детали и дополнительные признаки", expanded=False):
            if time_src:
                tech_left, tech_right = st.columns(2)
                with tech_left:
                    plot_weekly_counts(dfx, time_col=time_src, title=f"Число строк по неделям ({time_src})")
                with tech_right:
                    plot_daily_counts(dfx, time_col=time_src, title=f"Число строк по дням ({time_src})")

            if date_ranges:
                st.markdown("#### Диапазоны дат по колонкам")
                for col, info in date_ranges.items():
                    st.markdown(
                        f"- **{col}:** с {info.get('min', '—')} по {info.get('max', '—')} "
                        f"(доля заполненных: {info.get('non_null_share', '—')})"
                    )
            else:
                st.info("Автоматически определить период по датам не удалось — проверьте колонку времени на шаге 1.")

            if business_value_col:
                plot_numeric_bins(
                    dfx,
                    business_value_col,
                    title=f"Распределение показателя по строкам файла ({business_value_label})",
                    bins=28,
                )

            numeric_candidates = [
                c for c in state.extra_feature_cols if profile_extra_column(dfx, c)["kind"] == "numeric"
            ]
            if numeric_candidates:
                st.markdown("#### Дополнительные числовые признаки")
                for col in numeric_candidates[:4]:
                    plot_numeric_bins(dfx, col, title=f"Распределение `{col}`", bins=24)

            if state.template == "subscriptions" and "subscription_status" in dfx.columns:
                plot_category_profile(dfx, "subscription_status", top_n=10, title="Статусы подписок")

            if event_name_src and event_name_src in dfx.columns:
                plot_category_profile(dfx, event_name_src, top_n=12, title="Типы событий")

            country_src = _mapped_column(state.mapping, "country")
            if state.template == "transactions" and country_src and country_src in dfx.columns:
                plot_category_profile(dfx, country_src, top_n=12, title="География по строкам файла")

            category_extras = [
                c
                for c in state.extra_feature_cols
                if profile_extra_column(dfx, c)["kind"]
                in {"categorical_low_card", "categorical_medium_card", "categorical_high_card", "free_text", "id_like"}
            ]
            if category_extras:
                st.markdown("#### Дополнительные категориальные признаки")
                for col in category_extras[:4]:
                    plot_category_profile(dfx, col, top_n=10, title=f"Дополнительный признак: `{col}`")

            extra_df = _extra_feature_table(dfx, state)
            if not extra_df.empty:
                st.markdown("#### Выбранные дополнительные признаки")
                st.dataframe(extra_df, width="stretch", hide_index=True)

            st.info(
                "Проверку дисбаланса классов по исходному CSV здесь не показываем специально: "
                "в этой задаче реальный target формируется позже на snapshot-датасете."
            )

    with section_card("Дальше", "Когда файл выглядит корректно и графики читаются понятно, можно переходить к обучению."):
        st.write("Если всё выглядит ожидаемо, переходите к следующему шагу.")

    page_nav(STEP_1_DATA, STEP_3_TRAIN)
