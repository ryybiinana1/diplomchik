from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient
from ui.components.charts import plot_category_profile
from ui.components.dataset_helpers import (
    LOW_CARDINALITY_LIMIT,
    audit_extra_column,
    build_feature_config,
    profile_extra_column,
    transactions_financial_sources_ok,
)
from ui.components.nav import page_nav
from ui.state import df_to_csv_bytes, get_state, read_csv_cached
from ui.steps import STEP_2_ANALYTICS


def _file_sig(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _render_working_copy_metrics(df: pd.DataFrame, file_bytes: bytes) -> None:
    """Метрики по текущей рабочей копии; дельты — только если файл изменился (очистка и т.п.)."""
    rows = len(df)
    dups = int(df.duplicated().sum())
    miss = float(df.isna().mean().mean())
    sig = _file_sig(file_bytes)
    prev = st.session_state.get("_metric_snapshot")
    prev_for_delta = prev if (prev and prev.get("sig") != sig) else None

    d_rows = d_dups = d_miss = None
    if prev_for_delta:
        if rows != prev_for_delta["rows"]:
            d_rows = f"{rows - prev_for_delta['rows']:+d}"
        if dups != prev_for_delta["dups"]:
            d_dups = f"{dups - prev_for_delta['dups']:+d}"
        if abs(miss - prev_for_delta["miss"]) >= 1e-8:
            d_miss = f"{miss - prev_for_delta['miss']:+.1%}"

    miss_display = "0.0%" if miss < 1e-12 else f"{miss:.1%}"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Строк", f"{rows:,}".replace(",", " "), delta=d_rows, delta_color="inverse")
    m2.metric("Колонок", df.shape[1])
    m3.metric("Полных дубликатов", dups, delta=d_dups, delta_color="inverse")
    m4.metric("Средняя доля пропусков по колонкам", miss_display, delta=d_miss, delta_color="inverse")

    st.session_state["_metric_snapshot"] = {"rows": rows, "dups": dups, "miss": miss, "sig": sig}


def page():
    st.title("Данные и сопоставление колонок")
    st.caption(
        "Загрузите таблицу, выберите тип событий, посмотрите первые строки и укажите, какая колонка "
        "что означает. После сохранения можно выбрать дополнительные признаки и перейти к анализу."
    )

    api = ApiClient.from_env()
    state = get_state()

    if state.contracts is None:
        try:
            state.contracts = api.contracts()
        except Exception as e:
            st.error(f"Не удалось загрузить описание полей с сервера: {e}")
            return

    contracts = state.contracts["contracts"]

    up_col, type_col = st.columns([1.2, 1])
    with up_col:
        up = st.file_uploader("CSV-файл", type=["csv"], help="Файл с разделителем запятая, кодировка UTF-8.")
    with type_col:
        label_map = {
            "transactions": contracts["transactions"]["label"],
            "subscriptions": contracts["subscriptions"]["label"],
            "events": contracts["events"]["label"],
        }
        template_options = ["transactions", "subscriptions", "events"]
        current_template = state.template if state.template in template_options else "transactions"
        state.template = st.selectbox(
            "Тип данных в файле",
            options=template_options,
            index=template_options.index(current_template),
            format_func=lambda k: label_map.get(k, k),
        )

    if up is not None:
        new_name = up.name
        new_bytes = up.getvalue()
        # Streamlit отдаёт тот же файл на каждом прогоне, пока он выбран в виджете.
        # Сбрасываем сопоставление и рабочую копию только при реальной смене файла.
        if state.file_bytes != new_bytes or state.file_name != new_name:
            state.file_name = new_name
            state.file_bytes = new_bytes
            state.working_file_bytes = new_bytes
            state.inspect = None
            state.mapping = {}
            state.extra_feature_cols = []
            state.extra_feature_config = {}
            state.job_id = None
            state.job_status = None
            state.job_result = None
            state.score_id = None
            state.score_status = None
            state.score_result = None
            state.score_file_name = None
            state.score_file_bytes = None
            state.score_column_mapping = {}
            state.score_schema_check = None
            state.did_dedupe = False
            state.did_dropna = False
            state.clean_notice_dedupe = None
            state.clean_notice_dropna = None
            state.clean_notice_restore = None
            st.session_state.pop("_metric_snapshot", None)
            st.session_state.pop("_inspect_data_sig", None)

    if not state.working_file_bytes:
        st.info("Загрузите CSV, чтобы продолжить.")
        return

    try:
        df = read_csv_cached(state.working_file_bytes)
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {e}")
        return

    _render_working_copy_metrics(df, state.working_file_bytes)

    st.markdown("### Первые строки таблицы")
    st.dataframe(df.head(15), width="stretch", height=360)

    with st.expander("Справка по всем колонкам (тип и примеры)", expanded=False):
        meta_rows = []
        for col in df.columns:
            profile = profile_extra_column(df, col)
            meta_rows.append(
                {
                    "Колонка": col,
                    "Тип": profile["dtype"],
                    "Класс": profile["kind"],
                    "Пример": " | ".join(profile["samples"])[:100],
                    "Уникальных": profile["nunique"],
                    "Пропуски": f"{profile['missing_share']:.1%}",
                }
            )
        st.dataframe(pd.DataFrame(meta_rows), width="stretch", height=280)

    st.markdown("### Очистка (по желанию)")
    st.caption(
        "Действия независимы: можно удалить только дубликаты, только строки с пропусками или оба шага по очереди. "
        "Исходный файл на диске не меняется — только рабочая копия в этом сеансе."
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        if st.button("Удалить полные дубликаты", type="secondary", use_container_width=True):
            before_dup = int(df.duplicated().sum())
            cleaned = df.drop_duplicates().reset_index(drop=True)
            removed_dup = len(df) - len(cleaned)
            state.working_file_bytes = df_to_csv_bytes(cleaned)
            state.inspect = None
            state.did_dedupe = True
            if removed_dup == 0:
                state.clean_notice_dedupe = "Полных дубликатов не было — таблица по строкам не изменилась."
            else:
                state.clean_notice_dedupe = (
                    f"Удалено полных дубликатов: {_fmt_int(removed_dup)} "
                    f"(было помечено как дубликат строк: {_fmt_int(before_dup)})."
                )
            st.rerun()
        if state.clean_notice_dedupe:
            st.success(state.clean_notice_dedupe)
    with d2:
        if st.button("Удалить строки с пропусками", type="secondary", use_container_width=True):
            before_rows = len(df)
            cleaned = df.dropna(how="any").reset_index(drop=True)
            after_rows = len(cleaned)
            removed_na = before_rows - after_rows
            state.working_file_bytes = df_to_csv_bytes(cleaned)
            state.inspect = None
            state.did_dropna = True
            if removed_na == 0:
                state.clean_notice_dropna = "Строк с пропусками не найдено — таблица не изменилась."
            else:
                state.clean_notice_dropna = (
                    f"Удалено строк с хотя бы одним пропуском: {_fmt_int(removed_na)} "
                    f"(было {_fmt_int(before_rows)} → стало {_fmt_int(after_rows)})."
                )
            st.rerun()
        if state.clean_notice_dropna:
            st.success(state.clean_notice_dropna)
    with d3:
        if state.file_bytes and state.working_file_bytes != state.file_bytes:
            if st.button("Вернуть исходный файл", type="secondary", use_container_width=True):
                state.working_file_bytes = state.file_bytes
                state.inspect = None
                state.did_dedupe = False
                state.did_dropna = False
                state.clean_notice_dedupe = None
                state.clean_notice_dropna = None
                state.clean_notice_restore = "Снова используется исходный файл без очистки."
                st.session_state.pop("_metric_snapshot", None)
                st.session_state.pop("_inspect_data_sig", None)
                st.rerun()
        if state.clean_notice_restore:
            st.success(state.clean_notice_restore)
            state.clean_notice_restore = None

    if state.did_dedupe or state.did_dropna:
        note = []
        if state.did_dedupe:
            note.append("дубликаты: шаг выполнялся")
        if state.did_dropna:
            note.append("пропуски: шаг выполнялся")
        st.caption("Рабочая копия · " + " · ".join(note))

    columns = list(df.columns)
    contract = contracts[state.template]

    if state.inspect is None:
        with st.spinner("Проверяем файл на сервере…"):
            try:
                state.inspect = api.inspect(state.working_file_bytes, template=state.template)
                st.session_state["_inspect_data_sig"] = _file_sig(state.working_file_bytes)
            except Exception as e:
                st.error(f"Не удалось проанализировать файл: {e}")
                return

    suggested = (state.inspect or {}).get("suggested_mapping", {})
    required = contract["required"]
    optional = contract["optional"]

    st.markdown("### Обязательные поля")
    st.caption("Для каждой роли выберите колонку из вашей таблицы. Пустое значение — поле не используется.")
    if state.template == "transactions":
        st.info(
            "Если в файле **нет готовой суммы заказа**, в блоке «Необязательные поля» оставьте «Сумма операции» пустой "
            "и укажите **«Цена за единицу»** и **«Количество»** — для обучения сумма будет посчитана как цена × количество."
        )

    mapping: dict = {}
    with st.form("mapping_form_step1"):
        for field, meta in required.items():
            default = suggested.get(field, "")
            options = [""] + columns
            idx = options.index(default) if default in options else 0
            label_text = meta.get("meaning") or field
            st.markdown(f"**{meta.get('title', field)}** — {label_text}")
            choice = st.selectbox(
                "Колонка",
                options=options,
                index=idx,
                key=f"req_{field}",
                label_visibility="collapsed",
            )
            mapping[field] = choice or None

        st.markdown("### Необязательные поля")
        for field, meta in optional.items():
            default = suggested.get(field, "")
            options = [""] + columns
            idx = options.index(default) if default in options else 0
            label_text = meta.get("meaning") or field
            st.markdown(f"**{meta.get('title', field)}** — {label_text}")
            choice = st.selectbox(
                "Колонка",
                options=options,
                index=idx,
                key=f"opt_{field}",
                label_visibility="collapsed",
            )
            mapping[field] = choice or None

        submitted = st.form_submit_button("Сохранить сопоставление", type="primary", width="stretch")

    if submitted:
        state.mapping = mapping
        state.extra_feature_cols = []
        state.extra_feature_config = {}
        st.success("Сопоставление сохранено. Ниже можно выбрать дополнительные признаки.")

    if not state.mapping:
        st.info("Сохраните сопоставление, затем при необходимости выберите дополнительные колонки.")
        st.caption(
            "В боковой панели можно открыть «2. Анализ датасета» в любой момент; там будет напоминание, "
            "пока сопоставление не сохранено здесь. После сохранения обязательных полей внизу появится кнопка перехода."
        )
        return

    missing_required = [f for f in required if not state.mapping.get(f)]
    if missing_required:
        st.error("Заполните все обязательные поля: " + ", ".join(missing_required))
        st.caption(
            "Пока не заполнены обязательные роли, на шаге «Анализ» не построится полный отчёт. "
            "Исправьте сопоставление и снова нажмите «Сохранить»."
        )
        return

    if state.template == "transactions" and not transactions_financial_sources_ok(state.mapping):
        st.error(
            "Для покупок укажите либо колонку **«Сумма операции»**, либо обе — **«Цена за единицу»** и **«Количество»** "
            "(тогда сумма по строке будет вычислена автоматически)."
        )
        return

    used_columns = {v for v in state.mapping.values() if v}
    extra_cols = [c for c in columns if c not in used_columns]

    st.markdown("### Дополнительные признаки")
    st.caption(
        "Сначала выберите колонки, которые хотите добавить в модель. Для категориальных признаков ниже "
        "появится рекомендация по обработке и компактный график распределения."
    )

    audits = []
    for c in extra_cols:
        verdict, reason = audit_extra_column(df, c)
        profile = profile_extra_column(df, c)
        audits.append(
            {
                "Колонка": c,
                "Оценка": verdict,
                "Тип": profile["kind"],
                "Уникальных": profile["nunique"],
                "Пропуски": profile["missing_share"],
                "Комментарий": reason,
            }
        )

    audit_df = pd.DataFrame(audits)
    if not audit_df.empty:
        audit_df["Пропуски"] = audit_df["Пропуски"].map(lambda x: f"{x:.1%}")
        audit_df["Оценка"] = audit_df["Оценка"].map(
            {"allow": "Можно", "review": "Осторожно", "block": "Не рекомендуется"}
        )
        st.dataframe(audit_df, width="stretch", hide_index=True)

    selectable = [x["Колонка"] for x in audits if x["Оценка"] in ("allow", "review")]
    chosen = st.multiselect(
        "Выберите колонки для обучения",
        options=selectable,
        default=[c for c in state.extra_feature_cols if c in selectable],
    )
    state.extra_feature_cols = chosen

    state.extra_feature_config = {
        col: state.extra_feature_config.get(col, build_feature_config(df, col))
        for col in chosen
    }

    if chosen:
        st.markdown("### Настройка выбранных признаков")
        for col in chosen:
            profile = profile_extra_column(df, col)
            default_cfg = state.extra_feature_config.get(col, build_feature_config(df, col))
            recommended = default_cfg.get("recommended_encoding")

            with st.container(border=True):
                st.markdown(f"#### {col}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Тип", str(profile["kind"]))
                m2.metric("Уникальных", str(profile["nunique"]))
                m3.metric("Пропуски", f"{profile['missing_share']:.1%}")
                m4.metric("Рекомендация", "One-hot" if recommended == "onehot" else "Компактно")

                samples = profile["samples"] or ["—"]
                st.caption(f"Примеры значений: {' | '.join(samples)}")

                if profile["kind"] in {"categorical_low_card", "categorical_medium_card", "categorical_high_card", "free_text", "id_like"}:
                    options = ["onehot", "label"] if profile["kind"] != "free_text" else ["label"]
                    selected_encoding = st.radio(
                        f"Как обработать колонку `{col}`",
                        options=options,
                        index=options.index(default_cfg.get("encoding")) if default_cfg.get("encoding") in options else 0,
                        format_func=lambda x: (
                            "One-hot encoding"
                            if x == "onehot"
                            else "Компактное кодирование"
                        ),
                        horizontal=True,
                        key=f"extra_encoding_{col}",
                    )
                    recommendation = default_cfg.get("recommendation", "")
                    if recommendation:
                        st.info(recommendation)
                    if selected_encoding == "onehot" and int(profile["nunique"]) > LOW_CARDINALITY_LIMIT:
                        st.warning(
                            "Для этой колонки one-hot может создать слишком много признаков. "
                            "Лучше оставить компактный режим."
                        )
                    state.extra_feature_config[col] = {
                        **default_cfg,
                        "encoding": selected_encoding,
                        "top_values": profile["top_values"],
                    }
                    plot_category_profile(
                        df,
                        col,
                        top_n=min(max(int(profile["nunique"]), 5), 10),
                        title=f"Распределение значений в `{col}`",
                    )
                else:
                    st.success("Колонка будет агрегирована автоматически, без отдельной кодировки.")
                    state.extra_feature_config[col] = {
                        **default_cfg,
                        "encoding": None,
                        "top_values": profile["top_values"],
                    }

    page_nav(None, STEP_2_ANALYTICS)

