from __future__ import annotations

import re

import streamlit as st

from ui.api_client import ApiClient
from ui.components.nav import page_nav
from ui.poll_rerun import schedule_autorefresh
from ui.state import get_state, read_csv_cached
from ui.steps import STEP_4_QUALITY, format_model_kind


def _norm_name(value: str) -> str:
    return re.sub(r"[^\w]+", "_", str(value).strip().lower()).strip("_")


def _default_score_mapping(expected_cols: list[str], actual_cols: list[str]) -> dict[str, str]:
    by_norm = {_norm_name(col): col for col in actual_cols}
    out: dict[str, str] = {}
    for col in expected_cols:
        if col in actual_cols:
            out[col] = col
            continue
        match = by_norm.get(_norm_name(col))
        if match:
            out[col] = match
    return out


def page():
    st.title("Прогноз для новых данных")
    st.caption(
        "Выберите сохранённую модель, вручную сопоставьте все колонки из обучающего набора "
        "и получите оценку риска по каждой строке."
    )

    api = ApiClient.from_env()
    state = get_state()

    try:
        models = api.list_models().get("models", [])
    except Exception as e:
        st.error(f"Не удалось получить список моделей: {e}")
        return

    if not models:
        st.info("Пока нет сохранённых моделей. Сначала завершите обучение на шаге 3.")
        page_nav(STEP_4_QUALITY, None)
        return

    def fmt_model(m: dict) -> str:
        name = m.get("model_name") or "Без названия"
        template = m.get("template", "—")
        horizon = m.get("horizon_days", "—")
        mk = m.get("model_kind", "—")
        mk_ru = format_model_kind(str(mk))
        return f"{name} · {template} · {horizon} дн. · {mk_ru}"

    idx = st.selectbox(
        "Модель для прогноза",
        options=list(range(len(models))),
        format_func=lambda i: fmt_model(models[i]),
    )

    model_key = "scoring_model_idx"
    if model_key not in st.session_state:
        st.session_state[model_key] = idx
    elif st.session_state[model_key] != idx:
        st.session_state[model_key] = idx
        state.score_column_mapping = {}
        state.score_schema_check = None
        state.score_id = None
        state.score_status = None
        state.score_result = None

    chosen = models[idx]

    st.markdown("### Выбранная модель")
    c1, c2, c3 = st.columns(3)
    c1.metric("Название", str(chosen.get("model_name", "—")))
    c2.metric("Горизонт (дней)", str(chosen.get("horizon_days", "—")))
    mk = chosen.get("model_kind", "—")
    c3.metric("Алгоритм", format_model_kind(str(mk)))

    st.write(f"**Тип данных:** {chosen.get('template', '—')}")

    with st.expander("Технический путь к файлам модели (для поддержки)"):
        st.text(chosen.get("bundle_dir", ""))

    up = st.file_uploader("CSV с новыми данными", type=["csv"], key="score_uploader")
    if up is not None:
        new_bytes = up.getvalue()
        if state.score_file_bytes != new_bytes:
            state.score_column_mapping = {}
            state.score_schema_check = None
            state.score_id = None
            state.score_status = None
            state.score_result = None
        state.score_file_name = up.name
        state.score_file_bytes = new_bytes

    if not state.score_file_bytes:
        st.info("Загрузите файл, чтобы запустить прогноз.")
        page_nav(STEP_4_QUALITY, None)
        return

    df_score = read_csv_cached(state.score_file_bytes)
    actual_cols = list(df_score.columns)
    training_schema = chosen.get("training_schema") or {}
    expected_cols = list(training_schema.get("expected_source_columns") or training_schema.get("required_source_columns") or [])
    mapping_from_training = (training_schema.get("mapping") or {})
    reverse_mapping = {v: k for k, v in mapping_from_training.items() if v}

    st.markdown("### Сопоставление колонок")
    st.caption(
        "Нужно вручную подтвердить соответствие для каждой колонки, использованной при обучении. "
        "Лишние колонки в новом файле запрещены: прогноз запускается только при полном совпадении схемы."
    )

    auto_left, auto_right = st.columns([1, 2])
    with auto_left:
        if st.button("Попробовать автосопоставление", use_container_width=True):
            state.score_column_mapping = _default_score_mapping(expected_cols, actual_cols)
            state.score_schema_check = None
            state.score_id = None
            state.score_status = None
            state.score_result = None
            st.rerun()
    with auto_right:
        st.caption(
            "Эта кнопка только подсказывает совпадения по похожим названиям. "
            "Финальная проверка всё равно требует полного явного сопоставления."
        )

    if expected_cols:
        prev_mapping = dict(state.score_column_mapping)
        for source_col in expected_cols:
            role = reverse_mapping.get(source_col)
            label = role if role else f"Доп. признак: {source_col}"
            selected_value = state.score_column_mapping.get(source_col)
            options = [""] + actual_cols
            default_index = options.index(selected_value) if selected_value in options else 0
            chosen_value = st.selectbox(
                f"{label}",
                options=options,
                index=default_index,
                key=f"score_map_{source_col}",
                help=f"При обучении использовалась колонка `{source_col}`. Выберите соответствующую колонку из нового CSV вручную.",
            )
            if chosen_value:
                state.score_column_mapping[source_col] = chosen_value
            else:
                state.score_column_mapping.pop(source_col, None)
        if prev_mapping != state.score_column_mapping:
            state.score_schema_check = None
            state.score_id = None
            state.score_status = None
            state.score_result = None

    mapped_count = len([col for col in expected_cols if state.score_column_mapping.get(col)])
    used_actual_cols = {v for v in state.score_column_mapping.values() if v}
    unused_actual_cols = [col for col in actual_cols if col not in used_actual_cols]
    s1, s2, s3 = st.columns(3)
    s1.metric("Ожидается колонок", str(len(expected_cols)))
    s2.metric("Сопоставлено вручную", str(mapped_count))
    s3.metric("Неиспользованных колонок", str(len(unused_actual_cols)))

    if unused_actual_cols:
        st.warning(
            "В новом файле есть колонки, которые не входят в обучающий набор: "
            + ", ".join(unused_actual_cols[:12])
        )
        if len(unused_actual_cols) > 12:
            st.caption("Показаны первые 12 лишних колонок.")
    else:
        st.success("Лишних колонок не найдено.")

    if state.score_schema_check is None:
        try:
            state.score_schema_check = api.score_schema_check(
                state.score_file_bytes,
                bundle_dir=chosen["bundle_dir"],
                score_mapping=state.score_column_mapping,
            )
        except Exception as e:
            st.error(f"Не удалось проверить схему файла: {e}")
            page_nav(STEP_4_QUALITY, None)
            return

    schema = state.score_schema_check or {}
    st.markdown("### Проверка структуры файла")
    status = schema.get("status")
    if status == "ok":
        st.success("Структура файла полностью совпадает с тем, что использовалось при обучении модели.")
    elif status == "warning":
        st.warning("Файл частично подходит, но структура ещё не совпадает полностью.")
    else:
        st.error("Файл не подходит для этой модели: нужно полное сопоставление без лишних колонок.")

    summary_rows = []
    if schema.get("missing_mapping"):
        summary_rows.append(
            {
                "Проверка": "Не сопоставлены ожидаемые колонки",
                "Детали": ", ".join(schema["missing_mapping"]),
            }
        )
    if schema.get("missing_required"):
        summary_rows.append({"Проверка": "Не хватает обязательных колонок", "Детали": ", ".join(schema["missing_required"])})
    if schema.get("missing_expected"):
        summary_rows.append({"Проверка": "Не хватает колонок из обучающего набора", "Детали": ", ".join(schema["missing_expected"])})
    if schema.get("extra_columns"):
        summary_rows.append({"Проверка": "Есть лишние колонки", "Детали": ", ".join(schema["extra_columns"][:10])})
    if schema.get("invalid_mapping_keys"):
        summary_rows.append(
            {
                "Проверка": "Есть недопустимые ключи сопоставления",
                "Детали": ", ".join(schema["invalid_mapping_keys"]),
            }
        )
    if schema.get("dtype_mismatch"):
        details = ", ".join(
            f"{row['column']} ({row['actual_dtype']} вместо {row['expected_dtype']})"
            for row in schema["dtype_mismatch"][:6]
        )
        summary_rows.append({"Проверка": "Есть отличия по типам", "Детали": details})
    if not summary_rows:
        summary_rows.append({"Проверка": "Проверка схемы", "Детали": "Критичных отличий не найдено"})
    st.dataframe(summary_rows, width="stretch", hide_index=True)

    if st.button("Запустить прогноз", type="primary", use_container_width=True, disabled=not schema.get("ok")):
        try:
            resp = api.start_score(
                state.score_file_bytes,
                bundle_dir=chosen["bundle_dir"],
                score_mapping=state.score_column_mapping,
            )
            state.score_id = resp["score_id"]
            state.score_status = None
            state.score_result = None
            st.success("Прогноз поставлен в очередь.")
        except Exception as e:
            st.error(f"Не удалось запустить: {e}")
            return

    if not state.score_id:
        page_nav(STEP_4_QUALITY, None)
        return

    st.markdown("### Статус")
    if not state.score_status or state.score_status.get("status") not in ("done", "failed"):
        schedule_autorefresh(2500, key="scoring_job_poll")

    if not state.score_status or state.score_status.get("status") not in ("done", "failed"):
        try:
            state.score_status = api.score_job_status(state.score_id)
        except Exception as e:
            st.error(f"Не удалось получить статус: {e}")
            page_nav(STEP_4_QUALITY, None)
            return

    sc = state.score_status or {}
    prog = int(sc.get("progress") or 0)
    stage = sc.get("stage", sc.get("status", "ожидание"))
    st.progress(min(max(prog, 0), 100), text=f"{prog}% — {stage}")

    if sc.get("status") == "done" and sc.get("result"):
        state.score_result = sc["result"]
        st.success("Прогноз готов.")

        dl_url = f"{api.base_url}{state.score_result.get('download_url', '')}"
        if hasattr(st, "link_button"):
            st.link_button("Скачать таблицу результатов (в новой вкладке)", url=dl_url)
        else:
            st.markdown(f"[Скачать таблицу результатов]({dl_url})")

        try:
            csv_bytes = api.download_score_csv(state.score_id)
            st.download_button(
                label="Скачать CSV с результатами",
                data=csv_bytes,
                file_name="scored_clients.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Не удалось подготовить файл для кнопки: {e}")

        n = state.score_result.get("n_scored")
        st.metric("Строк в результате", str(n) if n is not None else "—")

        preview = state.score_result.get("preview", [])
        if preview:
            st.markdown("#### Первые строки результата")
            st.dataframe(preview, width="stretch", height=400)

        st.info(
            "**Как читать результат:** столбец с откалиброванной вероятностью — чем выше значение, тем выше риск; "
            "сегмент риска обобщает уровень; колонки с пояснениями помогают понять вклад типичных факторов."
        )

    elif sc.get("status") == "failed":
        st.error(f"Прогноз не выполнен: {sc.get('error')}")
    else:
        st.info("Ожидание…")

    page_nav(STEP_4_QUALITY, None)
