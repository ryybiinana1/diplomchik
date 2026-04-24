from __future__ import annotations

import hashlib
import io
import re

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient
from ui.components.charts import render_bar_chart
from ui.components.dataset_helpers import profile_extra_column
from ui.components.layout import render_field_intro, render_page_header, section_card
from ui.components.nav import page_nav
from ui.poll_rerun import schedule_autorefresh
from ui.state import get_state, read_csv_cached
from ui.steps import STEP_4_QUALITY, format_model_kind


def _file_sig(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _fmt_int(value: int | float | None) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}".replace(",", " ")
    except Exception:
        return "—"


def _norm_name(value: str) -> str:
    return re.sub(r"[^\w]+", "_", str(value).strip().lower()).strip("_")


def _reset_score_outputs(state, *, clear_mapping: bool = False) -> None:
    if clear_mapping:
        state.score_column_mapping = {}
    state.score_schema_check = None
    state.score_id = None
    state.score_status = None
    state.score_result = None


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


def _inspect_score_file(api: ApiClient, file_bytes: bytes, template: str) -> tuple[dict | None, str | None]:
    cache_key = "_score_inspect_cache"
    sig = _file_sig(file_bytes)
    cached = st.session_state.get(cache_key)
    if cached and cached.get("sig") == sig and cached.get("template") == template:
        return cached.get("payload"), cached.get("error")

    payload = None
    error = None
    with st.spinner("Анализируем файл для автосопоставления…"):
        try:
            payload = api.inspect(file_bytes, template=template)
        except Exception as e:
            error = str(e)

    st.session_state[cache_key] = {"sig": sig, "template": template, "payload": payload, "error": error}
    return payload, error


def _smart_score_mapping(
    expected_cols: list[str],
    actual_cols: list[str],
    mapping_from_training: dict[str, str],
    inspect_payload: dict | None,
) -> dict[str, str]:
    smart = _default_score_mapping(expected_cols, actual_cols)
    suggested = (inspect_payload or {}).get("suggested_mapping", {}) or {}
    source_by_canonical = {canon: src for canon, src in mapping_from_training.items() if src}
    for canonical_role, actual_col in suggested.items():
        source_col = source_by_canonical.get(canonical_role)
        if source_col and source_col in expected_cols and actual_col in actual_cols:
            smart[source_col] = actual_col
    return smart


def _safe_money(value) -> str:
    try:
        return f"{float(value):,.2f}".replace(",", " ")
    except Exception:
        return "—"


def _top_clients_table(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("EV", "EV_base", "EV_conservative", "EV_optimistic", "p_calibrated", "p_raw", "value_proxy"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    display_cols = [
        c
        for c in (
            "entity_id",
            "customer_id",
            "client_id",
            "account_id",
            "user_id",
            "risk_segment",
            "priority",
            "recommended_action",
            "p_calibrated",
            "value_proxy",
            "EV_base",
            "EV_conservative",
            "EV_optimistic",
            "reason_1",
        )
        if c in df.columns
    ]
    if not display_cols:
        display_cols = list(df.columns[: min(8, len(df.columns))])
    return df[display_cols].rename(
        columns={
            "entity_id": "Объект",
            "customer_id": "Клиент",
            "client_id": "Клиент",
            "account_id": "Аккаунт",
            "user_id": "Пользователь",
            "risk_segment": "Сегмент риска",
            "priority": "Приоритет",
            "recommended_action": "Рекомендация",
            "p_calibrated": "Вероятность оттока",
            "EV": "Ожидаемый эффект",
            "EV_base": "EV base",
            "EV_conservative": "EV conservative",
            "EV_optimistic": "EV optimistic",
            "value_proxy": "Ценность",
            "reason_1": "Главная причина",
        }
    )


def _campaign_table_from_priority_df(df: pd.DataFrame) -> pd.DataFrame:
    """Те же колонки и подписи, что в блоке «Кого брать в кампанию удержания»."""
    if df.empty:
        return pd.DataFrame()
    rows = df.to_dict(orient="records")
    return _top_clients_table(rows)


def _priority_segment_csv_bytes(df_priority: pd.DataFrame, priority_label: str) -> bytes:
    """CSV с рангами внутри сегмента приоритета; колонки как в таблице кампании."""
    if "priority" not in df_priority.columns:
        display = pd.DataFrame()
    else:
        sub = df_priority[df_priority["priority"].astype(str) == priority_label].copy()
        display = _campaign_table_from_priority_df(sub)
    if display.empty:
        empty_cols = ["Ранг"] + list(_campaign_table_from_priority_df(pd.DataFrame()).columns)
        display = pd.DataFrame(columns=empty_cols)
    else:
        display.insert(0, "Ранг", range(1, len(display) + 1))
    buf = io.StringIO()
    display.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue().encode("utf-8-sig")


def _pick_client_label_col(df: pd.DataFrame) -> str:
    for candidate in ("entity_id", "customer_id", "client_id", "account_id", "user_id"):
        if candidate in df.columns:
            return candidate
    return ""


def _contract_role_meta(contract: dict | None) -> dict[str, dict]:
    if not contract:
        return {}
    return {
        **(contract.get("required") or {}),
        **(contract.get("optional") or {}),
    }


def _mapping_label(source_col: str, reverse_mapping: dict[str, str], role_meta: dict[str, dict]) -> tuple[str, str]:
    role = reverse_mapping.get(source_col)
    meta = role_meta.get(role or "", {})
    if role:
        title = meta.get("title") or role
        meaning = meta.get("meaning") or "Выберите колонку из нового CSV, которая соответствует этой роли."
        note = (
            f"При обучении эта роль использовала колонку `{source_col}`. {meaning}"
        )
        return title, note
    return (
        f"Доп. признак: {source_col}",
        f"Колонка `{source_col}` была частью обучающего набора как дополнительный признак. "
        "Подтвердите соответствие вручную.",
    )


def _default_scenario_from_model(chosen: dict) -> tuple[float, float, float]:
    params = ((chosen.get("config") or {}).get("params") or {}) if isinstance(chosen.get("config"), dict) else {}
    return (
        float(chosen.get("business_margin", params.get("business_margin", 0.50))),
        float(chosen.get("business_cost", params.get("business_cost", 2.0))),
        float(chosen.get("business_success", params.get("business_success", 0.20))),
    )


def _build_scenarios_preview(margin: float, cost: float, success: float) -> list[dict[str, float | str]]:
    base_margin = min(max(float(margin), 0.0), 1.0)
    base_cost = max(float(cost), 0.0)
    base_success = min(max(float(success), 0.0), 1.0)
    return [
        {
            "name": "conservative",
            "margin": min(max(base_margin * 0.8, 0.0), 1.0),
            "cost": max(base_cost * 1.25, 0.0),
            "success": min(max(base_success * 0.7, 0.0), 1.0),
        },
        {
            "name": "base",
            "margin": base_margin,
            "cost": base_cost,
            "success": base_success,
        },
        {
            "name": "optimistic",
            "margin": min(max(base_margin * 1.15, 0.0), 1.0),
            "cost": max(base_cost * 0.8, 0.0),
            "success": min(max(base_success * 1.35, 0.0), 1.0),
        },
    ]


def _scenario_preview_table(margin: float, cost: float, success: float) -> pd.DataFrame:
    rows = []
    scenario_labels = {
        "conservative": ("Осторожный", "Пессимистичная проверка чувствительности"),
        "base": ("Базовый", "Основной сценарий для решения"),
        "optimistic": ("Оптимистичный", "Потолок эффекта при более мягких допущениях"),
    }
    for scenario in _build_scenarios_preview(margin=float(margin), cost=float(cost), success=float(success)):
        title, comment = scenario_labels.get(str(scenario["name"]), (str(scenario["name"]), ""))
        rows.append(
            {
                "Сценарий": title,
                "Margin": float(scenario["margin"]),
                "Cost": float(scenario["cost"]),
                "Success": float(scenario["success"]),
                "Комментарий": comment,
            }
        )
    return pd.DataFrame(rows)


def _scenario_summary_table(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    rename_map = {
        "scenario": "Сценарий",
        "margin": "Margin",
        "cost": "Cost",
        "success": "Success",
        "clients_with_positive_ev": "Клиентов с EV > 0",
        "best_k": "Рекомендуемый top-k",
        "max_profit": "Макс. эффект",
        "total_positive_ev": "Суммарный положительный EV",
        "mean_ev_top20": "Средний EV у top-20",
    }
    cols = [c for c in rename_map if c in df.columns]
    out = df[cols].rename(columns=rename_map)
    if "Сценарий" in out.columns:
        out["Сценарий"] = out["Сценарий"].map(
            {
                "conservative": "Осторожный",
                "base": "Базовый",
                "optimistic": "Оптимистичный",
            }
        ).fillna(out["Сценарий"])
    return out


def _scenario_label(name: str) -> str:
    return {
        "conservative": "Осторожный",
        "base": "Базовый",
        "optimistic": "Оптимистичный",
    }.get(str(name), str(name))


def _scenario_ev_col(name: str) -> str:
    return {
        "conservative": "EV_conservative",
        "base": "EV_base",
        "optimistic": "EV_optimistic",
    }.get(str(name), "EV_base")


def _summary_message(business_summary: dict) -> str:
    best_k = business_summary.get("best_k")
    positive = business_summary.get("clients_with_positive_ev")
    total = _safe_money(business_summary.get("total_positive_ev"))
    if best_k:
        return (
            f"Система рекомендует начать с top-{best_k}: именно эта ширина кампании даёт лучший ожидаемый эффект "
            f"в базовом сценарии. Потенциально положительный EV есть у {positive} клиентов, суммарно на {total}."
        )
    return (
        f"Положительный EV найден у {positive} клиентов. Сначала имеет смысл посмотреть таблицу сценариев "
        f"и отобрать клиентов с наибольшим EV_base; суммарный положительный эффект оценивается в {total}."
    )


def page():
    render_page_header(
        "Прогноз для новых данных",
        "Выберите сохранённую модель, сопоставьте колонки из обучающего набора и получите оценку риска по каждой строке.",
        eyebrow="Шаг 5",
    )

    api = ApiClient.from_env()
    state = get_state()

    if state.contracts is None:
        try:
            state.contracts = api.contracts()
        except Exception as e:
            st.warning(f"Не удалось загрузить описание полей с сервера: {e}")

    models_cache_key = "_scoring_models_cache"
    if st.button("Обновить список моделей", key="refresh_models_scoring"):
        st.session_state.pop(models_cache_key, None)

    if models_cache_key not in st.session_state:
        try:
            st.session_state[models_cache_key] = api.list_models().get("models", [])
        except Exception as e:
            st.error(f"Не удалось получить список моделей: {e}")
            return

    models = st.session_state.get(models_cache_key) or []

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
        _reset_score_outputs(state, clear_mapping=True)

    chosen = models[idx]
    widget_prefix = _norm_name(chosen.get("bundle_dir", f"model_{idx}")) or f"model_{idx}"
    default_margin, default_cost, default_success = _default_scenario_from_model(chosen)
    margin_key = f"score_margin_{widget_prefix}"
    cost_key = f"score_cost_{widget_prefix}"
    success_key = f"score_success_{widget_prefix}"
    st.session_state.setdefault(margin_key, float(default_margin))
    st.session_state.setdefault(cost_key, float(default_cost))
    st.session_state.setdefault(success_key, float(default_success))
    business_margin = float(st.session_state[margin_key])
    business_cost = float(st.session_state[cost_key])
    business_success = float(st.session_state[success_key])

    chosen_template = str(chosen.get("template", ""))
    contract = ((state.contracts or {}).get("contracts") or {}).get(chosen_template, {})
    role_meta = _contract_role_meta(contract)

    with section_card("Выбранная модель", "Проверьте, что для прогноза выбрана нужная сохранённая модель."):
        c1, c2, c3 = st.columns(3)
        c1.metric("Название", str(chosen.get("model_name", "—")))
        c2.metric("Горизонт (дней)", str(chosen.get("horizon_days", "—")))
        mk = chosen.get("model_kind", "—")
        c3.metric("Алгоритм", format_model_kind(str(mk)))

        st.write(f"**Тип данных:** {chosen_template or '—'}")

        with st.expander("Технический путь к файлам модели (для поддержки)"):
            st.text(chosen.get("bundle_dir", ""))

    up = st.file_uploader("CSV с новыми данными", type=["csv"], key="score_uploader")
    if up is not None:
        new_bytes = up.getvalue()
        if state.score_file_bytes != new_bytes:
            _reset_score_outputs(state, clear_mapping=True)
        state.score_file_name = up.name
        state.score_file_bytes = new_bytes

    if not state.score_file_bytes:
        st.info("Загрузите файл, чтобы запустить прогноз.")
        page_nav(STEP_4_QUALITY, None)
        return

    try:
        df_score = read_csv_cached(state.score_file_bytes)
    except Exception as e:
        st.error(f"Не удалось прочитать файл для прогноза: {e}")
        page_nav(STEP_4_QUALITY, None)
        return

    inspect_payload, inspect_error = _inspect_score_file(api, state.score_file_bytes, chosen_template)
    actual_cols = list(df_score.columns)
    score_file_sig = _file_sig(state.score_file_bytes)[:8]
    mapping_widget_prefix = f"{widget_prefix}_{score_file_sig}"

    with section_card(
        "Загруженный файл для прогноза",
        "Перед сопоставлением проверьте структуру нового CSV так же, как на первом шаге.",
    ):
        rows = len(df_score)
        dups = int(df_score.duplicated().sum())
        miss = float(df_score.isna().mean().mean())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Строк", _fmt_int(rows))
        c2.metric("Колонок", str(df_score.shape[1]))
        c3.metric("Дубликаты", _fmt_int(dups))
        c4.metric("Средняя доля пропусков", "0.0%" if miss < 1e-12 else f"{miss:.1%}")

        st.dataframe(df_score.head(12), width="stretch", height=260)
        with st.expander("Все колонки: типы и примеры", expanded=False):
            meta_rows = []
            for col in df_score.columns:
                profile = profile_extra_column(df_score, col)
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
            st.dataframe(pd.DataFrame(meta_rows), width="stretch", height=240, hide_index=True)

    with section_card(
        "Сценарии удержания",
        "По умолчанию система покажет три сценария эффекта. Ручная настройка спрятана в расширенный блок.",
    ):
        st.dataframe(
            _scenario_preview_table(
                margin=business_margin,
                cost=business_cost,
                success=business_success,
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Базовый сценарий задаёт твои бизнес-допущения, а осторожный и оптимистичный показывают "
            "чувствительность решения без переобучения модели."
        )
        with st.expander("Расширенные настройки сценариев", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                business_margin = st.number_input(
                    "Margin",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state[margin_key]),
                    step=0.05,
                    key=margin_key,
                )
            with c2:
                business_cost = st.number_input(
                    "Cost",
                    min_value=0.0,
                    value=float(st.session_state[cost_key]),
                    step=0.5,
                    key=cost_key,
                )
            with c3:
                business_success = st.number_input(
                    "Success",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state[success_key]),
                    step=0.05,
                    key=success_key,
                )
            st.caption(
                "Margin показывает, какая доля ценности клиента сохраняется как эффект. "
                "Cost — стоимость удерживающего контакта, Success — вероятность успешного удержания."
            )

    training_schema = chosen.get("training_schema") or {}
    mapping_from_training = (
        training_schema.get("mapping")
        or (chosen.get("params_used") or {}).get("mapping_used")
        or {}
    )
    expected_cols = list(
        training_schema.get("expected_source_columns")
        or training_schema.get("required_source_columns")
        or [v for v in mapping_from_training.values() if v]
    )
    reverse_mapping = {v: k for k, v in mapping_from_training.items() if v}
    suggested_mapping = _smart_score_mapping(expected_cols, actual_cols, mapping_from_training, inspect_payload)

    with section_card(
        "Сопоставление колонок",
        "Подтвердите соответствие для каждой колонки, использованной при обучении модели.",
    ):
        auto_left, auto_right = st.columns([1, 2])
        with auto_left:
            if st.button("Заполнить предложенные соответствия", use_container_width=True):
                state.score_column_mapping = suggested_mapping
                _reset_score_outputs(state, clear_mapping=False)
                st.rerun()
        with auto_right:
            if inspect_payload:
                st.caption(
                    "Подсказки собраны по серверному `inspect` и по похожим названиям колонок. "
                    "Их всё равно нужно подтвердить перед запуском."
                )
            else:
                st.caption(
                    "Серверные подсказки недоступны, поэтому используется только совпадение по похожим названиям."
                )

        if inspect_error:
            st.warning(f"Не удалось получить серверные подсказки для автосопоставления: {inspect_error}")

        if expected_cols:
            with st.form("score_mapping_form"):
                next_mapping: dict[str, str] = {}
                st.markdown('<p class="ui-form-block-title">Колонки, использованные при обучении</p>', unsafe_allow_html=True)
                for source_col in expected_cols:
                    title, note = _mapping_label(source_col, reverse_mapping, role_meta)
                    render_field_intro(title, note, mapping_compact=True)
                    selected_value = state.score_column_mapping.get(source_col) or suggested_mapping.get(source_col)
                    options = [""] + actual_cols
                    default_index = options.index(selected_value) if selected_value in options else 0
                    chosen_value = st.selectbox(
                        f"Колонка для {title}",
                        options=options,
                        index=default_index,
                        key=f"score_map_all_{mapping_widget_prefix}_{source_col}",
                        label_visibility="collapsed",
                    )
                    if chosen_value:
                        next_mapping[source_col] = chosen_value

                submitted = st.form_submit_button("Сохранить сопоставление", type="primary", width="stretch")

            if submitted:
                if next_mapping != state.score_column_mapping:
                    state.score_column_mapping = next_mapping
                    _reset_score_outputs(state, clear_mapping=False)
                st.success("Сопоставление для прогноза сохранено.")

        mapped_count = len([col for col in expected_cols if state.score_column_mapping.get(col)])
        used_actual_cols = {v for v in state.score_column_mapping.values() if v}
        unused_actual_cols = [col for col in actual_cols if col not in used_actual_cols]
        s1, s2, s3 = st.columns(3)
        s1.metric("Ожидается колонок", str(len(expected_cols)))
        s2.metric("Сопоставлено", str(mapped_count))
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

        if not state.score_column_mapping:
            st.info("Сначала сохраните сопоставление колонок для прогноза.")

    if not state.score_column_mapping:
        page_nav(STEP_4_QUALITY, None)
        return

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
    with section_card("Проверка структуры файла", "Прогноз запускается только при полном совпадении схемы."):
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
                    scenario_params={
                        "business_margin": float(business_margin),
                        "business_cost": float(business_cost),
                        "business_success": float(business_success),
                    },
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

    with section_card("Статус", "Страница обновляется автоматически, пока прогноз обрабатывается на сервере."):
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

            n = state.score_result.get("n_scored")
            business_summary = state.score_result.get("business_summary") or {}
            top_clients = state.score_result.get("top_clients_preview") or []
            scenario_priority_previews = state.score_result.get("scenario_priority_previews") or {}
            scenario_rows = state.score_result.get("scenario_summary_preview") or business_summary.get("scenario_rows") or []
            scenario_df = _scenario_summary_table(scenario_rows)
            top_table = _top_clients_table(top_clients)
            scenario_client_tables = {
                name: _top_clients_table(rows)
                for name, rows in scenario_priority_previews.items()
                if rows
            }
            scenario_artifacts = {
                "conservative": "retention_priority_list_conservative.csv",
                "base": "retention_priority_list_base.csv",
                "optimistic": "retention_priority_list_optimistic.csv",
            }

            with section_card(
                "Готовые материалы",
                "Скачайте результат в том формате, который нужен конкретной роли: аналитик, руководитель или операционная команда.",
            ):
                st.metric("Строк в результате", str(n) if n is not None else "—")
                try:
                    csv_bytes = api.download_score_csv(state.score_id)
                    st.download_button(
                        label="Скачать полный CSV со скорингом",
                        data=csv_bytes,
                        file_name="scored_clients.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"Не удалось подготовить файл для кнопки: {e}")

                dl1, dl2 = st.columns(2)
                try:
                    priority_bytes = api.download_score_artifact(state.score_id, "retention_priority_list_base.csv")
                    dl1.download_button(
                        label="Скачать базовый список клиентов",
                        data=priority_bytes,
                        file_name="retention_priority_list_base.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as e:
                    dl1.warning(f"Не удалось подготовить base list: {e}")
                try:
                    summary_bytes = api.download_score_artifact(state.score_id, "scenario_summary.csv")
                    dl2.download_button(
                        label="Скачать сводку сценариев",
                        data=summary_bytes,
                        file_name="scenario_summary.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as e:
                    dl2.warning(f"Не удалось подготовить scenario summary: {e}")

                st.markdown("#### Списки клиентов по сценариям")
                scenario_cols = st.columns(3)
                for idx, scenario_name in enumerate(("conservative", "base", "optimistic")):
                    with scenario_cols[idx]:
                        try:
                            scenario_bytes = api.download_score_artifact(state.score_id, scenario_artifacts[scenario_name])
                            st.download_button(
                                label=f"Скачать {_scenario_label(scenario_name)} список",
                                data=scenario_bytes,
                                file_name=scenario_artifacts[scenario_name],
                                mime="text/csv",
                                use_container_width=True,
                                key=f"download_{scenario_name}_{state.score_id}",
                            )
                        except Exception as e:
                            st.warning(f"Не удалось подготовить {_scenario_label(scenario_name).lower()} список: {e}")

            result_tabs = st.tabs(["Рекомендация", "Сценарии", "Клиенты", "Полный результат"])

            with result_tabs[0]:
                with section_card(
                    "Что делать сейчас",
                    "Это основной управленческий вывод по базовому сценарию удержания.",
                ):
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Клиентов с EV > 0", str(business_summary.get("clients_with_positive_ev", "—")))
                    m2.metric("Суммарный положительный EV", _safe_money(business_summary.get("total_positive_ev")))
                    m3.metric("Макс. EV", _safe_money(business_summary.get("max_ev")))
                    m4.metric("Рекомендуемый top-k", str(business_summary.get("best_k", "—")))
                    st.write(f"**Вывод:** {_summary_message(business_summary)}")

                    sc_params = business_summary.get("scenario_params") or {}
                    if sc_params:
                        st.caption(
                            "Базовый сценарий: "
                            f"margin={float(sc_params.get('margin', 0)):.2f}, "
                            f"cost={float(sc_params.get('cost', 0)):.2f}, "
                            f"success={float(sc_params.get('success', 0)):.2f}"
                        )
                    if not business_summary.get("is_monetary", True):
                        st.info(
                            "Для текущего типа данных ценность клиента рассчитана в условных единицах активности, "
                            "поэтому EV тоже не интерпретируется как деньги."
                        )

                if not scenario_df.empty:
                    with section_card(
                        "Таблица сценариев",
                        "Сначала посмотрите, насколько устойчиво решение к изменению бизнес-допущений.",
                    ):
                        st.dataframe(scenario_df, width="stretch", hide_index=True)

                if not top_table.empty:
                    with section_card(
                        "Кого брать в кампанию удержания",
                        "Ниже показан базовый сценарий. На вкладке «Клиенты» доступны отдельные списки по всем трём сценариям.",
                    ):
                        st.dataframe(top_table.head(12), width="stretch", hide_index=True)

                risk_counts = business_summary.get("risk_segment_counts") or {}
                if risk_counts:
                    risk_df = pd.DataFrame(
                        [{"Сегмент": str(k), "Количество": int(v)} for k, v in risk_counts.items()]
                    )
                    render_bar_chart(
                        risk_df,
                        x_col="Сегмент",
                        y_col="Количество",
                        title="Распределение клиентов по сегментам риска",
                        x_title="Сегмент риска",
                        y_title="Количество клиентов",
                    )

            with result_tabs[1]:
                if not scenario_df.empty:
                    with section_card(
                        "Сценарии экономического эффекта",
                        "Этот блок показывает, насколько устойчиво решение к изменению бизнес-допущений.",
                    ):
                        st.dataframe(scenario_df, width="stretch", hide_index=True)
                        if {"Сценарий", "Макс. эффект"}.issubset(set(scenario_df.columns)):
                            chart_df = scenario_df.rename(columns={"Сценарий": "scenario", "Макс. эффект": "max_profit"})
                            render_bar_chart(
                                chart_df,
                                x_col="scenario",
                                y_col="max_profit",
                                title="Максимальный эффект по сценариям",
                                x_title="Сценарий",
                                y_title="Макс. эффект",
                            )
                else:
                    st.info("Сводка сценариев пока недоступна.")

                curve_rows = business_summary.get("scenario_curve_rows") or []
                if curve_rows:
                    curve_df = pd.DataFrame(curve_rows)
                    if "top_k" in curve_df.columns:
                        with section_card(
                            "Кумулятивный эффект по top-k",
                            "График показывает, сколько эффекта даёт расширение удерживающей кампании.",
                        ):
                            st.line_chart(curve_df.set_index("top_k"), width="stretch")

            with result_tabs[2]:
                if scenario_client_tables:
                    with section_card(
                        "Клиенты для удержания",
                        "Для каждого сценария список ранжируется отдельно, поэтому состав и порядок клиентов могут отличаться.",
                    ):
                        scenario_tabs = st.tabs(
                            [_scenario_label(name) for name in ("conservative", "base", "optimistic")]
                        )
                        for idx, scenario_name in enumerate(("conservative", "base", "optimistic")):
                            with scenario_tabs[idx]:
                                scenario_rows = scenario_priority_previews.get(scenario_name) or []
                                scenario_top_df = pd.DataFrame(scenario_rows)
                                scenario_top_table = scenario_client_tables.get(scenario_name, pd.DataFrame())
                                if not scenario_top_table.empty:
                                    label_col = _pick_client_label_col(scenario_top_df)
                                    ev_col = "scenario_ev" if "scenario_ev" in scenario_top_df.columns else _scenario_ev_col(scenario_name)
                                    if label_col and ev_col in scenario_top_df.columns:
                                        top_chart = scenario_top_df.head(15).copy()
                                        top_chart["_label"] = top_chart[label_col].astype(str)
                                        top_chart[ev_col] = pd.to_numeric(top_chart[ev_col], errors="coerce")
                                        render_bar_chart(
                                            top_chart,
                                            x_col="_label",
                                            y_col=ev_col,
                                            title=f"Лидеры по эффекту: {_scenario_label(scenario_name)} сценарий",
                                            x_title="Клиент",
                                            y_title="Ожидаемый эффект",
                                            horizontal=True,
                                            height=420,
                                        )
                                    st.dataframe(scenario_top_table, width="stretch", hide_index=True)
                                else:
                                    st.info(f"Список для сценария «{_scenario_label(scenario_name)}» пока недоступен.")
                else:
                    st.info("Сценарные списки клиентов пока недоступны.")

            with result_tabs[3]:
                preview = state.score_result.get("preview", [])
                if preview:
                    with section_card(
                        "Полный результат скоринга",
                        "Этот блок полезен для аналитической проверки и просмотра всех колонок результата.",
                    ):
                        st.dataframe(preview, width="stretch", height=400)
                else:
                    st.info("Предпросмотр результата пока недоступен.")

                st.info(
                    "**Как читать результат:** чем выше откалиброванная вероятность, тем выше риск; "
                    "чем выше EV_base, тем выгоднее включить клиента в удерживающую кампанию. "
                    "Сегмент риска даёт быстрое обобщение, а поясняющие колонки помогают интерпретации."
                )

        elif sc.get("status") == "failed":
            st.error(f"Прогноз не выполнен: {sc.get('error')}")
        else:
            st.info("Ожидание…")

    page_nav(STEP_4_QUALITY, None)
