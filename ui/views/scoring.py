from __future__ import annotations

import hashlib
import io
import re

import pandas as pd
import streamlit as st

from ui.api_client import ApiClient
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
    if expected_cols and not state.score_column_mapping:
        state.score_column_mapping = suggested_mapping.copy()

    with section_card(
        "Сопоставление колонок",
        "Подтвердите соответствие для каждой колонки, использованной при обучении модели.",
    ):
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
            st.info(
                "В новом файле есть дополнительные колонки. Модель не строит прогноз по ним: они не участвуют в расчёте и не передаются в признаки прогноза: "
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
    if not schema.get("ok"):
        missing = schema.get("missing_mapping") or schema.get("missing_required") or schema.get("missing_expected") or []
        details = ", ".join(missing[:8]) if missing else "проверьте сопоставление обязательных колонок"
        st.warning(f"Для запуска прогноза нужно сопоставить обязательные колонки: {details}")

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
            st.rerun()
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

            with section_card(
                "Готовые материалы",
                "Скачайте единый архив: внутри DOCX-отчёт и CSV-файлы с результатами.",
            ):
                try:
                    zip_bytes = api.download_score_zip(state.score_id)
                    st.download_button(
                        label="Скачать результаты прогноза ZIP",
                        data=zip_bytes,
                        file_name=f"scoring_results_{state.score_id[:8]}.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"Не удалось подготовить ZIP: {e}")

        elif sc.get("status") == "failed":
            st.error(f"Прогноз не выполнен: {sc.get('error')}")
        else:
            st.info("Ожидание…")

    page_nav(STEP_4_QUALITY, None)
