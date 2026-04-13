# ui/pages/schema_mapping.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components.nav import page_nav
from ui.api_client import ApiClient
from ui.components.status_cards import column_risk_badge
from ui.state import get_state, read_csv_cached


def page():
    st.title("Сопоставление колонок")
    st.caption("Шаг 2: связываем ваши колонки с обязательными полями модели. Дополнительные колонки подключаем только контролируемо.")

    api = ApiClient.from_env()
    state = get_state()

    if not state.file_bytes:
        st.warning("Сначала загрузите данные на странице «Готовность данных».")
        return

    if state.contracts is None:
        try:
            state.contracts = api.contracts()
        except Exception as e:
            st.error(f"Не удалось получить contracts: {e}")
            return

    contracts = state.contracts["contracts"]
    contract = contracts[state.template]

    df = read_csv_cached(state.file_bytes)
    columns = list(df.columns)
    preview_df = pd.DataFrame((state.inspect or {}).get("preview", []))

    st.markdown("### Подсказка по колонкам из датасета")

    meta_rows = []
    for col in columns:
        examples = []
        if col in preview_df.columns:
            examples = (
                preview_df[col]
                .dropna()
                .astype(str)
                .head(5)
                .tolist()
            )

        meta_rows.append({
            "column": col,
            "dtype_preview": str(preview_df[col].dtype) if col in preview_df.columns else str(df[col].dtype),
            "examples": " | ".join(examples),
        })

    meta_df = pd.DataFrame(meta_rows)
    st.dataframe(meta_df, use_container_width=True)
    # suggested mapping from inspect
    suggested = (state.inspect or {}).get("suggested_mapping", {})

    required = contract["required"]
    optional = contract["optional"]
    st.info(
        "Тут нужно сопоставить поля модели с колонками Вашего датасета. "
        "Например: customer_id — это идентификатор клиента, "
        "event_time — дата операции, amount — сумма."
    )
    st.subheader("Обязательные поля")
    mapping = {}
    with st.form("mapping_form"):
        for field, meta in required.items():
            default = suggested.get(field, "")
            options = [""] + columns
            idx = options.index(default) if default in options else 0

            st.markdown(f"**{field}** — {meta['title']}")
            if meta.get("meaning"):
                st.caption(meta["meaning"])

            choice = st.selectbox(
                label=f"Выбери колонку для `{field}`",
                options=options,
                index=idx,
                key=f"required_{field}",
            )

            mapping[field] = choice if choice else None

            if choice and choice in preview_df.columns:
                ex = preview_df[choice].dropna().astype(str).head(5).tolist()
                if ex:
                    st.caption(f"Примеры значений: {' | '.join(ex)}")


        st.subheader("Необязательные поля (если есть)")
        for field, meta in optional.items():
            default = suggested.get(field, "")
            options = [""] + columns
            idx = options.index(default) if default in options else 0

            st.markdown(f"**{field}** — {meta['title']}")
            if meta.get("meaning"):
                st.caption(meta["meaning"])

            choice = st.selectbox(
                label=f"Выбери колонку для `{field}`",
                options=options,
                index=idx,
                key=f"optional_{field}",
            )

            mapping[field] = choice if choice else None

            if choice and choice in preview_df.columns:
                ex = preview_df[choice].dropna().astype(str).head(5).tolist()
                if ex:
                    st.caption(f"Примеры значений: {' | '.join(ex)}")

        submitted = st.form_submit_button("Сохранить сопоставление")

    if submitted:
        state.mapping = mapping
        state.extra_feature_cols = []
        st.success("Сопоставление сохранено.")

    if not state.mapping:
        st.info("Сохраните сопоставление, чтобы перейти к дополнительным колонкам.")
        return

    # show check: required filled
    missing_required = [f for f in required.keys() if not state.mapping.get(f)]
    if missing_required:
        st.error(f"Не заполнены обязательные поля: {missing_required}")
        return

    # Extra columns: controlled feature selection
    st.subheader("Дополнительные колонки")
    st.caption("Модель НЕ должна видеть признаки, содержащие таргет/будущее. Поэтому мы ограничиваем подключение.")

    used = {v for v in state.mapping.values() if v}
    contract_cols = set(required.keys()) | set(optional.keys())
    extra_cols = [c for c in columns if c not in used and c not in contract_cols]

    if not extra_cols:
        st.info("Лишних колонок не найдено — можно идти дальше.")
        state.extra_feature_cols = []
        return

    # backend audit via /inspect? (нет отдельного endpoint), поэтому делаем лёгкую эвристику на UI:
    def audit_ui(col: str) -> tuple[str, str]:
        name = col.lower()
        if any(x in name for x in ["churn", "target", "label", "outcome", "cancel", "retained", "lost", "status_after"]):
            return "block", "Похоже на таргет/исход/статус (высокий риск утечки)."
        nunique = df[col].nunique(dropna=True)
        non_null = df[col].notna().sum()
        unique_share = nunique / max(non_null, 1)
        if unique_share > 0.97 and nunique > 200:
            return "review", "Почти уникальные значения — риск переобучения (возможно технический ID)."
        if df[col].isna().mean() > 0.6:
            return "review", "Слишком много пропусков — может ухудшить качество."
        return "allow", "Выглядит безопасно по базовым признакам."

    audits = []
    for c in extra_cols:
        v, r = audit_ui(c)
        audits.append({"column": c, "verdict": v, "reason": r})

    # UI list
    for a in audits:
        st.write(f"{column_risk_badge(a['verdict'])}  `{a['column']}` — {a['reason']}")

    selectable = [a["column"] for a in audits if a["verdict"] in ("allow", "review")]
    blocked = [a["column"] for a in audits if a["verdict"] == "block"]

    if blocked:
        st.warning("Некоторые колонки заблокированы (риск утечки). Их нельзя подключить.")

    chosen = st.multiselect(
        "Выберите дополнительные колонки для агрегирования (не более 20)",
        options=selectable,
        default=(state.extra_feature_cols or []),
        max_selections=20,
        help="Мы не используем строки напрямую: преобразуем в числовые агрегаты (entropy/top1/missing/mean и т.д.).",
    )
    state.extra_feature_cols = chosen

    st.success(f"Дополнительные колонки выбраны: {len(chosen)}")

    st.info("Следующий шаг: выбрать режим и запустить обучение на странице «Обучение».")
    page_nav("Данные", "Обучение")