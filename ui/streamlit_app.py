import json
import os
from io import BytesIO
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Churn/Retention Self-Serve",
    layout="wide",
)

st.title("Churn / Retention Self-Serve")

def api_post(url: str, data=None, files=None, timeout=180):
    return requests.post(url, data=data, files=files, timeout=timeout)


def api_get(url: str, timeout=60):
    return requests.get(url, timeout=timeout)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def read_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(raw_bytes))


def detect_id_like_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for col in df.columns:
        name = str(col).lower()

        if any(x in name for x in [
            "id", "customer", "client", "account", "subject",
            "invoice", "order", "transaction", "no"
        ]):
            out.append(col)
            continue

        try:
            nunique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
            if nunique_ratio > 0.95:
                out.append(col)
        except Exception:
            pass

    return list(dict.fromkeys(out))


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    id_like = set(detect_id_like_columns(df))

    for col in df.columns:
        if col in id_like:
            continue

        name = str(col).lower()

        if any(x in name for x in ["date", "time", "timestamp", "datetime"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                valid_share = parsed.notna().mean()
                if valid_share > 0.7:
                    years = parsed.dropna().dt.year
                    if not years.empty and years.between(2000, 2100).mean() > 0.8:
                        candidates.append(col)
                        continue
            except Exception:
                pass

        if df[col].dtype == "object":
            try:
                sample = df[col].dropna().astype(str).head(500)
                parsed = pd.to_datetime(sample, errors="coerce")
                valid_share = parsed.notna().mean()

                if valid_share > 0.8:
                    years = parsed.dropna().dt.year
                    if not years.empty and years.between(2000, 2100).mean() > 0.8:
                        candidates.append(col)
            except Exception:
                pass

    return list(dict.fromkeys(candidates))


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    out = []
    id_like = set(detect_id_like_columns(df))

    for col in df.columns:
        if col in id_like:
            continue

        try:
            s = pd.to_numeric(df[col], errors="coerce")
            valid_share = s.notna().mean()
            if valid_share < 0.8:
                continue

            nunique_ratio = s.nunique(dropna=True) / max(len(s.dropna()), 1)
            if nunique_ratio > 0.95:
                continue

            out.append(col)
        except Exception:
            pass

    return out


def render_quality_overview(quality: Dict[str, Any]):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк", quality.get("n_rows"))
    c2.metric("Колонок", quality.get("n_cols"))
    c3.metric("Дубликаты", quality.get("duplicates_full_rows"))
    c4.metric("Warnings", len(quality.get("warnings", [])))

    st.subheader("Пропуски по колонкам")
    null_df = pd.DataFrame({
        "column": list(quality.get("nulls_by_column", {}).keys()),
        "nulls": list(quality.get("nulls_by_column", {}).values()),
        "null_share": list(quality.get("null_share_by_column", {}).values()),
    })
    if not null_df.empty:
        st.dataframe(null_df, use_container_width=True)
    else:
        st.info("Нет данных о пропусках.")

    if quality.get("date_ranges"):
        st.subheader("Диапазоны дат")
        st.json(quality["date_ranges"])

    if quality.get("warnings"):
        st.subheader("Warnings")
        for w in quality["warnings"]:
            st.warning(w)


def render_transactions_overview(df: pd.DataFrame):
    st.subheader("Готовность transactional-данных")

    id_candidates = detect_id_like_columns(df)
    date_candidates = detect_date_columns(df)
    num_candidates = detect_numeric_columns(df)

    customer_col = next((c for c in df.columns if "customer" in c.lower()), id_candidates[0] if id_candidates else None)
    tx_col = next((c for c in df.columns if "invoice" in c.lower() or "transaction" in c.lower()), None)
    date_col = date_candidates[0] if date_candidates else None
    amount_col = next(
        (c for c in num_candidates if any(x in c.lower() for x in ["amount", "revenue", "price"])),
        num_candidates[0] if num_candidates else None
    )

    unique_customers = int(df[customer_col].nunique()) if customer_col and customer_col in df.columns else None
    unique_tx = int(df[tx_col].nunique()) if tx_col and tx_col in df.columns else None

    median_tx_per_customer = None
    share_customers_2plus = None
    if customer_col and tx_col:
        tx_per_cust = df.groupby(customer_col)[tx_col].nunique()
        median_tx_per_customer = float(tx_per_cust.median()) if not tx_per_cust.empty else None
        share_customers_2plus = float((tx_per_cust >= 2).mean()) if not tx_per_cust.empty else None

    date_range_str = "—"
    if date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not parsed.empty:
            date_range_str = f"{parsed.min().date()} — {parsed.max().date()}"

    c1, c2, c3 = st.columns(3)
    c1.metric("Строк", len(df))
    c2.metric("Клиентов", unique_customers if unique_customers is not None else "—")
    c3.metric("Транзакций", unique_tx if unique_tx is not None else "—")

    c4, c5, c6 = st.columns(3)
    c4.metric("Период", date_range_str)
    c5.metric("Median tx/client", round(median_tx_per_customer, 2) if median_tx_per_customer is not None else "—")
    c6.metric("Клиентов с 2+ покупками", f"{share_customers_2plus * 100:.1f}%" if share_customers_2plus is not None else "—")

    left, right = st.columns(2)

    with left:
        st.subheader("Транзакции по месяцам")
        if date_col:
            parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not parsed.empty:
                ts = parsed.dt.to_period("M").astype(str).value_counts().sort_index()
                st.bar_chart(ts)
            else:
                st.info("Не удалось построить график по времени.")
        else:
            st.info("Дата-колонка не распознана.")

    with right:
        st.subheader("Выручка по месяцам")
        if date_col and amount_col:
            tmp = df[[date_col, amount_col]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
            tmp = tmp.dropna()
            if not tmp.empty:
                tmp["month"] = tmp[date_col].dt.to_period("M").astype(str)
                monthly = tmp.groupby("month")[amount_col].sum().sort_index()
                st.bar_chart(monthly)
            else:
                st.info("Не удалось построить график выручки.")
        else:
            st.info("Не удалось определить колонки даты и суммы.")

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Покупок на клиента")
        if customer_col and tx_col:
            tx_per_cust = df.groupby(customer_col)[tx_col].nunique()
            counts, edges = np.histogram(tx_per_cust.values, bins=20)
            hist_df = pd.DataFrame({"bin_left": edges[:-1], "count": counts})
            st.bar_chart(hist_df.set_index("bin_left")["count"])
        else:
            st.info("Не удалось определить customer_id / transaction_id.")

    with right2:
        st.subheader("Выручка на клиента")
        if customer_col and amount_col:
            tmp = df[[customer_col, amount_col]].copy()
            tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
            tmp = tmp.dropna()
            if not tmp.empty:
                rev_per_cust = tmp.groupby(customer_col)[amount_col].sum()
                counts, edges = np.histogram(rev_per_cust.values, bins=20)
                hist_df = pd.DataFrame({"bin_left": edges[:-1], "count": counts})
                st.bar_chart(hist_df.set_index("bin_left")["count"])
            else:
                st.info("Нет валидных значений для выручки на клиента.")
        else:
            st.info("Не удалось определить customer_id / amount.")


def render_subscriptions_overview(df: pd.DataFrame):
    st.subheader("Готовность subscription-данных")
    st.dataframe(df.head(20), use_container_width=True)


def render_events_overview(df: pd.DataFrame):
    st.subheader("Готовность event-данных")
    st.dataframe(df.head(20), use_container_width=True)


def get_unused_columns(all_columns: List[str], mapping_result: Dict[str, Any]) -> List[str]:
    used = {v for v in mapping_result.values() if v}
    return [c for c in all_columns if c not in used]


def risk_segment_preview(df: pd.DataFrame):
    if "risk_segment" in df.columns:
        st.subheader("Распределение risk_segment")
        st.bar_chart(df["risk_segment"].astype(str).value_counts())


CONTRACTS = {
    "transactions": {
        "required": {
            "customer_id": "Идентификатор клиента",
            "transaction_id": "Идентификатор транзакции/чека",
            "event_time": "Дата и время транзакции",
            "amount": "Денежная сумма транзакции",
        },
        "optional": {
            "item_id": "Идентификатор товара",
            "quantity": "Количество",
            "country": "Страна/регион",
            "unit_price": "Цена единицы",
            "is_cancellation": "Флаг отмены/возврата",
            "promo_code": "Промокод",
        }
    },
    "subscriptions": {
        "required": {
            "account_id": "Идентификатор аккаунта",
            "period_start": "Начало периода",
            "period_end": "Конец периода",
            "mrr": "Регулярная выручка",
            "subscription_status": "Статус подписки",
        },
        "optional": {
            "churn_date": "Дата оттока",
            "plan_name": "Название тарифа",
            "seats_purchased": "Куплено мест",
            "seats_used": "Использовано мест",
        }
    },
    "events": {
        "required": {
            "subject_id": "Идентификатор пользователя/субъекта",
            "event_time": "Дата и время события",
            "event_name": "Название события",
        },
        "optional": {
            "account_id": "Идентификатор аккаунта",
            "platform": "Платформа/устройство",
        }
    }
}


# =========================
# session init
# =========================

if "working_csv_bytes" not in st.session_state:
    st.session_state["working_csv_bytes"] = None

if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None

if "inspect_data" not in st.session_state:
    st.session_state["inspect_data"] = None

if "mapping_result" not in st.session_state:
    st.session_state["mapping_result"] = {}

if "extra_feature_columns" not in st.session_state:
    st.session_state["extra_feature_columns"] = []

if "job_info" not in st.session_state:
    st.session_state["job_info"] = {}

if "job_result" not in st.session_state:
    st.session_state["job_result"] = None

if "job_status" not in st.session_state:
    st.session_state["job_status"] = None

if "models_cache" not in st.session_state:
    st.session_state["models_cache"] = None

if "score_result" not in st.session_state:
    st.session_state["score_result"] = None


# =========================
# tabs
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Готовность данных",
    "2. Контракт и схема",
    "3. Обучение",
    "4. Мониторинг и результаты",
    "5. Модели и прогноз",
])


# =========================
# TAB 1
# =========================

with tab1:
    st.header("Готовность данных")

    template = st.selectbox(
        "Шаблон данных",
        ["transactions", "subscriptions", "events"],
        key="template",
    )

    uploaded = st.file_uploader(
        "Загрузите CSV",
        type=["csv"],
        key="uploaded_file",
    )

    if uploaded is not None:
        current_bytes = uploaded.getvalue()
        current_name = uploaded.name

        if (
            st.session_state["working_csv_bytes"] is None
            or st.session_state["uploaded_file_name"] != current_name
        ):
            st.session_state["working_csv_bytes"] = current_bytes
            st.session_state["uploaded_file_name"] = current_name
            st.session_state["inspect_data"] = None
            st.session_state["mapping_result"] = {}

    if st.session_state["working_csv_bytes"] is not None:
        raw_df = read_csv_bytes(st.session_state["working_csv_bytes"])

        dup_count = int(raw_df.duplicated().sum())
        if dup_count > 0:
            st.warning(f"Найдено полных дубликатов строк: {dup_count}")
            if st.button("Удалить полные дубликаты для текущей сессии"):
                raw_df = raw_df.drop_duplicates().copy()
                st.session_state["working_csv_bytes"] = df_to_csv_bytes(raw_df)
                st.session_state["inspect_data"] = None
                st.session_state["mapping_result"] = {}
                st.success("Полные дубликаты удалены из рабочей версии датасета.")

        try:
            if template == "transactions":
                render_transactions_overview(raw_df)
            elif template == "subscriptions":
                render_subscriptions_overview(raw_df)
            else:
                render_events_overview(raw_df)
        except Exception as e:
            st.error(f"Ошибка при построении обзора данных: {e}")

        st.subheader("Preview")
        st.dataframe(raw_df.head(20), use_container_width=True)

        if st.button("Подготовить quality report и suggested mapping"):
            files = {
                "file": (
                    st.session_state["uploaded_file_name"],
                    st.session_state["working_csv_bytes"],
                    "text/csv",
                )
            }
            resp = api_post(
                f"{API_URL}/inspect",
                data={"template": template},
                files=files,
                timeout=180,
            )
            if resp.status_code == 200:
                st.session_state["inspect_data"] = resp.json()
                st.success("Файл проанализирован.")
            else:
                st.error(resp.text)

    if st.session_state["inspect_data"]:
        st.divider()
        st.subheader("Quality report")
        try:
            render_quality_overview(st.session_state["inspect_data"]["quality_report"])
        except Exception as e:
            st.error(f"Ошибка при отображении quality report: {e}")


# =========================
# TAB 2
# =========================

with tab2:
    st.header("Контракт и схема")

    inspect_data = st.session_state.get("inspect_data")
    current_template = st.session_state.get("template", "transactions")

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Обязательные поля")
        req_df = pd.DataFrame({
            "canonical_field": list(CONTRACTS[current_template]["required"].keys()),
            "description": list(CONTRACTS[current_template]["required"].values()),
        })
        st.dataframe(req_df, use_container_width=True)

        st.subheader("Рекомендованные поля")
        opt_df = pd.DataFrame({
            "canonical_field": list(CONTRACTS[current_template]["optional"].keys()),
            "description": list(CONTRACTS[current_template]["optional"].values()),
        })
        st.dataframe(opt_df, use_container_width=True)

        st.info(
            "Дополнительные пользовательские колонки можно сохранить в схеме проекта, "
            "но они не будут автоматически включены в модель без отдельной проверки на утечки и пригодность."
        )

    with right:
        if not inspect_data:
            st.info("Сначала проанализируйте файл на первой странице.")
        else:
            columns = inspect_data["columns"]
            suggested = inspect_data["suggested_mapping"]

            st.subheader("Сопоставление колонок")
            mapping_result = {}

            for canon, suggested_col in suggested.items():
                options = ["-- не выбрано --"] + columns
                default_idx = options.index(suggested_col) if suggested_col in options else 0
                selected = st.selectbox(
                    canon,
                    options,
                    index=default_idx,
                    key=f"map_{canon}",
                )
                mapping_result[canon] = None if selected == "-- не выбрано --" else selected

            st.session_state["mapping_result"] = mapping_result

            st.divider()
            st.subheader("Дополнительные колонки")
            unused_columns = get_unused_columns(columns, mapping_result)

            extra_feature_columns = st.multiselect(
                "Сохранить дополнительные колонки в схеме проекта",
                options=unused_columns,
                default=st.session_state.get("extra_feature_columns", []),
                key="extra_columns_select",
            )
            st.session_state["extra_feature_columns"] = extra_feature_columns

            st.subheader("Итоговая схема")
            st.json({
                "mapping": mapping_result,
                "extra_feature_columns": extra_feature_columns,
            })


# =========================
# TAB 3
# =========================

with tab3:
    st.header("Обучение")

    inspect_data = st.session_state.get("inspect_data")
    mapping_result = st.session_state.get("mapping_result", {})

    if inspect_data is None or not mapping_result:
        st.info("Сначала загрузите файл и заполните схему.")
    else:
        st.subheader("Полный автоматический анализ")
        st.caption(
            "Система сама выполнит полный перебор разумных горизонтов, моделей и калибровок, "
            "сравнит качество, устойчивость и бизнес-эффект, а затем выберет лучший вариант."
        )

        model_name = st.text_input(
            "Название проекта / модели",
            value="retention_model_full_analysis",
            key="model_name",
        )

        params = {
            "horizon_days_grid": [30, 60, 90],
            "history_days_grid": [180, 365],
            "step_days_grid": [30],
            "model_kind_grid": ["logreg", "sklearn_gbdt", "lightgbm", "catboost"],
            "calibration_grid": ["sigmoid", "isotonic"],
            "selection_metric": "pr_auc",
            "use_class_weight": True,
            "model_name": model_name,
            "extra_feature_columns": st.session_state.get("extra_feature_columns", []),
        }

        with st.expander("Дополнительные настройки"):
            params["horizon_days_grid"] = st.multiselect(
                "Горизонты прогноза",
                [30, 60, 90, 120],
                default=params["horizon_days_grid"],
            )
            params["history_days_grid"] = st.multiselect(
                "Окно истории",
                [90, 180, 365],
                default=params["history_days_grid"],
            )
            params["step_days_grid"] = st.multiselect(
                "Шаг anchor",
                [7, 14, 30],
                default=params["step_days_grid"],
            )

        if st.button("Запустить полный анализ"):
            if st.session_state["working_csv_bytes"] is None:
                st.error("Нет рабочего CSV. Перезагрузите файл на первой странице.")
            else:
                files = {
                    "file": (
                        st.session_state["uploaded_file_name"],
                        st.session_state["working_csv_bytes"],
                        "text/csv",
                    )
                }

                payload = {
                    "template": st.session_state["template"],
                    "mapping_json": json.dumps(mapping_result, ensure_ascii=False),
                    "params_json": json.dumps(params, ensure_ascii=False),
                }

                resp = api_post(
                    f"{API_URL}/jobs",
                    data=payload,
                    files=files,
                    timeout=180,
                )

                if resp.status_code == 200:
                    st.session_state["job_info"] = resp.json()
                    st.success(resp.json())
                else:
                    st.error(resp.text)

        if st.session_state.get("job_info"):
            st.subheader("Текущая job")
            st.json(st.session_state["job_info"])


# =========================
# TAB 4
# =========================

with tab4:
    st.header("Мониторинг и результаты")

    default_job_id = st.session_state.get("job_info", {}).get("job_id", "")
    job_id = st.text_input("job_id", value=default_job_id, key="result_job_id")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Проверить статус"):
            if not job_id:
                st.warning("Введите job_id")
            else:
                resp = api_get(f"{API_URL}/jobs/{job_id}")
                if resp.status_code == 200:
                    status_data = resp.json()
                    st.session_state["job_status"] = status_data
                else:
                    st.error(resp.text)

    with c2:
        if st.button("Загрузить результат"):
            if not job_id:
                st.warning("Введите job_id")
            else:
                resp = api_get(f"{API_URL}/jobs/{job_id}/result")
                if resp.status_code == 200:
                    st.session_state["job_result"] = resp.json()
                else:
                    st.error(resp.text)

    if st.session_state.get("job_status"):
        status_data = st.session_state["job_status"]

        st.subheader("Статус обучения")

        c1, c2, c3 = st.columns(3)
        c1.metric("Status", status_data.get("status", "—"))
        c2.metric("Stage", status_data.get("stage", "—"))
        c3.metric("Progress", f"{status_data.get('progress', 0)}%")

        progress = status_data.get("progress")
        if progress is not None:
            st.progress(int(progress))

        current_exp = status_data.get("current_experiment")
        if current_exp:
            st.subheader("Текущий эксперимент")
            st.json(current_exp)

        best_so_far = status_data.get("best_so_far")
        if best_so_far:
            st.subheader("Лучший результат на текущий момент")
            st.json(best_so_far)

        last_finished = status_data.get("last_finished")
        if last_finished:
            st.subheader("Последний завершённый эксперимент")
            st.json(last_finished)

    result = st.session_state.get("job_result")
    if result:
        st.divider()
        st.subheader("Победившая конфигурация")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Template", result.get("template", "—"))
        c2.metric("Snapshots", result.get("n_snapshots", "—"))
        c3.metric("Features", result.get("feature_count", "—"))
        c4.metric("Best k", result.get("business_metrics", {}).get("base_best_k", "—"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("PR-AUC", round(result.get("test_metrics_cal", {}).get("pr_auc", 0), 4))
        c6.metric("ROC-AUC", round(result.get("test_metrics_cal", {}).get("roc_auc", 0), 4))
        c7.metric("Brier", round(result.get("test_metrics_cal", {}).get("brier", 0), 4))
        c8.metric("MaxProfit", round(result.get("business_metrics", {}).get("base_max_profit", 0), 2))

        st.subheader("Параметры победителя")
        st.json(result.get("params_used", {}))

        st.subheader("Калибровка")
        st.json(result.get("calibration", {}))

        st.subheader("Walk-forward")
        st.json(result.get("walk_forward", {}))

        st.subheader("Артефакты")
        st.json(result.get("artifacts", {}))

        download_job_id = st.text_input("job_id для скачивания ZIP", value=job_id, key="download_job_id")
        if st.button("Скачать report.zip"):
            r = api_get(f"{API_URL}/jobs/{download_job_id}/download", timeout=120)
            if r.status_code == 200:
                st.download_button(
                    "Download report.zip",
                    data=r.content,
                    file_name="report.zip",
                )
            else:
                st.error(r.text)


# =========================
# TAB 5
# =========================

with tab5:
    st.header("Модели и прогноз")

    if st.button("Обновить список моделей"):
        resp = api_get(f"{API_URL}/models")
        if resp.status_code == 200:
            st.session_state["models_cache"] = resp.json()["models"]
        else:
            st.error(resp.text)

    models = st.session_state.get("models_cache")
    if models:
        models_df = pd.DataFrame(models)
        st.subheader("Реестр моделей")
        st.dataframe(models_df, use_container_width=True)

        model_options = {
            f"{m['job_id']} | {m.get('template')} | {m.get('params_used', {}).get('model_kind', '')}": m
            for m in models
        }

        selected_label = st.selectbox(
            "Выберите модель",
            list(model_options.keys()),
        )
        selected_model = model_options[selected_label]

        st.subheader("Детали модели")
        st.json(selected_model)

        st.divider()
        st.subheader("Прогноз на новых данных")

        score_file = st.file_uploader(
            "Загрузите CSV для scoring",
            type=["csv"],
            key="score_file",
        )

        if score_file is not None and st.button("Запустить прогноз"):
            files = {"file": (score_file.name, score_file.getvalue(), "text/csv")}
            resp = api_post(
                f"{API_URL}/score",
                data={"bundle_dir": selected_model["bundle_dir"]},
                files=files,
                timeout=180,
            )

            if resp.status_code == 200:
                score_result = resp.json()
                st.session_state["score_result"] = score_result
            else:
                st.error(resp.text)

    score_result = st.session_state.get("score_result")
    if score_result:
        st.subheader("Результат прогноза")
        st.json({
            "n_rows": score_result.get("n_rows"),
            "output_csv": score_result.get("output_csv"),
        })

        preview_df = pd.DataFrame(score_result.get("preview", []))
        if not preview_df.empty:
            st.dataframe(preview_df, use_container_width=True)
            risk_segment_preview(preview_df)




'''import json
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Churn/Retention Self-Serve",
    layout="wide",
)

st.title("Churn / Retention Self-Serve")


def api_post(url: str, data=None, files=None, timeout=180):
    return requests.post(url, data=data, files=files, timeout=timeout)


def api_get(url: str, timeout=60):
    return requests.get(url, timeout=timeout)


def safe_read_csv(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def detect_id_like_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for col in df.columns:
        name = str(col).lower()

        if any(x in name for x in [
            "id", "customer", "client", "account", "subject",
            "invoice", "order", "transaction", "no"
        ]):
            out.append(col)
            continue

        try:
            nunique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
            if nunique_ratio > 0.95:
                out.append(col)
        except Exception:
            pass

    return list(dict.fromkeys(out))


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    id_like = set(detect_id_like_columns(df))

    for col in df.columns:
        if col in id_like:
            continue

        name = str(col).lower()

        if any(x in name for x in ["date", "time", "timestamp", "datetime"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                valid_share = parsed.notna().mean()
                if valid_share > 0.7:
                    years = parsed.dropna().dt.year
                    if not years.empty and years.between(2000, 2100).mean() > 0.8:
                        candidates.append(col)
                        continue
            except Exception:
                pass

        if df[col].dtype == "object":
            try:
                sample = df[col].dropna().astype(str).head(500)
                parsed = pd.to_datetime(sample, errors="coerce")
                valid_share = parsed.notna().mean()

                if valid_share > 0.8:
                    years = parsed.dropna().dt.year
                    if not years.empty and years.between(2000, 2100).mean() > 0.8:
                        candidates.append(col)
            except Exception:
                pass

    return list(dict.fromkeys(candidates))


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    out = []
    id_like = set(detect_id_like_columns(df))

    for col in df.columns:
        if col in id_like:
            continue

        try:
            s = pd.to_numeric(df[col], errors="coerce")
            valid_share = s.notna().mean()
            if valid_share < 0.8:
                continue

            nunique_ratio = s.nunique(dropna=True) / max(len(s.dropna()), 1)
            if nunique_ratio > 0.95:
                continue

            out.append(col)
        except Exception:
            pass

    return out


def render_quality_overview(quality: Dict[str, Any]):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк", quality.get("n_rows"))
    c2.metric("Колонок", quality.get("n_cols"))
    c3.metric("Дубликаты", quality.get("duplicates_full_rows"))
    c4.metric("Warnings", len(quality.get("warnings", [])))

    st.subheader("Пропуски по колонкам")
    null_df = pd.DataFrame({
        "column": list(quality.get("nulls_by_column", {}).keys()),
        "nulls": list(quality.get("nulls_by_column", {}).values()),
        "null_share": list(quality.get("null_share_by_column", {}).values()),
    })
    if not null_df.empty:
        st.dataframe(null_df, use_container_width=True)
    else:
        st.info("Нет данных о пропусках.")

    if quality.get("date_ranges"):
        st.subheader("Диапазоны дат")
        st.json(quality["date_ranges"])

    if quality.get("warnings"):
        st.subheader("Warnings")
        for w in quality["warnings"]:
            st.warning(w)


def render_raw_overview(df: pd.DataFrame):
    st.subheader("Основные показатели")

    n_rows = len(df)
    n_cols = len(df.columns)
    dup = int(df.duplicated().sum())
    null_share = float(df.isna().sum().sum() / max(df.shape[0] * df.shape[1], 1))

    id_cols = detect_id_like_columns(df)
    date_cols = detect_date_columns(df)
    num_cols = detect_numeric_columns(df)

    unique_entities = None
    if id_cols:
        try:
            unique_entities = int(df[id_cols[0]].nunique(dropna=True))
        except Exception:
            unique_entities = None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Строк", n_rows)
    c2.metric("Колонок", n_cols)
    c3.metric("Дубликаты", dup)
    c4.metric("Пропуски %", f"{null_share * 100:.1f}%")
    c5.metric("Уникальных ID", unique_entities if unique_entities is not None else "—")

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Данные во времени")
        if date_cols:
            col = date_cols[0]
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                parsed = parsed.dropna()

                if not parsed.empty:
                    tmp = (
                        parsed.dt.to_period("M")
                        .astype(str)
                        .value_counts()
                        .sort_index()
                        .reset_index()
                    )
                    tmp.columns = ["month", "count"]
                    st.bar_chart(tmp.set_index("month")["count"])
                    st.caption(f"Диапазон: {str(parsed.min())} — {str(parsed.max())}")
                else:
                    st.info("Не удалось получить валидные даты.")
            except Exception as e:
                st.warning(f"Не удалось построить график по времени: {e}")
        else:
            st.info("Дата-колонка не распознана на сыром CSV.")

    with right:
        st.subheader("Распределение числовой колонки")
        if num_cols:
            col = num_cols[0]
            try:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if not vals.empty:
                    counts, edges = np.histogram(vals, bins=20)
                    hist_df = pd.DataFrame({
                        "bin_left": edges[:-1],
                        "count": counts,
                    })
                    st.bar_chart(hist_df.set_index("bin_left")["count"])
                else:
                    st.info("Нет числовых значений для построения графика.")
            except Exception as e:
                st.warning(f"Не удалось построить числовой график: {e}")
        else:
            st.info("Подходящая числовая колонка не распознана.")

    if id_cols:
        st.subheader(f"Топ по числу записей: {id_cols[0]}")
        try:
            vc = df[id_cols[0]].astype(str).value_counts().head(20)
            top_df = vc.reset_index()
            top_df.columns = ["entity", "count"]
            st.bar_chart(top_df.set_index("entity")["count"])
        except Exception as e:
            st.warning(f"Не удалось построить график top entities: {e}")


def default_training_params() -> Dict[str, Any]:
    return {
        "horizon_days_grid": [30, 60, 90],
        "history_days_grid": [180, 365],
        "step_days_grid": [30],
        "model_kind_grid": ["logreg", "sklearn_gbdt", "lightgbm", "catboost"],
        "calibration_grid": ["sigmoid", "isotonic"],
        "selection_metric": "pr_auc",
        "use_class_weight": True,
    }


def get_unused_columns(all_columns: List[str], mapping_result: Dict[str, Any]) -> List[str]:
    used = {v for v in mapping_result.values() if v}
    return [c for c in all_columns if c not in used]


def risk_segment_preview(df: pd.DataFrame):
    if "risk_segment" in df.columns:
        st.subheader("Распределение risk_segment")
        st.bar_chart(df["risk_segment"].astype(str).value_counts())

def render_transactions_overview(df: pd.DataFrame):
    st.subheader("Готовность transactional-данных")

    id_candidates = detect_id_like_columns(df)
    date_candidates = detect_date_columns(df)
    num_candidates = detect_numeric_columns(df)

    customer_col = next((c for c in df.columns if "customer" in c.lower()), id_candidates[0] if id_candidates else None)
    tx_col = next((c for c in df.columns if "invoice" in c.lower() or "transaction" in c.lower()), None)
    date_col = date_candidates[0] if date_candidates else None
    amount_col = next((c for c in num_candidates if "amount" in c.lower() or "revenue" in c.lower() or "price" in c.lower()), num_candidates[0] if num_candidates else None)

    unique_customers = int(df[customer_col].nunique()) if customer_col in df.columns else None
    unique_tx = int(df[tx_col].nunique()) if tx_col and tx_col in df.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк", len(df))
    c2.metric("Клиентов", unique_customers if unique_customers is not None else "—")
    c3.metric("Транзакций", unique_tx if unique_tx is not None else "—")
    c4.metric("Колонок", len(df.columns))

    left, right = st.columns(2)

    with left:
        if date_col:
            parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not parsed.empty:
                ts = parsed.dt.to_period("M").astype(str).value_counts().sort_index()
                st.subheader("Транзакции по месяцам")
                st.bar_chart(ts)
                st.caption(f"Период покрытия: {parsed.min()} — {parsed.max()}")

    with right:
        if amount_col:
            vals = pd.to_numeric(df[amount_col], errors="coerce").dropna()
            if not vals.empty:
                counts, edges = np.histogram(vals, bins=20)
                hist_df = pd.DataFrame({"bin_left": edges[:-1], "count": counts})
                st.subheader("Распределение суммы")
                st.bar_chart(hist_df.set_index("bin_left")["count"])

    if customer_col and tx_col:
        cust_counts = df.groupby(customer_col)[tx_col].nunique().sort_values(ascending=False)
        st.subheader("Покупок на клиента")
        counts, edges = np.histogram(cust_counts.values, bins=20)
        hist_df = pd.DataFrame({"bin_left": edges[:-1], "count": counts})
        st.bar_chart(hist_df.set_index("bin_left")["count"])

def render_subscriptions_overview(df: pd.DataFrame):
    st.subheader("Готовность subscription-данных")
    st.dataframe(df.head(20), use_container_width=True)


def render_events_overview(df: pd.DataFrame):
    st.subheader("Готовность event-данных")
    st.dataframe(df.head(20), use_container_width=True)


# =========================
# session init
# =========================

if "inspect_data" not in st.session_state:
    st.session_state["inspect_data"] = None

if "mapping_result" not in st.session_state:
    st.session_state["mapping_result"] = {}

if "extra_feature_columns" not in st.session_state:
    st.session_state["extra_feature_columns"] = []

if "job_info" not in st.session_state:
    st.session_state["job_info"] = {}

if "job_result" not in st.session_state:
    st.session_state["job_result"] = None

if "job_status" not in st.session_state:
    st.session_state["job_status"] = None

if "models_cache" not in st.session_state:
    st.session_state["models_cache"] = None

if "score_result" not in st.session_state:
    st.session_state["score_result"] = None


# =========================
# tabs
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Обзор данных",
    "2. Схема и колонки",
    "3. Обучение",
    "4. Результаты",
    "5. Модели и прогноз",
])


# =========================
# TAB 1
# =========================

with tab1:
    st.header("Обзор данных")

    template = st.selectbox(
        "Шаблон данных",
        ["transactions", "subscriptions", "events"],
        key="template",
    )

    uploaded = st.file_uploader(
        "Загрузите CSV",
        type=["csv"],
        key="uploaded_file",
    )

    if uploaded is not None:
        raw_df = safe_read_csv(uploaded)

        try:
            if template == "transactions":
                render_transactions_overview(raw_df)
            elif template == "subscriptions":
                render_subscriptions_overview(raw_df)
            else:
                render_events_overview(raw_df)
        except Exception as e:
            st.error(f"Ошибка при построении обзора данных: {e}")

        if st.button("Подготовить quality report и suggested mapping"):
            files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
            resp = api_post(
                f"{API_URL}/inspect",
                data={"template": template},
                files=files,
                timeout=180,
            )
            if resp.status_code == 200:
                st.session_state["inspect_data"] = resp.json()
                st.success("Файл проанализирован.")
            else:
                st.error(resp.text)

    if st.session_state["inspect_data"]:
        st.divider()
        st.subheader("Quality report")
        try:
            render_quality_overview(st.session_state["inspect_data"]["quality_report"])
        except Exception as e:
            st.error(f"Ошибка при отображении quality report: {e}")


# =========================
# TAB 2
# =========================

with tab2:
    st.header("Схема и колонки")

    inspect_data = st.session_state.get("inspect_data")
    if not inspect_data:
        st.info("Сначала проанализируйте файл на первой странице.")
    else:
        columns = inspect_data["columns"]
        suggested = inspect_data["suggested_mapping"]

        st.subheader("Обязательные и рекомендованные поля")
        st.caption("Сначала подтвердите основные поля, затем при необходимости добавьте дополнительные колонки.")

        mapping_result = {}

        for canon, suggested_col in suggested.items():
            options = ["-- не выбрано --"] + columns
            default_idx = options.index(suggested_col) if suggested_col in options else 0
            selected = st.selectbox(
                canon,
                options,
                index=default_idx,
                key=f"map_{canon}",
            )
            mapping_result[canon] = None if selected == "-- не выбрано --" else selected

        st.session_state["mapping_result"] = mapping_result

        st.divider()
        st.subheader("Дополнительные колонки")
        unused_columns = get_unused_columns(columns, mapping_result)

        extra_feature_columns = st.multiselect(
            "Выберите дополнительные колонки, которые хотите сохранить в схеме проекта",
            options=unused_columns,
            default=st.session_state.get("extra_feature_columns", []),
            key="extra_columns_select",
        )
        st.session_state["extra_feature_columns"] = extra_feature_columns

        st.info(
            "Дополнительные колонки будут сохранены в конфигурации проекта. "
            "Их можно использовать в расширенном feature engineering на следующем этапе."
        )

        st.subheader("Итоговая схема")
        st.json({
            "mapping": mapping_result,
            "extra_feature_columns": extra_feature_columns,
        })


# =========================
# TAB 3
# =========================

with tab3:
    st.header("Обучение")

    inspect_data = st.session_state.get("inspect_data")
    mapping_result = st.session_state.get("mapping_result", {})

    if inspect_data is None or not mapping_result:
        st.info("Сначала загрузите файл и заполните схему.")
    else:
        uploaded = st.session_state.get("uploaded_file")

        st.subheader("Автоматический подбор")
        st.caption(
            "Система сама переберет модели, горизонты и калибровки, "
            "сравнит результаты и выберет лучший вариант."
        )
        training_mode = st.selectbox(
            "Режим обучения",
            ["Quick", "Standard", "Deep", "Custom"],
            index=1,
        )
        model_name = st.text_input(
            "Название модели / проекта",
            value="retention_model_v1",
            key="model_name",
        )

        horizons = st.multiselect(
            "Горизонты прогноза",
            [30, 60, 90, 120],
            default=[30, 60, 90],
            key="horizons_auto",
        )

        history_grid = st.multiselect(
            "Окно истории",
            [90, 180, 365],
            default=[180, 365],
            key="history_auto",
        )

        step_grid = st.multiselect(
            "Шаг anchor",
            [7, 14, 30],
            default=[30],
            key="step_auto",
        )

        with st.expander("Расширенные настройки"):
            model_kind_grid = st.multiselect(
                "Какие модели включить",
                ["logreg", "sklearn_gbdt", "lightgbm", "catboost", "random_forest"],
                default=["logreg", "sklearn_gbdt", "lightgbm", "catboost"],
                key="model_kind_grid_adv",
            )

            calibration_grid = st.multiselect(
                "Какие калибровки включить",
                ["sigmoid", "isotonic"],
                default=["sigmoid", "isotonic"],
                key="calibration_grid_adv",
            )

            selection_metric = st.selectbox(
                "Главная метрика выбора",
                ["pr_auc", "roc_auc", "brier", "base_max_profit"],
                index=0,
                key="selection_metric_adv",
            )

            use_class_weight = st.checkbox(
                "Использовать class weights",
                value=True,
                key="use_class_weight_adv",
            )

        if training_mode == "Quick":
            params = {
                "horizon_days_grid": [30],
                "history_days_grid": [180],
                "step_days_grid": [30],
                "model_kind_grid": ["logreg", "lightgbm"],
                "calibration_grid": ["sigmoid"],
                "selection_metric": "pr_auc",
                "use_class_weight": True,
            }

        elif training_mode == "Standard":
            params = {
                "horizon_days_grid": [30, 60, 90],
                "history_days_grid": [180, 365],
                "step_days_grid": [30],
                "model_kind_grid": ["logreg", "sklearn_gbdt", "lightgbm", "catboost"],
                "calibration_grid": ["sigmoid", "isotonic"],
                "selection_metric": "pr_auc",
                "use_class_weight": True,
            }

        elif training_mode == "Deep":
            params = {
                "horizon_days_grid": [30, 60, 90, 120],
                "history_days_grid": [180, 365],
                "step_days_grid": [14, 30],
                "model_kind_grid": ["logreg", "sklearn_gbdt", "lightgbm", "catboost", "random_forest"],
                "calibration_grid": ["sigmoid", "isotonic"],
                "selection_metric": "pr_auc",
                "use_class_weight": True,
            }

        else:
            horizons = st.multiselect(
                "Горизонты прогноза",
                [30, 60, 90, 120],
                default=[30, 60, 90],
                key="horizons_auto",
            )

            history_grid = st.multiselect(
                "Окно истории",
                [90, 180, 365],
                default=[180, 365],
                key="history_auto",
            )

            step_grid = st.multiselect(
                "Шаг anchor",
                [7, 14, 30],
                default=[30],
                key="step_auto",
            )

            model_kind_grid = st.multiselect(
                "Какие модели включить",
                ["logreg", "sklearn_gbdt", "lightgbm", "catboost", "random_forest"],
                default=["logreg", "sklearn_gbdt", "lightgbm", "catboost"],
                key="model_kind_grid_adv",
            )

            calibration_grid = st.multiselect(
                "Какие калибровки включить",
                ["sigmoid", "isotonic"],
                default=["sigmoid", "isotonic"],
                key="calibration_grid_adv",
            )

            selection_metric = st.selectbox(
                "Главная метрика выбора",
                ["pr_auc", "roc_auc", "brier", "base_max_profit"],
                index=0,
                key="selection_metric_adv",
            )

            use_class_weight = st.checkbox(
                "Использовать class weights",
                value=True,
                key="use_class_weight_adv",
            )

            params = {
                "horizon_days_grid": horizons,
                "history_days_grid": history_grid,
                "step_days_grid": step_grid,
                "model_kind_grid": model_kind_grid,
                "calibration_grid": calibration_grid,
                "selection_metric": selection_metric,
                "use_class_weight": use_class_weight,
            }

        params["model_name"] = model_name
        params["extra_feature_columns"] = st.session_state.get("extra_feature_columns", [])
        if st.button("Запустить автоматическое обучение"):
            if uploaded is None:
                st.error("Файл больше недоступен в сессии. Перезагрузите CSV на первой странице.")
            else:
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                payload = {
                    "template": st.session_state["template"],
                    "mapping_json": json.dumps(mapping_result, ensure_ascii=False),
                    "params_json": json.dumps(params, ensure_ascii=False),
                }

                resp = api_post(
                    f"{API_URL}/jobs",
                    data=payload,
                    files=files,
                    timeout=180,
                )

                if resp.status_code == 200:
                    st.session_state["job_info"] = resp.json()
                    st.success(resp.json())
                else:
                    st.error(resp.text)

        if st.session_state.get("job_info"):
            st.subheader("Текущая job")
            st.json(st.session_state["job_info"])


# =========================
# TAB 4
# =========================

with tab4:
    st.header("Результаты")

    default_job_id = st.session_state.get("job_info", {}).get("job_id", "")
    job_id = st.text_input("job_id", value=default_job_id, key="result_job_id")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Проверить статус"):
            if not job_id:
                st.warning("Введите job_id")
            else:
                resp = api_get(f"{API_URL}/jobs/{job_id}")
                if resp.status_code == 200:
                    status_data = resp.json()
                    st.session_state["job_status"] = status_data
                else:
                    st.error(resp.text)

    with c2:
        if st.button("Загрузить результат"):
            if not job_id:
                st.warning("Введите job_id")
            else:
                resp = api_get(f"{API_URL}/jobs/{job_id}/result")
                if resp.status_code == 200:
                    st.session_state["job_result"] = resp.json()
                else:
                    st.error(resp.text)

    if st.session_state.get("job_status"):
        status_data = st.session_state["job_status"]

        st.subheader("Статус обучения")

        c1, c2, c3 = st.columns(3)
        c1.metric("Status", status_data.get("status", "—"))
        c2.metric("Stage", status_data.get("stage", "—"))
        c3.metric("Progress", f"{status_data.get('progress', 0)}%")

        progress = status_data.get("progress")
        if progress is not None:
            st.progress(int(progress))

        current_exp = status_data.get("current_experiment")
        if current_exp:
            st.subheader("Текущий эксперимент")
            st.json(current_exp)

        best_so_far = status_data.get("best_so_far")
        if best_so_far:
            st.subheader("Лучший результат на текущий момент")
            st.json(best_so_far)

        last_finished = status_data.get("last_finished")
        if last_finished:
            st.subheader("Последний завершенный эксперимент")
            st.json(last_finished)

    result = st.session_state.get("job_result")

    if result:
        st.divider()
        st.subheader("Победившая конфигурация")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Template", result.get("template", "—"))
        c2.metric("Snapshots", result.get("n_snapshots", "—"))
        c3.metric("Features", result.get("feature_count", "—"))
        c4.metric("Best k", result.get("business_metrics", {}).get("base_best_k", "—"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("PR-AUC", round(result.get("test_metrics_cal", {}).get("pr_auc", 0), 4))
        c6.metric("ROC-AUC", round(result.get("test_metrics_cal", {}).get("roc_auc", 0), 4))
        c7.metric("Brier", round(result.get("test_metrics_cal", {}).get("brier", 0), 4))
        c8.metric("MaxProfit", round(result.get("business_metrics", {}).get("base_max_profit", 0), 2))

        st.subheader("Параметры победителя")
        st.json(result.get("params_used", {}))

        st.subheader("Калибровка")
        st.json(result.get("calibration", {}))

        st.subheader("Walk-forward")
        st.json(result.get("walk_forward", {}))

        st.subheader("Артефакты")
        st.json(result.get("artifacts", {}))

        download_job_id = st.text_input("job_id для скачивания ZIP", value=job_id, key="download_job_id")
        if st.button("Скачать report.zip"):
            r = api_get(f"{API_URL}/jobs/{download_job_id}/download", timeout=120)
            if r.status_code == 200:
                st.download_button(
                    "Download report.zip",
                    data=r.content,
                    file_name="report.zip",
                )
            else:
                st.error(r.text)


with tab5:
    st.header("Модели и прогноз")

    if st.button("Обновить список моделей"):
        resp = api_get(f"{API_URL}/models")
        if resp.status_code == 200:
            st.session_state["models_cache"] = resp.json()["models"]
        else:
            st.error(resp.text)

    models = st.session_state.get("models_cache")
    if models:
        models_df = pd.DataFrame(models)
        st.subheader("Реестр моделей")
        st.dataframe(models_df, use_container_width=True)

        model_options = {
            f"{m['job_id']} | {m.get('template')} | {m.get('params_used', {}).get('model_kind', '')}": m
            for m in models
        }

        selected_label = st.selectbox(
            "Выберите модель",
            list(model_options.keys()),
        )
        selected_model = model_options[selected_label]

        st.subheader("Детали модели")
        st.json(selected_model)

        st.divider()
        st.subheader("Прогноз на новых данных")

        score_file = st.file_uploader(
            "Загрузите CSV для scoring",
            type=["csv"],
            key="score_file",
        )

        if score_file is not None and st.button("Запустить прогноз"):
            files = {"file": (score_file.name, score_file.getvalue(), "text/csv")}
            resp = api_post(
                f"{API_URL}/score",
                data={"bundle_dir": selected_model["bundle_dir"]},
                files=files,
                timeout=180,
            )

            if resp.status_code == 200:
                score_result = resp.json()
                st.session_state["score_result"] = score_result
            else:
                st.error(resp.text)

    score_result = st.session_state.get("score_result")
    if score_result:
        st.subheader("Результат прогноза")
        st.json({
            "n_rows": score_result.get("n_rows"),
            "output_csv": score_result.get("output_csv"),
        })

        preview_df = pd.DataFrame(score_result.get("preview", []))
        if not preview_df.empty:
            st.dataframe(preview_df, use_container_width=True)
            risk_segment_preview(preview_df)

'''
