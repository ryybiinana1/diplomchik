from __future__ import annotations

from typing import Any

import streamlit as st


def verdict_card(verdict: str, reasons: list[str], metrics: dict[str, Any] | None = None) -> None:
    verdict = (verdict or "").lower()

    if verdict in ("ready", "готово"):
        st.success("Датасет выглядит пригодным для обучения")
    elif verdict in ("partial", "частично"):
        st.warning("Датасет частично готов: можно обучать, но есть риски")
    else:
        st.error("Датасет пока не рекомендуется использовать без доработки")

    if reasons:
        for r in reasons:
            st.write(f"• {r}")

    if metrics:
        with st.expander("Показатели диагностики"):
            st.json(metrics)


def column_risk_badge(verdict: str) -> str:
    mapping = {
        "allow": "🟢 можно",
        "review": "🟡 осторожно",
        "block": "🔴 запрет",
    }
    return mapping.get(verdict, "⚪ неизвестно")


def metric_card_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (title, value) in zip(cols, items):
        col.metric(title, value)