# ui/components/status_cards.py
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def verdict_card(verdict: str, reasons: List[str], metrics: Dict[str, Any] | None = None) -> None:
    verdict = (verdict or "").lower()

    if verdict in ("ready", "готово"):
        st.success("Готово к обучению")
    elif verdict in ("partial", "частично"):
        st.warning("Частично готово")
    else:
        st.error("Не рекомендуется обучать")

    if reasons:
        st.write("Причины:")
        for r in reasons:
            st.write(f"- {r}")

    if metrics:
        with st.expander("Показатели диагностики"):
            st.json(metrics)


def column_risk_badge(verdict: str) -> str:
    if verdict == "allow":
        return "🟢 можно"
    if verdict == "review":
        return "🟡 осторожно"
    return "🔴 запрет"
