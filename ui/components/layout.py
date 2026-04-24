from __future__ import annotations

from contextlib import contextmanager
from html import escape
from typing import Iterator

import streamlit as st


def render_page_header(title: str, subtitle: str, eyebrow: str | None = None) -> None:
    parts: list[str] = ['<div class="ui-page-header">']
    if eyebrow:
        parts.append(f'<div class="ui-eyebrow">{escape(eyebrow)}</div>')
    parts.append(f'<h1 class="ui-page-title">{escape(title)}</h1>')
    parts.append(f'<p class="ui-page-subtitle">{escape(subtitle)}</p>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_section_intro(title: str, note: str | None = None) -> None:
    st.markdown(f'<div class="ui-section-title">{escape(title)}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<p class="ui-section-note">{escape(note)}</p>', unsafe_allow_html=True)


def render_field_intro(title: str, note: str | None = None, *, mapping_compact: bool = False) -> None:
    """mapping_compact: плотнее блоки в форме сопоставления колонок (шаг 1), без уменьшения шрифта подписей."""
    intro_cls = "ui-field-intro ui-field-intro--mapping" if mapping_compact else "ui-field-intro"
    st.markdown(
        (
            f'<div class="{intro_cls}">'
            f'<div class="ui-field-title">{escape(title)}</div>'
            f'<div class="ui-field-note">{escape(note or "")}</div>'
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


@contextmanager
def section_card(title: str, note: str | None = None) -> Iterator[None]:
    with st.container(border=True):
        render_section_intro(title, note)
        yield
