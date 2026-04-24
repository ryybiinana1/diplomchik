from __future__ import annotations

import streamlit as st

# Корень rem для всего приложения (px). +2 к предыдущему 18 → 20.
_UI_FONT_ROOT_PX = 20

_THEME_STYLE_ELEMENT_ID = "diplom-ui-theme"


def _build_theme_css(px: int) -> str:
    return f"""
            :root {{
                --ui-bg: #f5f7fb;
                --ui-surface: #ffffff;
                --ui-surface-muted: #f8fafc;
                --ui-border: rgba(15, 23, 42, 0.10);
                --ui-border-strong: rgba(15, 23, 42, 0.16);
                --ui-text: #0f172a;
                --ui-text-soft: rgba(15, 23, 42, 0.72);
                --ui-text-muted: rgba(15, 23, 42, 0.58);
                --ui-accent: #2563eb;
                --ui-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
                --ui-font-root: {px}px;
            }}

            /* Масштабирует весь UI (включая виджеты с px из Base Web и текст в SVG графиков) */
            .stApp {{
                zoom: 1.18 !important;
            }}

            /* Корень rem */
            html {{
                font-size: {px}px !important;
            }}

            body {{
                color: var(--ui-text);
                font-size: {px}px !important;
            }}

            .stApp {{
                font-size: {px}px !important;
                background: var(--ui-bg);
            }}

            [data-testid="stAppViewContainer"] {{
                background: var(--ui-bg);
                font-size: {px}px !important;
            }}

            /* Основная колонка и сайдбар: явный базовый кегль */
            section[data-testid="stMain"],
            section[data-testid="stMain"] .block-container {{
                font-size: 1rem !important;
            }}

            section[data-testid="stSidebar"],
            section[data-testid="stSidebar"] .block-container {{
                font-size: 0.98rem !important;
            }}

            .block-container {{
                max-width: min(92rem, calc(100vw - 2rem)) !important;
                width: 100%;
                padding-top: 1.25rem;
                padding-bottom: 2.5rem;
                padding-left: clamp(1rem, 2.4vw, 2.35rem);
                padding-right: clamp(1rem, 2.4vw, 2.35rem);
            }}

            /* Запас снизу: иначе при zoom / длинной форме кнопки и page_nav обрезаются при прокрутке */
            section[data-testid="stMain"] .block-container {{
                padding-bottom: max(13rem, 24vh) !important;
                overflow-x: clip !important;
            }}

            section[data-testid="stMain"] [data-testid="stHorizontalBlock"] > div,
            section[data-testid="stMain"] [data-testid="column"] {{
                min-width: 0 !important;
            }}

            section[data-testid="stMain"] img,
            section[data-testid="stMain"] svg,
            section[data-testid="stMain"] canvas {{
                max-width: 100% !important;
                height: auto !important;
            }}

            section[data-testid="stSidebar"] {{
                width: 20rem !important;
                min-width: 19rem !important;
                max-width: 21rem !important;
                border-right: 1px solid var(--ui-border);
                background: rgba(255, 255, 255, 0.92);
                backdrop-filter: blur(6px);
            }}

            section[data-testid="stSidebar"] .block-container {{
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            section[data-testid="stMain"] {{
                line-height: 1.6;
            }}

            /* Текст: типичные узлы Streamlit + markdown */
            section[data-testid="stMain"] p,
            section[data-testid="stMain"] li,
            section[data-testid="stMain"] td,
            section[data-testid="stMain"] th,
            section[data-testid="stMain"] [data-testid="stCaptionContainer"],
            section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
            section[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
            section[data-testid="stMain"] [data-testid="stText"] {{
                font-size: 1rem !important;
                line-height: 1.65 !important;
                color: var(--ui-text-soft);
            }}

            section[data-testid="stMain"] label[data-testid="stWidgetLabel"] p,
            section[data-testid="stMain"] label[data-testid="stWidgetLabel"] span {{
                font-size: 1rem !important;
            }}

            section[data-testid="stMain"] div[role="radiogroup"] label,
            section[data-testid="stMain"] div[role="group"] label {{
                font-size: 1rem !important;
            }}

            section[data-testid="stMain"] [data-testid="stAlert"] p {{
                font-size: 1rem !important;
            }}

            /* Base Web: у виджетов часто фиксированный px в инлайне — перебиваем */
            section[data-testid="stMain"] [data-baseweb="input"] input,
            section[data-testid="stMain"] [data-baseweb="textarea"] textarea,
            section[data-testid="stMain"] [data-baseweb="select"] > div,
            section[data-testid="stMain"] [data-baseweb="popover"],
            section[data-testid="stMain"] [data-baseweb="base-input"] input {{
                font-size: 0.98rem !important;
            }}

            /* Обычный текст в строках markdown (не трогаем span внутри заголовков — см. блок ниже) */
            section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p span,
            section[data-testid="stMain"] [data-testid="stMarkdownContainer"] li span,
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p span,
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li span {{
                font-size: 1rem !important;
            }}

            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] li,
            section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
                font-size: 1rem !important;
                line-height: 1.65 !important;
            }}

            [data-testid="stExpander"] summary,
            [data-testid="stExpander"] details summary {{
                font-size: 1.08rem !important;
                font-weight: 600 !important;
            }}

            [data-testid="stExpander"] .stMarkdown p,
            [data-testid="stExpander"] .stMarkdown li {{
                font-size: 1rem !important;
            }}

            section[data-testid="stMain"] [data-baseweb="checkbox"] span,
            section[data-testid="stMain"] [data-baseweb="checkbox"] label {{
                font-size: 1rem !important;
            }}

            h1,
            .ui-page-title {{
                margin: 0;
                font-size: clamp(2.15rem, 3.2vw, 2.95rem) !important;
                line-height: 1.12;
                font-weight: 700;
                letter-spacing: -0.02em;
                color: var(--ui-text);
            }}

            h2 {{
                font-size: 1.55rem !important;
                line-height: 1.28 !important;
                font-weight: 700 !important;
                color: var(--ui-text);
            }}

            h3 {{
                font-size: 1.35rem !important;
                line-height: 1.28 !important;
                font-weight: 700 !important;
                color: var(--ui-text);
            }}

            h4 {{
                font-size: 1.18rem !important;
                line-height: 1.35 !important;
                font-weight: 700 !important;
                color: var(--ui-text);
            }}

            /* Streamlit разбивает заголовки на span — наследуем кегль от h1–h6 */
            .stMarkdown h1 span,
            .stMarkdown h2 span,
            .stMarkdown h3 span,
            .stMarkdown h4 span,
            .stMarkdown h5 span,
            .stMarkdown h6 span,
            h1.ui-page-title span,
            .ui-page-title span {{
                font-size: inherit !important;
                font-weight: inherit !important;
            }}

            .ui-page-header {{
                margin-bottom: 1rem;
            }}

            .ui-eyebrow {{
                margin-bottom: 0.35rem;
                font-size: 0.82rem !important;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--ui-accent);
            }}

            .ui-page-subtitle {{
                margin: 0.5rem 0 0 0;
                max-width: 58rem;
                font-size: 1.02rem !important;
                font-weight: 600 !important;
                line-height: 1.6;
                color: var(--ui-text-soft);
            }}

            .ui-section-title {{
                margin: 0 0 0.35rem 0;
                font-size: 1.18rem !important;
                font-weight: 600;
                line-height: 1.3;
                color: var(--ui-text);
            }}

            .ui-section-note {{
                margin: 0 0 1rem 0;
                font-size: 0.96rem !important;
                font-weight: 600 !important;
                line-height: 1.6;
                color: var(--ui-text-muted);
            }}

            /* Заголовки блоков внутри формы сопоставления (шаг 1): на +2px к базовому h4 */
            .ui-form-block-title {{
                font-size: calc(1.18rem + 2px) !important;
                font-weight: 600 !important;
                color: var(--ui-text);
                margin: 0.65rem 0 0.5rem 0 !important;
                line-height: 1.35 !important;
            }}

            .ui-field-intro {{
                padding: 0.85rem 0.95rem;
                margin-bottom: 0.5rem;
                border: 1px solid var(--ui-border);
                border-radius: 0.9rem;
                background: var(--ui-surface-muted);
            }}

            .ui-field-title {{
                margin: 0 0 0.2rem 0;
                font-size: 0.98rem !important;
                font-weight: 600;
                color: var(--ui-text);
            }}

            .ui-field-note {{
                margin: 0;
                font-size: 0.92rem !important;
                line-height: 1.55;
                color: var(--ui-text-muted);
            }}

            /* Сопоставление колонок: аккуратные карточки (обводка + воздух), без налезания текста */
            .ui-field-intro--mapping {{
                padding: 0.62rem 0.8rem !important;
                margin-bottom: 0.5rem !important;
                border: 1px solid var(--ui-border) !important;
                border-radius: 0.9rem !important;
                background: var(--ui-surface-muted) !important;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05) !important;
            }}

            .ui-field-intro--mapping .ui-field-title {{
                font-size: calc(0.98rem + 2px) !important;
                margin: 0 0 0.3rem 0 !important;
                line-height: 1.38 !important;
            }}

            .ui-field-intro--mapping .ui-field-note {{
                line-height: 1.52 !important;
                margin: 0 !important;
            }}

            /* Лёгкий зазор между строками формы — без сжатия виджетов до наложения */
            div[data-testid="stForm"] div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] {{
                padding-top: 0.12rem !important;
                padding-bottom: 0.12rem !important;
            }}

            [data-testid="stVerticalBlockBorderWrapper"] {{
                border: 1px solid var(--ui-border) !important;
                border-radius: 1rem !important;
                background: var(--ui-surface) !important;
                box-shadow: var(--ui-shadow);
            }}

            [data-testid="stVerticalBlockBorderWrapper"] > div {{
                padding: 0.35rem 0.45rem 0.2rem 0.45rem;
            }}

            [data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stForm"]) > div {{
                padding: 0.55rem 0.65rem 0.45rem 0.65rem !important;
            }}

            div[data-testid="stForm"] {{
                border: none !important;
                padding: 0 !important;
            }}

            .stAlert {{
                border-radius: 0.95rem;
            }}

            div[data-baseweb="select"] > div,
            div[data-testid="stFileUploaderDropzone"],
            div[data-testid="stTextInputRootElement"] > div,
            div[data-testid="stNumberInput"] > div,
            div[data-baseweb="base-input"] > div,
            textarea {{
                border-radius: 0.85rem !important;
            }}

            div[data-testid="stFileUploaderDropzone"] {{
                padding: 1rem 1rem !important;
                background: var(--ui-surface-muted);
                border: 1px dashed var(--ui-border-strong) !important;
            }}

            div[data-testid="stFileUploaderDropzone"] small,
            div[data-testid="stFileUploaderDropzoneInstructions"] small {{
                font-size: 0.92rem !important;
            }}

            div[data-testid="stFileUploaderDropzone"] button,
            .stButton > button,
            .stDownloadButton > button {{
                min-height: 2.8rem;
                border-radius: 0.85rem;
                font-size: 0.98rem !important;
                font-weight: 600;
                border: 1px solid var(--ui-border-strong);
            }}

            .stButton > button[kind="primary"] {{
                border-color: transparent;
            }}

            div[data-testid="stMetric"] {{
                padding: 0.9rem 1rem;
                border: 1px solid var(--ui-border);
                border-radius: 0.95rem;
                background: var(--ui-surface-muted);
            }}

            div[data-testid="stMetricLabel"] p {{
                font-size: 0.9rem !important;
                color: var(--ui-text-muted) !important;
            }}

            div[data-testid="stMetricValue"] {{
                font-size: 1.3rem !important;
                color: var(--ui-text) !important;
            }}

            div[data-testid="stDataFrame"] {{
                border: 1px solid var(--ui-border);
                border-radius: 0.95rem;
                overflow: auto !important;
                max-width: 100% !important;
                background: var(--ui-surface);
                font-size: 0.88rem !important;
            }}

            div[data-testid="stDataFrame"] [role="columnheader"],
            div[data-testid="stDataFrame"] [data-testid="column-header"],
            div[data-testid="stTable"] th,
            section[data-testid="stMain"] table th {{
                font-weight: 700 !important;
                color: var(--ui-text) !important;
            }}

            section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
            section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h4,
            section[data-testid="stMain"] [data-testid="stCaptionContainer"] p strong {{
                font-weight: 700 !important;
            }}

            button[kind="header"],
            button[kind="secondary"] {{
                box-shadow: none !important;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.35rem;
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: 0.8rem 0.8rem 0 0;
                padding-left: 0.9rem;
                padding-right: 0.9rem;
                font-size: 0.95rem !important;
                font-weight: 700 !important;
            }}

            section[data-testid="stSidebar"] .stMarkdown h2 {{
                font-size: 1.45rem !important;
                line-height: 1.25 !important;
                margin-bottom: 0.35rem !important;
                font-weight: 700 !important;
            }}

            section[data-testid="stSidebar"] .stMarkdown h2 span {{
                font-size: inherit !important;
            }}

            section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
                font-size: 0.92rem !important;
                color: var(--ui-text-muted) !important;
            }}

            section[data-testid="stSidebar"] div[role="radiogroup"] {{
                gap: 0.25rem;
            }}

            section[data-testid="stSidebar"] div[role="radiogroup"] label {{
                border: 1px solid transparent !important;
                border-radius: 0.85rem !important;
                padding: 0.45rem 0.5rem !important;
                margin: 0 !important;
                background: transparent !important;
            }}

            section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
                font-size: 0.96rem !important;
                font-weight: 500 !important;
                color: var(--ui-text-soft) !important;
            }}

            section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
            section[data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] {{
                background: rgba(37, 99, 235, 0.08) !important;
                border-color: rgba(37, 99, 235, 0.18) !important;
            }}

            section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
            section[data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] p {{
                color: var(--ui-text) !important;
                font-weight: 600 !important;
            }}

            pre,
            code {{
                font-size: 0.92rem !important;
                line-height: 1.5 !important;
            }}

            @media (max-width: 1200px) {{
                html {{
                    font-size: {px - 1}px !important;
                }}

                section[data-testid="stSidebar"] {{
                    width: 18rem !important;
                    min-width: 18rem !important;
                }}
            }}
"""


def inject_global_theme() -> None:
    px = _UI_FONT_ROOT_PX
    css = _build_theme_css(px)
    st.markdown(f'<style id="{_THEME_STYLE_ELEMENT_ID}">{css}</style>', unsafe_allow_html=True)
