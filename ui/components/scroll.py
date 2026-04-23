from __future__ import annotations

import streamlit.components.v1 as components


def scroll_to_top() -> None:
    """Прокрутка основной области Streamlit в начало после смены шага."""
    components.html(
        """
        <script>
        (function () {
          const w = window.parent;
          if (!w || !w.document) return;
          const doc = w.document;
          const sel = [
            '[data-testid="stAppViewContainer"]',
            "section.main",
            ".main",
          ];
          for (const s of sel) {
            const el = doc.querySelector(s);
            if (el) {
              el.scrollTo({ top: 0, behavior: "instant" });
              break;
            }
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )
