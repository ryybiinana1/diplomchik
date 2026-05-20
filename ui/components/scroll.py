from __future__ import annotations

import streamlit.components.v1 as components


def scroll_to_top() -> None:
    """Прокрутка основной области Streamlit в начало после смены шага."""
    components.html(
        """
        <script>
        (function () {
          function jump() {
            const w = window.parent;
            if (!w || !w.document) return;
            const doc = w.document;
            w.scrollTo(0, 0);
            doc.documentElement.scrollTop = 0;
            doc.body.scrollTop = 0;
            const selectors = [
              '[data-testid="stAppViewContainer"]',
              '[data-testid="stVerticalBlock"]',
              'section.main',
              '.main',
            ];
            selectors.forEach(function (selector) {
              doc.querySelectorAll(selector).forEach(function (el) {
                try { el.scrollTop = 0; el.scrollTo(0, 0); } catch (e) {}
              });
            });
          }
          jump();
          setTimeout(jump, 50);
          setTimeout(jump, 200);
        })();
        </script>
        """,
        height=0,
        width=0,
    )
