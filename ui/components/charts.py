from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - fallback for lightweight containers
    plt = None


def _format_compact_number(value: float | int) -> str:
    try:
        return f"{float(value):,.0f}".replace(",", " ")
    except Exception:
        return str(value)


def _trim_numeric_series(series: pd.Series, upper_q: float = 0.99) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return s
    upper = float(s.quantile(upper_q))
    return s.clip(upper=upper)


def _figure(figsize: tuple[float, float] = (8.8, 4.8)):
    if plt is None:
        return None, None
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _finalize_plot(fig, ax, *, title: str, x_title: str, y_title: str, rotate_x: int = 0) -> None:
    if plt is None or fig is None or ax is None:
        return
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel(x_title, fontsize=13, labelpad=10)
    ax.set_ylabel(y_title, fontsize=13, labelpad=10)
    ax.tick_params(axis="both", labelsize=11)
    if rotate_x:
        for label in ax.get_xticklabels():
            label.set_rotation(rotate_x)
            label.set_horizontalalignment("right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_bar_chart(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    x_title: str,
    y_title: str,
    horizontal: bool = False,
    color: str = "#4C78A8",
    height: int = 320,
) -> None:
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        st.info("Недостаточно данных для построения графика.")
        return
    if plt is None:
        st.markdown(f"#### {title}")
        st.caption(f"Ось X: {y_title if horizontal else x_title} · Ось Y: {x_title if horizontal else y_title}")
        series = pd.to_numeric(data[y_col], errors="coerce").fillna(0)
        if horizontal:
            fallback = pd.Series(series.values, index=data[x_col].astype(str))
            st.bar_chart(fallback, width="stretch")
        else:
            fallback = pd.Series(series.values, index=data[x_col].astype(str))
            st.bar_chart(fallback, width="stretch")
        return
    fig, ax = _figure((8.6, max(4.2, min(height / 72.0, 6.6))))
    if horizontal:
        ax.barh(data[x_col].astype(str), pd.to_numeric(data[y_col], errors="coerce").fillna(0), color=color)
        _finalize_plot(fig, ax, title=title, x_title=y_title, y_title=x_title)
    else:
        ax.bar(data[x_col].astype(str), pd.to_numeric(data[y_col], errors="coerce").fillna(0), color=color)
        rotate_x = 30 if len(data) > 6 else 0
        _finalize_plot(fig, ax, title=title, x_title=x_title, y_title=y_title, rotate_x=rotate_x)


def render_line_chart(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    x_title: str,
    y_title: str,
    color: str = "#4C78A8",
    height: int = 320,
) -> None:
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        st.info("Недостаточно данных для построения графика.")
        return
    if plt is None:
        st.markdown(f"#### {title}")
        st.caption(f"Ось X: {x_title} · Ось Y: {y_title}")
        fallback = pd.DataFrame({y_col: pd.to_numeric(data[y_col], errors="coerce").fillna(0).values}, index=data[x_col])
        st.line_chart(fallback, width="stretch")
        return
    fig, ax = _figure((8.4, max(4.2, min(height / 72.0, 6.2))))
    x = data[x_col]
    y = pd.to_numeric(data[y_col], errors="coerce").fillna(0)
    ax.plot(x, y, marker="o", color=color, linewidth=2)
    _finalize_plot(fig, ax, title=title, x_title=x_title, y_title=y_title, rotate_x=30 if not pd.api.types.is_datetime64_any_dtype(x) else 0)


def render_histogram(
    labels: list[str],
    values: list[int | float],
    *,
    title: str,
    x_title: str,
    y_title: str,
    height: int = 320,
) -> None:
    data = pd.DataFrame({"bin": labels, "value": values})
    render_bar_chart(
        data,
        x_col="bin",
        y_col="value",
        title=title,
        x_title=x_title,
        y_title=y_title,
        height=height,
    )


def category_distribution_table(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    vc = df[col].fillna("NA").astype(str).value_counts(dropna=False)
    total = int(vc.sum())
    top = vc.head(top_n).copy()
    other = int(vc.iloc[top_n:].sum()) if len(vc) > top_n else 0
    rows = []
    for value, count in top.items():
        rows.append({"Значение": str(value), "Строк": int(count), "Доля": float(count / max(total, 1))})
    if other:
        rows.append({"Значение": "Прочие", "Строк": other, "Доля": float(other / max(total, 1))})
    return pd.DataFrame(rows)


def monthly_counts(
    df: pd.DataFrame,
    time_col: str,
    value_col: str | None = None,
    title: str = "",
) -> None:
    if time_col not in df.columns:
        st.info(f"Колонка `{time_col}` не найдена.")
        return

    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])

    if d.empty:
        st.info("Недостаточно корректных дат для построения графика.")
        return

    d["month"] = d[time_col].dt.to_period("M").dt.to_timestamp()

    if title:
        chart_title = title
    else:
        chart_title = "Динамика по месяцам"

    if value_col is None:
        agg = d.groupby("month").size().rename("count").reset_index()
        render_line_chart(
            agg,
            x_col="month",
            y_col="count",
            title=chart_title,
            x_title="Месяц",
            y_title="Число записей",
        )
    else:
        if value_col not in d.columns:
            st.info(f"Колонка `{value_col}` не найдена.")
            return
        d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
        agg = d.groupby("month")[value_col].sum(min_count=1).rename("sum").reset_index()
        render_line_chart(
            agg,
            x_col="month",
            y_col="sum",
            title=chart_title,
            x_title="Месяц",
            y_title="Суммарное значение",
        )


def monthly_unique_entities(
    df: pd.DataFrame,
    *,
    time_col: str,
    entity_col: str,
    title: str = "",
) -> None:
    if time_col not in df.columns or entity_col not in df.columns:
        st.info("Недостаточно данных для графика уникальных объектов по месяцам.")
        return

    d = df[[time_col, entity_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col, entity_col])
    if d.empty:
        st.info("Недостаточно корректных дат или идентификаторов для графика.")
        return

    d["month"] = d[time_col].dt.to_period("M").dt.to_timestamp()
    agg = d.groupby("month")[entity_col].nunique().rename("nunique_entities").reset_index()
    render_line_chart(
        agg,
        x_col="month",
        y_col="nunique_entities",
        title=title or f"Уникальные объекты по месяцам ({entity_col})",
        x_title="Месяц",
        y_title="Число уникальных объектов",
    )


def monthly_average_value(
    df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    title: str = "",
) -> None:
    if time_col not in df.columns or value_col not in df.columns:
        st.info("Недостаточно данных для графика среднего значения по месяцам.")
        return

    d = df[[time_col, value_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[time_col, value_col])
    if d.empty:
        st.info("Недостаточно корректных значений для графика.")
        return

    d["month"] = d[time_col].dt.to_period("M").dt.to_timestamp()
    agg = d.groupby("month")[value_col].mean().rename("avg_value").reset_index()
    render_line_chart(
        agg,
        x_col="month",
        y_col="avg_value",
        title=title or f"Среднее значение по месяцам ({value_col})",
        x_title="Месяц",
        y_title="Среднее значение",
    )


def top_entities_by_value(
    df: pd.DataFrame,
    *,
    entity_col: str,
    value_col: str,
    title: str = "",
    top_n: int = 10,
) -> None:
    if entity_col not in df.columns or value_col not in df.columns:
        st.info("Недостаточно данных для графика по ключевым объектам.")
        return

    d = df[[entity_col, value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[entity_col, value_col])
    if d.empty:
        st.info("Нет числовых данных для агрегации по объектам.")
        return

    agg = (
        d.groupby(entity_col)[value_col]
        .sum(min_count=1)
        .sort_values(ascending=False)
        .head(top_n)
        .rename("total_value")
        .reset_index()
    )
    render_bar_chart(
        agg,
        x_col=entity_col,
        y_col="total_value",
        title=title or f"Топ-{top_n} объектов по суммарному значению",
        x_title="Объект",
        y_title="Суммарное значение",
        horizontal=True,
        height=420,
    )


def hist_per_entity(
    df: pd.DataFrame,
    entity_col: str,
    title: str = "",
    max_bins: int = 50,
) -> None:
    if entity_col not in df.columns:
        st.info(f"Колонка `{entity_col}` не найдена.")
        return

    d = df[[entity_col]].dropna().copy()
    if d.empty:
        st.info("Недостаточно данных для графика.")
        return

    per = d.groupby(entity_col).size()
    hist = per.value_counts().sort_index().head(max_bins)
    render_bar_chart(
        hist.rename_axis("records_per_entity").reset_index(name="entities"),
        x_col="records_per_entity",
        y_col="entities",
        title=title or f"Сколько строк истории приходится на один объект ({entity_col})",
        x_title="Строк истории на один объект",
        y_title="Количество объектов",
    )


def plot_target_distribution(df: pd.DataFrame, target_col: str, *, title: str | None = None) -> None:
    if target_col not in df.columns:
        return

    vc = df[target_col].astype(str).fillna("NaN").value_counts(dropna=False)
    if title:
        chart_title = title
    else:
        chart_title = f"Баланс классов в `{target_col}`"
    render_bar_chart(
        vc.rename_axis("class").reset_index(name="count"),
        x_col="class",
        y_col="count",
        title=chart_title,
        x_title="Значение класса",
        y_title="Количество snapshot-объектов",
    )


def plot_missing_by_column(df: pd.DataFrame, top_n: int = 15) -> None:
    miss = (df.isna().mean() * 100).sort_values(ascending=False).head(top_n)
    miss = miss[miss > 0]
    if miss.empty:
        st.markdown("#### Доля пропусков по колонкам")
        st.info("Пропусков в данных не обнаружено (или все колонки полностью заполнены).")
        return
    render_bar_chart(
        miss.rename_axis("column").reset_index(name="missing_pct"),
        x_col="column",
        y_col="missing_pct",
        title="Топ колонок по доле пропусков",
        x_title="Колонка",
        y_title="Пропуски, %",
        horizontal=True,
        color="#E45756",
        height=360,
    )


def plot_top_categories(df: pd.DataFrame, col: str, top_n: int = 10) -> None:
    if col not in df.columns:
        return
    table = category_distribution_table(df, col, top_n=top_n)
    if table.empty:
        return
    render_bar_chart(
        table,
        x_col="Значение",
        y_col="Строк",
        title=f"Топ значений в `{col}`",
        x_title="Категория",
        y_title="Число строк",
        horizontal=True,
        height=340,
    )
    st.dataframe(
        table.assign(Доля=table["Доля"].map(lambda x: f"{x:.1%}")),
        width="stretch",
        hide_index=True,
    )


def plot_weekly_counts(df: pd.DataFrame, time_col: str, title: str = "") -> None:
    """Активность по неделям — удобно при коротком горизонте данных."""
    if time_col not in df.columns:
        st.info(f"Колонка `{time_col}` не найдена.")
        return
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])
    if d.empty:
        st.info("Недостаточно корректных дат для недельного графика.")
        return
    d["week"] = d[time_col].dt.to_period("W").dt.start_time
    agg = d.groupby("week").size().rename("count").reset_index()
    render_line_chart(
        agg,
        x_col="week",
        y_col="count",
        title=title or "Активность по неделям",
        x_title="Неделя",
        y_title="Число записей",
    )


def plot_daily_counts(df: pd.DataFrame, time_col: str, title: str = "") -> None:
    """Активность по дням."""
    if time_col not in df.columns:
        st.info(f"Колонка `{time_col}` не найдена.")
        return
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])
    if d.empty:
        st.info("Недостаточно корректных дат для дневного графика.")
        return
    d["day"] = d[time_col].dt.to_period("D").dt.to_timestamp()
    agg = d.groupby("day").size().rename("count").reset_index()
    render_line_chart(
        agg,
        x_col="day",
        y_col="count",
        title=title or "Активность по дням",
        x_title="День",
        y_title="Число записей",
    )


def plot_numeric_bins(df: pd.DataFrame, value_col: str, title: str = "", bins: int = 30) -> None:
    """Гистограмма суммы/платежа (числовая колонка из сопоставления)."""
    if value_col not in df.columns:
        return
    s = _trim_numeric_series(df[value_col], upper_q=0.99).to_numpy(dtype=float)
    if s.size == 0:
        st.info(f"В колонке «{value_col}» нет числовых значений для графика.")
        return
    n_bins = min(bins, max(5, int(np.sqrt(float(s.size)))))
    hist, edges = np.histogram(s, bins=n_bins)
    labels = [f"{_format_compact_number(edges[i])}–{_format_compact_number(edges[i + 1])}" for i in range(len(hist))]
    render_histogram(
        labels,
        hist.tolist(),
        title=title,
        x_title="Интервал значений",
        y_title="Число строк файла",
    )
    st.caption(
        "График построен по строкам файла, а не по клиентам. Верхний 1% экстремальных значений обрезан только для удобства чтения."
    )


def plot_category_profile(df: pd.DataFrame, col: str, *, top_n: int = 10, title: str | None = None) -> None:
    if col not in df.columns:
        st.info(f"Колонка `{col}` не найдена.")
        return
    table = category_distribution_table(df, col, top_n=top_n)
    if table.empty:
        st.info("Нет значений для построения распределения.")
        return
    render_bar_chart(
        table,
        x_col="Значение",
        y_col="Строк",
        title=title or f"Распределение значений в `{col}`",
        x_title="Категория",
        y_title="Число строк файла",
        horizontal=True,
        height=360,
    )
    st.caption("Показаны самые частые значения по строкам файла; редкие объединены в группу «Прочие».")
    st.dataframe(
        table.assign(
            Строк=table["Строк"].map(_format_compact_number),
            Доля=table["Доля"].map(lambda x: f"{x:.1%}"),
        ),
        width="stretch",
        hide_index=True,
    )


def plot_experiment_scores(df: pd.DataFrame, label_col: str, score_col: str, title: str) -> None:
    if df.empty or label_col not in df.columns or score_col not in df.columns:
        st.info("Недостаточно данных для сравнения вариантов.")
        return
    st.markdown(f"#### {title}")
    data = df[[label_col, score_col]].dropna().copy()
    if data.empty:
        st.info("Для построения графика нет успешных вариантов.")
        return
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data = data.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    render_bar_chart(
        data,
        x_col=label_col,
        y_col=score_col,
        title=title,
        x_title="Вариант",
        y_title="Значение метрики",
        horizontal=True,
        height=360,
    )