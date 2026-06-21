# -*- coding: utf-8 -*-
"""
visualization.py
================
Класс Visualizer — интерактивные графики (Plotly) для интерфейса Streamlit.
Каждый метод возвращает готовую фигуру, которую app.py показывает через
st.plotly_chart().
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class Visualizer:
    """Строит графики на основе данных из DataProcessor."""

    WEEKDAYS_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

    # Приблизительные координаты центров областей — для карты активности
    OBLAST_COORDS = {
        "Vinnytska oblast": (49.23, 28.47), "Volynska oblast": (50.75, 25.33),
        "Dnipropetrovska oblast": (48.46, 35.04), "Donetska oblast": (48.00, 37.80),
        "Zhytomyrska oblast": (50.25, 28.66), "Zakarpatska oblast": (48.62, 22.29),
        "Zaporizka oblast": (47.84, 35.14), "Ivano-Frankivska oblast": (48.92, 24.71),
        "Kyivska oblast": (50.10, 30.30), "Kyiv City": (50.45, 30.52),
        "Kirovohradska oblast": (48.51, 32.26), "Luhanska oblast": (48.57, 39.31),
        "Lvivska oblast": (49.84, 24.03), "Mykolaivska oblast": (46.97, 31.99),
        "Odeska oblast": (46.48, 30.72), "Poltavska oblast": (49.59, 34.55),
        "Rivnenska oblast": (50.62, 26.25), "Sumska oblast": (50.91, 34.80),
        "Ternopilska oblast": (49.55, 25.59), "Kharkivska oblast": (49.99, 36.23),
        "Khersonska oblast": (46.64, 32.61), "Khmelnytska oblast": (49.42, 26.99),
        "Cherkaska oblast": (49.44, 32.06), "Chernivetska oblast": (48.29, 25.94),
        "Chernihivska oblast": (51.49, 31.29),
    }

    _LAYOUT = dict(
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )

    def __init__(self, processor):
        """processor — экземпляр DataProcessor (даёт доступ к df и grid)."""
        self.proc = processor

    @property
    def grid(self) -> pd.DataFrame:
        return self.proc.grid

    # =================================================================
    # Графики
    # =================================================================
    def daily_counts(self, oblast: str) -> go.Figure:
        """Количество тревог по дням + скользящее среднее (7 дней)."""
        s = self.proc.daily_counts(oblast)
        ma = s.rolling(7).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                 name="Тревог в день",
                                 line=dict(color="#93c5fd", width=1)))
        fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode="lines",
                                 name="Среднее (7 дней)",
                                 line=dict(color="#1d4ed8", width=2.5)))
        fig.update_layout(title=f"Количество тревог по дням — {oblast}",
                          xaxis_title="Дата", yaxis_title="Тревог в день", **self._LAYOUT)
        return fig

    def hour_probability(self, probs, selected_hour: int) -> go.Figure:
        """Вероятность тревоги по часам суток для выбранного дня."""
        colors = ["#f59e0b" if h == selected_hour else "#cbd5e1" for h in range(24)]
        fig = go.Figure(go.Bar(
            x=list(range(24)), y=(probs * 100).round(1), marker_color=colors,
            hovertemplate="%{x}:00 — %{y:.0f}%<extra></extra>"))
        fig.update_layout(title="Вероятность тревоги по часам суток",
                          xaxis_title="Час суток", yaxis_title="Вероятность, %",
                          yaxis_range=[0, 100], **self._LAYOUT)
        fig.update_xaxes(dtick=2)
        return fig

    def heatmap(self, oblast: str) -> go.Figure:
        """Тепловая карта: день недели × час (историческая доля тревожных часов)."""
        sub = self.grid[self.grid["oblast"] == oblast]
        pivot = (sub.pivot_table(index="day_of_week", columns="hour",
                                 values="alert", aggfunc="mean")
                 .reindex(index=range(7), columns=range(24)))
        fig = px.imshow(pivot.values * 100, color_continuous_scale="Inferno",
                        labels=dict(x="Час суток", y="День недели", color="% тревог"),
                        x=list(range(24)), y=self.WEEKDAYS_RU, aspect="auto")
        fig.update_layout(title=f"Активность тревог: день недели × час — {oblast}",
                          **self._LAYOUT)
        fig.update_xaxes(dtick=2)
        return fig

    def oblast_bar(self, stats: pd.DataFrame, highlight: str, top_n: int = 15) -> go.Figure:
        """Сравнение областей по количеству тревог (топ-N)."""
        top = stats.head(top_n).iloc[::-1]
        colors = ["#ea580c" if name == highlight else "#3b82f6" for name in top.index]
        fig = go.Figure(go.Bar(
            x=top["alerts"], y=top.index, orientation="h", marker_color=colors,
            hovertemplate="%{y}: %{x:,} тревог<extra></extra>"))
        fig.update_layout(title=f"Сравнение областей по количеству тревог (топ-{top_n})",
                          xaxis_title="Тревог за весь период", **self._LAYOUT)
        return fig

    def top_busiest(self, top: pd.DataFrame) -> go.Figure:
        """График ТОП-N самых напряжённых дней (часов под тревогой)."""
        data = top.iloc[::-1]  # самый тяжёлый — наверху
        hours = (data["total_min"] / 60).round(1)
        fig = go.Figure(go.Bar(
            x=hours, y=data["label"], orientation="h",
            marker=dict(color=data["count"], colorscale="Inferno",
                        colorbar=dict(title="Тревог")),
            customdata=data["count"],
            hovertemplate="%{y}<br>Под тревогой: %{x} ч<br>"
                          "Тревог за день: %{customdata}<extra></extra>"))
        fig.update_layout(title="ТОП самых напряжённых дней (часов под тревогой)",
                          xaxis_title="Время под тревогой, часов", **self._LAYOUT)
        return fig

    def oblast_map(self, stats: pd.DataFrame, highlight: str) -> go.Figure:
        """Карта активности областей: размер/цвет точки = число тревог."""
        rows = []
        for name, row in stats.iterrows():
            if name in self.OBLAST_COORDS:
                lat, lon = self.OBLAST_COORDS[name]
                rows.append({"oblast": name, "lat": lat, "lon": lon,
                             "alerts": int(row["alerts"]),
                             "rate": round(float(row["hourly_rate"]) * 100, 1)})
        mp = pd.DataFrame(rows)
        try:  # в Plotly 6 актуальна scatter_map (MapLibre)
            fig = px.scatter_map(
                mp, lat="lat", lon="lon", size="alerts", color="alerts",
                color_continuous_scale="Inferno", size_max=40, zoom=4.2,
                hover_name="oblast",
                hover_data={"alerts": True, "rate": True, "lat": False, "lon": False},
                map_style="open-street-map")
        except AttributeError:  # запасной путь для старых версий
            fig = px.scatter_mapbox(
                mp, lat="lat", lon="lon", size="alerts", color="alerts",
                color_continuous_scale="Inferno", size_max=40, zoom=4.2,
                hover_name="oblast", mapbox_style="open-street-map")
        fig.update_layout(title="Карта активности тревог по областям",
                          margin=dict(l=0, r=0, t=50, b=0),
                          paper_bgcolor="rgba(0,0,0,0)")
        return fig
