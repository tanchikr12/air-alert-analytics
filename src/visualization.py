# -*- coding: utf-8 -*-
"""
visualization.py
================
Visualizer class — interactive charts (Plotly) for the Streamlit UI.
Each method returns a ready figure that app.py renders via st.plotly_chart().
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class Visualizer:
    """Builds charts from the data provided by DataProcessor."""

    WEEKDAYS_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Approximate oblast-center coordinates — for the activity map
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
        """processor — a DataProcessor instance (gives access to df and grid)."""
        self.proc = processor

    @property
    def grid(self) -> pd.DataFrame:
        return self.proc.grid

    # =================================================================
    # Charts
    # =================================================================
    def daily_counts(self, oblast: str) -> go.Figure:
        """Alerts per day + 7-day moving average."""
        s = self.proc.daily_counts(oblast)
        ma = s.rolling(7).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                 name="Alerts per day",
                                 line=dict(color="#93c5fd", width=1)))
        fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode="lines",
                                 name="7-day average",
                                 line=dict(color="#1d4ed8", width=2.5)))
        fig.update_layout(title=f"Alerts per day — {oblast}",
                          xaxis_title="Date", yaxis_title="Alerts per day", **self._LAYOUT)
        return fig

    def hour_probability(self, probs, selected_hour: int) -> go.Figure:
        """Alert probability by hour of day for the selected day."""
        colors = ["#f59e0b" if h == selected_hour else "#cbd5e1" for h in range(24)]
        fig = go.Figure(go.Bar(
            x=list(range(24)), y=(probs * 100).round(1), marker_color=colors,
            hovertemplate="%{x}:00 — %{y:.0f}%<extra></extra>"))
        fig.update_layout(title="Alert probability by hour of day",
                          xaxis_title="Hour of day", yaxis_title="Probability, %",
                          yaxis_range=[0, 100], **self._LAYOUT)
        fig.update_xaxes(dtick=2)
        return fig

    def heatmap(self, oblast: str) -> go.Figure:
        """Heatmap: weekday × hour (historical share of alert hours)."""
        sub = self.grid[self.grid["oblast"] == oblast]
        pivot = (sub.pivot_table(index="day_of_week", columns="hour",
                                 values="alert", aggfunc="mean")
                 .reindex(index=range(7), columns=range(24)))
        fig = px.imshow(pivot.values * 100, color_continuous_scale="Inferno",
                        labels=dict(x="Hour of day", y="Weekday", color="% alerts"),
                        x=list(range(24)), y=self.WEEKDAYS_EN, aspect="auto")
        fig.update_layout(title=f"Alert activity: weekday × hour — {oblast}",
                          **self._LAYOUT)
        fig.update_xaxes(dtick=2)
        return fig

    def oblast_bar(self, stats: pd.DataFrame, highlight: str, top_n: int = 15) -> go.Figure:
        """Comparison of oblasts by number of alerts (top-N)."""
        top = stats.head(top_n).iloc[::-1]
        colors = ["#ea580c" if name == highlight else "#3b82f6" for name in top.index]
        fig = go.Figure(go.Bar(
            x=top["alerts"], y=top.index, orientation="h", marker_color=colors,
            hovertemplate="%{y}: %{x:,} alerts<extra></extra>"))
        fig.update_layout(title=f"Oblasts by number of alerts (top-{top_n})",
                          xaxis_title="Alerts over the whole period", **self._LAYOUT)
        return fig

    def top_busiest(self, top: pd.DataFrame) -> go.Figure:
        """Chart of the top-N busiest days (hours under alert)."""
        data = top.iloc[::-1]  # the busiest one on top
        hours = (data["total_min"] / 60).round(1)
        fig = go.Figure(go.Bar(
            x=hours, y=data["label"], orientation="h",
            marker=dict(color=data["count"], colorscale="Inferno",
                        colorbar=dict(title="Alerts")),
            customdata=data["count"],
            hovertemplate="%{y}<br>Under alert: %{x} h<br>"
                          "Alerts that day: %{customdata}<extra></extra>"))
        fig.update_layout(title="Top busiest days (hours under alert)",
                          xaxis_title="Time under alert, hours", **self._LAYOUT)
        return fig

    def oblast_map(self, stats: pd.DataFrame, highlight: str) -> go.Figure:
        """Oblast activity map: point size/color = number of alerts."""
        rows = []
        for name, row in stats.iterrows():
            if name in self.OBLAST_COORDS:
                lat, lon = self.OBLAST_COORDS[name]
                rows.append({"oblast": name, "lat": lat, "lon": lon,
                             "alerts": int(row["alerts"]),
                             "rate": round(float(row["hourly_rate"]) * 100, 1)})
        mp = pd.DataFrame(rows)
        try:  # in Plotly 6 the current API is scatter_map (MapLibre)
            fig = px.scatter_map(
                mp, lat="lat", lon="lon", size="alerts", color="alerts",
                color_continuous_scale="Inferno", size_max=40, zoom=4.2,
                hover_name="oblast",
                hover_data={"alerts": True, "rate": True, "lat": False, "lon": False},
                map_style="open-street-map")
        except AttributeError:  # fallback for older versions
            fig = px.scatter_mapbox(
                mp, lat="lat", lon="lon", size="alerts", color="alerts",
                color_continuous_scale="Inferno", size_max=40, zoom=4.2,
                hover_name="oblast", mapbox_style="open-street-map")
        fig.update_layout(title="Alert activity map by oblast",
                          margin=dict(l=0, r=0, t=50, b=0),
                          paper_bgcolor="rgba(0,0,0,0)")
        return fig
