# -*- coding: utf-8 -*-
"""
app.py — аналитическая система «Воздушные тревоги Украины» (ООП-версия)
======================================================================
Запуск:  streamlit run app.py

Архитектура (ООП):
  • DataProcessor      — данные и почасовая сетка;
  • AlertForecaster    — модель и вероятностный прогноз;
  • HistoricalAnalyzer — проверка факта и самые напряжённые дни;
  • Visualizer         — графики;
  • AlertApp           — интерфейс Streamlit (три вкладки).
"""

import datetime as dt

import plotly.graph_objects as go
import streamlit as st

from src.data_processing import DataProcessor, DEFAULT_DATA_PATH
from src.forecasting import AlertForecaster
from src.historical_analysis import HistoricalAnalyzer
from src.visualization import Visualizer


@st.cache_resource(show_spinner=False)
def build_components(data_path: str):
    """Тяжёлая инициализация (один раз за сессию, кэшируется Streamlit)."""
    proc = DataProcessor(data_path).prepare()
    forecaster = AlertForecaster.get_or_train(proc.grid, proc.df)  # обучит при 1-м запуске
    analyzer = HistoricalAnalyzer(proc.df)
    viz = Visualizer(proc)
    return {
        "proc": proc, "forecaster": forecaster, "analyzer": analyzer, "viz": viz,
        "stats": proc.oblast_stats(),
        "busiest": analyzer.busiest_days(),
        "top10": analyzer.top_busiest_days(10),
    }


class AlertApp:
    """Streamlit-приложение: связывает доменные объекты и рисует интерфейс."""

    CSS = """
    <style>
        .main .block-container {padding-top: 2rem;}
        .card {background: var(--secondary-background-color); border-radius: 16px;
               padding: 22px 26px; margin-bottom: 8px; box-shadow: 0 2px 10px rgba(0,0,0,.08);}
        .card .big {font-size: 56px; font-weight: 800; line-height: 1;}
        .card .sub {font-size: 22px; font-weight: 700; margin-top: 6px;}
        .card .moment {opacity: .7; margin-top: 6px; font-size: 14px;}
        .reason-box {background: var(--secondary-background-color); border-radius: 14px;
                     padding: 16px 20px;}
        .reason-box li {margin-bottom: 6px;}
    </style>"""

    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
        st.set_page_config(page_title="Воздушные тревоги Украины",
                           page_icon="🛡️", layout="wide")
        st.markdown(self.CSS, unsafe_allow_html=True)

        with st.spinner("Загружаю данные и готовлю модель… "
                        "(первый запуск ~1–2 минуты, дальше — мгновенно)"):
            c = build_components(data_path)
        self.proc = c["proc"]
        self.forecaster = c["forecaster"]
        self.analyzer = c["analyzer"]
        self.viz = c["viz"]
        self.stats = c["stats"]
        self.busiest = c["busiest"]
        self.top10 = c["top10"]

        self.oblasts = self.forecaster.oblast_list
        self.default_oblast = ("Kyivska oblast" if "Kyivska oblast" in self.oblasts
                               else self.oblasts[0])
        self.data_min = self.forecaster.min_date.date()
        self.data_max = self.forecaster.max_date.date()

    # =================================================================
    # Точка входа
    # =================================================================
    def run(self) -> None:
        self._render_sidebar()
        st.title("🛡️ Аналитическая система воздушных тревог Украины")
        st.caption("Основано на реальных исторических данных. Три режима: прогноз "
                   "на 30 дней, проверка факта по истории и самые напряжённые дни.")
        tab1, tab2, tab3 = st.tabs([
            "🔮 Прогноз на 30 дней",
            "🔎 Историческая проверка",
            "🔥 Самые напряжённые дни"])
        with tab1:
            self._render_forecast_tab()
        with tab2:
            self._render_history_tab()
        with tab3:
            self._render_busiest_tab()

    # =================================================================
    # Переиспользуемые элементы
    # =================================================================
    def _render_sidebar(self) -> None:
        st.sidebar.title("ℹ️ О системе")
        st.sidebar.markdown(
            f"**Период данных:** {self.data_min} — {self.data_max}\n\n"
            f"**Областей:** {len(self.oblasts)}\n\n"
            f"**Качество модели (ROC-AUC):** {self.forecaster.auc:.3f}")
        st.sidebar.markdown("---")
        st.sidebar.caption("⚠️ Это аналитический инструмент, а не система оповещения. "
                           "Реальные тревоги — только в официальных источниках.")

    @staticmethod
    def _gauge(prob: float, color: str) -> None:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 40}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                   "steps": [{"range": [0, 33], "color": "#dcfce7"},
                             {"range": [33, 66], "color": "#fef9c3"},
                             {"range": [66, 100], "color": "#fee2e2"}]}))
        fig.update_layout(height=240, margin=dict(l=20, r=20, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    def _region_metrics(self, oblast: str) -> None:
        s = self.proc.region_summary(oblast)
        c = st.columns(5)
        c[0].metric("Всего тревог", f"{s['total_alerts']:,}")
        c[1].metric("В среднем в день", f"{s['avg_per_day']:.1f}")
        c[2].metric("Медианная длительность", f"{s['median_duration']:.0f} мин")
        c[3].metric("Доля «тревожных» часов", f"{s['hourly_rate']*100:.0f}%")
        c[4].metric("Пиковый час", f"{s['peak_hour']:02d}:00")

    # =================================================================
    # Вкладка 1. Прогноз на будущее (30 дней)
    # =================================================================
    def _render_forecast_tab(self) -> None:
        st.subheader("Прогноз вероятности тревоги на ближайшие 30 дней")
        today = dt.date.today()
        horizon = today + dt.timedelta(days=30)
        st.caption(f"Доступный диапазон: **{today:%d.%m.%Y} — {horizon:%d.%m.%Y}** "
                   f"(сегодня + 30 дней).")

        cc = st.columns([1.2, 1, 1, 0.9])
        oblast = cc[0].selectbox("🗺️ Область", self.oblasts,
                                 index=self.oblasts.index(self.default_oblast), key="f_obl")
        f_date = cc[1].date_input("📆 Дата", value=today, min_value=today,
                                  max_value=horizon, key="f_date")
        f_time = cc[2].time_input("🕒 Время", value=dt.time(22, 0), key="f_time")
        cc[3].markdown("<br>", unsafe_allow_html=True)
        if cc[3].button("🔮 Прогноз", type="primary", width="stretch", key="f_btn"):
            st.session_state["did_forecast"] = True

        if st.session_state.get("did_forecast"):
            moment = dt.datetime.combine(f_date, f_time)
            prob = self.forecaster.predict_proba(oblast, moment)
            risk = self.forecaster.risk_level(prob)
            exp_cnt = self.forecaster.expected_daily_count(oblast, f_date)
            reasons = self.forecaster.explain(oblast, moment)

            st.markdown("#### Результат")
            a, b = st.columns([1, 1.3])
            with a:
                st.markdown(f"""
                <div class="card" style="border-left:8px solid {risk['color']}">
                    <div class="big" style="color:{risk['color']}">{prob*100:.0f}%</div>
                    <div class="sub" style="color:{risk['color']}">
                        {risk['emoji']} Риск: {risk['label']}</div>
                    <div class="moment">{oblast} · {moment:%d.%m.%Y} · {moment:%H:%M}
                        <br>Предполагаемое число тревог за день: <b>≈ {exp_cnt:.0f}</b></div>
                </div>""", unsafe_allow_html=True)
                self._gauge(prob, risk["color"])
            with b:
                reasons_html = "".join(f"<li>{r}</li>" for r in reasons)
                st.markdown(f'<div class="reason-box"><b>Причины:</b>'
                            f'<ul>{reasons_html}</ul></div>', unsafe_allow_html=True)
                probs = self.forecaster.predict_day_curve(oblast, f_date)
                st.plotly_chart(self.viz.hour_probability(probs, f_time.hour),
                                width="stretch")
            st.divider()
        else:
            st.info("👆 Выберите область, дату и время, затем нажмите «🔮 Прогноз».")

        st.markdown(f"##### 📊 Статистика региона: {oblast}")
        self._region_metrics(oblast)
        with st.expander("📈 История тревог по дням и тепловая карта"):
            st.plotly_chart(self.viz.daily_counts(oblast), width="stretch")
            st.plotly_chart(self.viz.heatmap(oblast), width="stretch")

    # =================================================================
    # Вкладка 2. Историческая проверка
    # =================================================================
    def _render_history_tab(self) -> None:
        st.subheader("Была ли тревога в конкретный момент? (по реальным данным)")
        inv = self.analyzer.INVASION_START
        st.caption(f"Диапазон проверки: **{inv:%d.%m.%Y} — {self.data_max:%d.%m.%Y}** "
                   f"(от начала вторжения до конца данных).")

        cc = st.columns([1.2, 1, 1, 0.9])
        oblast = cc[0].selectbox("🗺️ Область", self.oblasts,
                                 index=self.oblasts.index(self.default_oblast), key="h_obl")
        h_date = cc[1].date_input("📆 Дата", value=self.data_max, min_value=inv,
                                  max_value=self.data_max, key="h_date")
        h_time = cc[2].time_input("🕒 Время", value=dt.time(22, 0), key="h_time")
        cc[3].markdown("<br>", unsafe_allow_html=True)
        go_check = cc[3].button("🔎 Проверить", type="primary", width="stretch", key="h_btn")

        if not go_check:
            st.info("👆 Выберите область, дату и время, затем нажмите «🔎 Проверить».")
            return

        moment = dt.datetime.combine(h_date, h_time)
        res = self.analyzer.check_alert_at(oblast, moment)
        fmt = self.analyzer.format_duration
        st.markdown("#### Результат проверки")
        if res["was_alert"]:
            extra = (f" (одновременно активных тревог: {res['overlap_count']})"
                     if res["overlap_count"] > 1 else "")
            st.markdown(f"""
            <div class="card" style="border-left:8px solid #dc2626">
                <div class="big" style="color:#dc2626">🔴 Тревога была: ДА</div>
                <div class="moment">{oblast} · {moment:%d.%m.%Y %H:%M}{extra}</div>
            </div>""", unsafe_allow_html=True)
            m = st.columns(3)
            m[0].metric("🟢 Начало", res["started"].strftime("%d.%m.%Y %H:%M"))
            m[1].metric("🔴 Окончание", res["finished"].strftime("%d.%m.%Y %H:%M"))
            m[2].metric("⏱️ Длительность", fmt(res["duration_min"]))
        else:
            st.markdown(f"""
            <div class="card" style="border-left:8px solid #16a34a">
                <div class="big" style="color:#16a34a">🟢 Тревоги не было: НЕТ</div>
                <div class="moment">{oblast} · {moment:%d.%m.%Y %H:%M}</div>
            </div>""", unsafe_allow_html=True)
            st.caption("В данных нет тревоги, которая охватывала бы этот момент "
                       "в выбранной области.")

    # =================================================================
    # Вкладка 3. Самые напряжённые дни
    # =================================================================
    def _render_busiest_tab(self) -> None:
        st.subheader("Самый тяжёлый день каждой области за весь период войны")
        st.caption("Критерий: день с максимальным **временем под тревогой** "
                   "(объединение интервалов всех тревог, ≤ 24 ч/день).")
        fmt = self.analyzer.format_duration

        worst = self.busiest.iloc[0]
        m = st.columns(3)
        m[0].metric("🥇 Самая тяжёлая область", worst["oblast"])
        m[1].metric("📆 День", worst["date"].strftime("%d.%m.%Y"))
        m[2].metric("⏱️ Под тревогой", fmt(worst["total_min"]))

        st.plotly_chart(self.viz.top_busiest(self.top10), width="stretch")

        st.markdown("##### Таблица: самый напряжённый день по каждой области")
        table = self.busiest.copy()
        table.insert(0, "№", range(1, len(table) + 1))
        table["Дата"] = table["date"].dt.strftime("%d.%m.%Y")
        table["Под тревогой"] = table["total_min"].map(fmt)
        table = table.rename(columns={"oblast": "Область", "count": "Кол-во тревог"})
        st.dataframe(
            table[["№", "Область", "Дата", "Под тревогой", "Кол-во тревог"]],
            hide_index=True, width="stretch")

        st.markdown("##### Сравнение областей и карта активности")
        cc = st.columns(2)
        with cc[0]:
            st.plotly_chart(self.viz.oblast_bar(self.stats, worst["oblast"]),
                            width="stretch")
        with cc[1]:
            st.plotly_chart(self.viz.oblast_map(self.stats, worst["oblast"]),
                            width="stretch")


# Streamlit выполняет файл как скрипт — просто создаём приложение и запускаем.
AlertApp().run()
