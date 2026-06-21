# -*- coding: utf-8 -*-
"""
historical_analysis.py
======================
Класс HistoricalAnalyzer — анализ РЕАЛЬНЫХ исторических данных (без прогноза).

Используется двумя вкладками интерфейса:
  • «Историческая проверка» — была ли тревога в конкретный момент времени;
  • «Самые напряжённые дни» — для каждой области день с максимальным
    временем под тревогой.
"""

import datetime as dt
import numpy as np
import pandas as pd


class HistoricalAnalyzer:
    """Фактологический анализ по данным (проверка момента + напряжённые дни)."""

    # Дата начала полномасштабного вторжения — нижняя граница для проверки
    INVASION_START = dt.date(2022, 2, 24)
    MAX_DURATION_HOURS = 24
    _SEC_DAY = 86_400  # секунд в сутках

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._daily_cache: pd.DataFrame | None = None  # ленивый кэш _daily_load

    # =================================================================
    # Вкладка 2. Была ли тревога в выбранный момент?
    # =================================================================
    def check_alert_at(self, oblast: str, moment: dt.datetime) -> dict:
        """
        Проверяет, шла ли тревога в области `oblast` в момент `moment`
        (местное время). Тревога активна, если start_local <= moment <= end_local.

        Возвращает словарь:
          was_alert=False                       — тревоги не было;
          was_alert=True, started, finished,    — тревога была (показываем интервал
          duration_min, overlap_count             с максимальной длительностью).
        """
        sub = self.df[self.df["oblast"] == oblast]
        mask = (sub["start_local"] <= moment) & (sub["end_local"] >= moment)
        hits = sub[mask]
        if hits.empty:
            return {"was_alert": False}

        row = hits.sort_values("duration_min", ascending=False).iloc[0]
        return {
            "was_alert": True,
            "started": row["start_local"].to_pydatetime(),
            "finished": row["end_local"].to_pydatetime(),
            "duration_min": float(row["duration_min"]),
            "overlap_count": int(len(hits)),
        }

    # =================================================================
    # Вкладка 3. Самые напряжённые дни
    # =================================================================
    def _daily_load(self) -> pd.DataFrame:
        """
        Для каждого (область, день) считает:
          total_min — сколько МИНУТ в этот день область была под тревогой;
          count     — сколько тревог было объявлено (записей за этот день).

        В данных тревоги разных уровней (oblast/raion/hromada) накладываются
        друг на друга, поэтому простая сумма длительностей даёт нереальные
        значения (сотни часов в сутки). Мы берём ОБЪЕДИНЕНИЕ интервалов (union)
        и разбиваем его по календарным дням — total_min ограничен 24 часами.
        Результат кэшируется (считается один раз).
        """
        if self._daily_cache is not None:
            return self._daily_cache

        df = self.df
        SEC_DAY = self._SEC_DAY

        # Ограничиваем конец каждой тревоги сутками от начала (защита от выбросов)
        cap_end = np.minimum(
            df["end_local"].to_numpy(),
            (df["start_local"] + pd.Timedelta(hours=self.MAX_DURATION_HOURS)).to_numpy())
        codes, oblasts = pd.factorize(df["oblast"].to_numpy())
        s_sec = df["start_local"].to_numpy().astype("datetime64[s]").astype("int64")
        e_sec = cap_end.astype("datetime64[s]").astype("int64")

        # --- 1. Векторизованное объединение пересекающихся интервалов ---
        w = pd.DataFrame({"ob": codes, "s": s_sec, "e": e_sec}).sort_values(["ob", "s"])
        prev_max_end = w.groupby("ob")["e"].cummax().groupby(w["ob"]).shift()
        new_seg = (w["s"] > prev_max_end) | prev_max_end.isna()
        w["seg"] = new_seg.cumsum()
        merged = w.groupby("seg").agg(ob=("ob", "first"), s=("s", "min"), e=("e", "max"))
        ob = merged["ob"].to_numpy()
        ms = merged["s"].to_numpy()
        me = merged["e"].to_numpy()

        # --- 2. Векторизованное разбиение интервалов по календарным дням ---
        day0 = ms // SEC_DAY
        day_last = (me - 1) // SEC_DAY
        ndays = np.maximum(day_last - day0 + 1, 0)
        total = int(ndays.sum())
        ob_rep = np.repeat(ob, ndays)
        ms_rep = np.repeat(ms, ndays)
        me_rep = np.repeat(me, ndays)
        within = np.arange(total) - np.repeat(np.cumsum(ndays) - ndays, ndays)
        day = np.repeat(day0, ndays) + within
        seg_start = np.maximum(ms_rep, day * SEC_DAY)
        seg_end = np.minimum(me_rep, (day + 1) * SEC_DAY)
        seg_min = (seg_end - seg_start) / 60.0

        load = (pd.DataFrame({"oblast": oblasts[ob_rep],
                              "date": (day * SEC_DAY).astype("datetime64[s]"),
                              "total_min": seg_min})
                .groupby(["oblast", "date"], as_index=False)["total_min"].sum())
        load["date"] = load["date"].astype("datetime64[ns]").dt.normalize()
        load["total_min"] = load["total_min"].clip(upper=SEC_DAY / 60)

        counts = df.groupby(["oblast", "date"]).size().rename("count").reset_index()
        self._daily_cache = load.merge(counts, on=["oblast", "date"], how="left")
        return self._daily_cache

    def busiest_days(self) -> pd.DataFrame:
        """
        Для каждой области — самый тяжёлый день (макс. время под тревогой; при
        равенстве — больше тревог). Колонки: oblast, date, total_min, count.
        """
        daily = self._daily_load().sort_values(["total_min", "count"], ascending=False)
        res = daily.drop_duplicates("oblast", keep="first")
        return res.sort_values(["total_min", "count"], ascending=False).reset_index(drop=True)

    def top_busiest_days(self, n: int = 10) -> pd.DataFrame:
        """ТОП-N самых напряжённых отдельных дней по всей стране (для графика)."""
        daily = self._daily_load().sort_values(["total_min", "count"], ascending=False)
        top = daily.head(n).reset_index(drop=True)
        top["label"] = top["oblast"] + " · " + top["date"].dt.strftime("%d.%m.%Y")
        return top

    @staticmethod
    def format_duration(minutes: float) -> str:
        """Минуты -> 'Xч Yм' (например, 1122 -> '18ч 42м')."""
        total = int(round(minutes))
        h, m = divmod(total, 60)
        if h and m:
            return f"{h}ч {m}м"
        if h:
            return f"{h}ч"
        return f"{m}м"
