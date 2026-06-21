# -*- coding: utf-8 -*-
"""
historical_analysis.py
======================
HistoricalAnalyzer class — analysis of the REAL historical data (no forecast).

Used by two UI tabs:
  • "Historical check"  — was there an alert at a specific moment in time;
  • "Busiest days"      — for each oblast, the day with the maximum amount of
    time spent under alert.
"""

import datetime as dt
import numpy as np
import pandas as pd


class HistoricalAnalyzer:
    """Fact-based analysis of the data (moment check + busiest days)."""

    # Start date of the full-scale invasion — lower bound for the check
    INVASION_START = dt.date(2022, 2, 24)
    MAX_DURATION_HOURS = 24
    _SEC_DAY = 86_400  # seconds in a day

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._daily_cache: pd.DataFrame | None = None  # lazy cache for _daily_load

    # =================================================================
    # Tab 2. Was there an alert at the selected moment?
    # =================================================================
    def check_alert_at(self, oblast: str, moment: dt.datetime) -> dict:
        """
        Checks whether an alert was active in `oblast` at `moment` (local time).
        An alert is active if start_local <= moment <= end_local.

        Returns a dict:
          was_alert=False                       — there was no alert;
          was_alert=True, started, finished,    — there was an alert (we show the
          duration_min, overlap_count             interval with the longest duration).
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
    # Tab 3. Busiest days
    # =================================================================
    def _daily_load(self) -> pd.DataFrame:
        """
        For each (oblast, day) computes:
          total_min — how many MINUTES that day the oblast was under alert;
          count     — how many alerts were declared (records for that day).

        In the data, alerts of different levels (oblast/raion/hromada) overlap
        each other, so a plain sum of durations gives unrealistic values
        (hundreds of hours per day). We take the UNION of intervals and split it
        by calendar day — total_min is capped at 24 hours.
        The result is cached (computed only once).
        """
        if self._daily_cache is not None:
            return self._daily_cache

        df = self.df
        SEC_DAY = self._SEC_DAY

        # Cap the end of each alert at one day from its start (outlier protection)
        cap_end = np.minimum(
            df["end_local"].to_numpy(),
            (df["start_local"] + pd.Timedelta(hours=self.MAX_DURATION_HOURS)).to_numpy())
        codes, oblasts = pd.factorize(df["oblast"].to_numpy())
        s_sec = df["start_local"].to_numpy().astype("datetime64[s]").astype("int64")
        e_sec = cap_end.astype("datetime64[s]").astype("int64")

        # --- 1. Vectorized union of overlapping intervals ---
        w = pd.DataFrame({"ob": codes, "s": s_sec, "e": e_sec}).sort_values(["ob", "s"])
        prev_max_end = w.groupby("ob")["e"].cummax().groupby(w["ob"]).shift()
        new_seg = (w["s"] > prev_max_end) | prev_max_end.isna()
        w["seg"] = new_seg.cumsum()
        merged = w.groupby("seg").agg(ob=("ob", "first"), s=("s", "min"), e=("e", "max"))
        ob = merged["ob"].to_numpy()
        ms = merged["s"].to_numpy()
        me = merged["e"].to_numpy()

        # --- 2. Vectorized split of intervals by calendar day ---
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
        For each oblast — the busiest day (max time under alert; ties broken by
        the number of alerts). Columns: oblast, date, total_min, count.
        """
        daily = self._daily_load().sort_values(["total_min", "count"], ascending=False)
        res = daily.drop_duplicates("oblast", keep="first")
        return res.sort_values(["total_min", "count"], ascending=False).reset_index(drop=True)

    def top_busiest_days(self, n: int = 10) -> pd.DataFrame:
        """Top-N busiest individual days across the whole country (for the chart)."""
        daily = self._daily_load().sort_values(["total_min", "count"], ascending=False)
        top = daily.head(n).reset_index(drop=True)
        top["label"] = top["oblast"] + " · " + top["date"].dt.strftime("%d.%m.%Y")
        return top

    @staticmethod
    def format_duration(minutes: float) -> str:
        """Minutes -> 'Xh Ym' (e.g. 1122 -> '18h 42m')."""
        total = int(round(minutes))
        h, m = divmod(total, 60)
        if h and m:
            return f"{h}h {m}m"
        if h:
            return f"{h}h"
        return f"{m}m"
