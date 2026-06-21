# -*- coding: utf-8 -*-
"""
data_processing.py
==================
DataProcessor class — loading, cleaning and preparing the air-raid alert data.

Core idea:
  The source CSV is an EVENT LOG (every alert has a start and an end).
  To train a model that answers "will there be an alert in a given hour", we
  need a different shape — an HOURLY GRID: for every (oblast, specific hour)
  pair we set the label to 1 if an alert was active during that hour, else 0.

  build() is exactly what turns the event log into such a grid and adds the
  features used for machine learning.
"""

import os
import numpy as np
import pandas as pd

# Default data path (the data/ folder sits next to the src/ folder)
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "official_data_en.csv"
)


class DataProcessor:
    """
    Loads and prepares the data. After calling prepare() you get:
      • self.df   — cleaned event log (one row = one alert);
      • self.grid — hourly grid with 0/1 labels and features.
    """

    # --- Processing settings ---
    LOCAL_TZ = "Europe/Kyiv"      # the file stores UTC time, we convert to Kyiv time
    MAX_DURATION_HOURS = 24       # alerts longer than a day are treated as outliers
    ROLLING_DAYS = 7              # window for the "alerts in the last N days" feature

    def __init__(self, data_path: str = DEFAULT_DATA_PATH, rolling_days: int | None = None):
        self.data_path = data_path
        self.rolling_days = rolling_days or self.ROLLING_DAYS
        self.df: pd.DataFrame | None = None     # cleaned events
        self.grid: pd.DataFrame | None = None   # hourly grid

    # =================================================================
    # Public preparation pipeline
    # =================================================================
    def prepare(self) -> "DataProcessor":
        """Load + clean + build the hourly grid. Returns self."""
        self.df = self._clean(self._load_raw())
        self.grid = self._build_hourly_dataset()
        return self

    # =================================================================
    # Loading and cleaning
    # =================================================================
    def _load_raw(self) -> pd.DataFrame:
        """Reads the CSV (only the needed columns)."""
        cols = ["oblast", "level", "started_at", "finished_at"]
        return pd.read_csv(self.data_path, usecols=cols)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data cleaning:
          - parse dates (UTC -> local time);
          - drop missing values and invalid intervals;
          - compute alert duration;
          - flag duration outliers;
          - add helper columns (local start/end time, date, hour).
        """
        df = df.copy()

        # Dates: utc=True normalizes everything to one zone; format speeds up parsing.
        df["started_at"] = pd.to_datetime(df["started_at"], utc=True,
                                          format="ISO8601", errors="coerce")
        df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True,
                                           format="ISO8601", errors="coerce")

        # Missing values and invalid intervals (end before start)
        df = df.dropna(subset=["started_at", "finished_at"])
        df = df[df["finished_at"] >= df["started_at"]].copy()

        # Duration in minutes
        df["duration_min"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60.0
        # Outlier flag (such rows are excluded when averaging duration)
        df["duration_ok"] = df["duration_min"] <= self.MAX_DURATION_HOURS * 60
        # "Capped" duration: keeps outliers from dominating aggregate metrics
        df["duration_capped"] = df["duration_min"].clip(upper=self.MAX_DURATION_HOURS * 60)

        # Local (naive, tz-free) start and end time of each alert.
        df["start_local"] = df["started_at"].dt.tz_convert(self.LOCAL_TZ).dt.tz_localize(None)
        df["end_local"] = df["finished_at"].dt.tz_convert(self.LOCAL_TZ).dt.tz_localize(None)
        df["date"] = df["start_local"].dt.normalize()   # naive date (tz-free)
        df["hour"] = df["start_local"].dt.hour
        df["weekday"] = df["start_local"].dt.dayofweek
        return df.reset_index(drop=True)

    # =================================================================
    # Turning the event log into an hourly grid with 0/1 labels
    # =================================================================
    def _build_hourly_dataset(self) -> pd.DataFrame:
        """
        Returns a DataFrame where one row = (oblast, specific hour).
        Columns: oblast, ts, alert (0/1), hour, day_of_week, month, year, date,
                 recent_cnt, recent_dur.
        """
        df = self.df
        oblast_list = self.oblasts
        code_of = {name: i for i, name in enumerate(oblast_list)}
        oblast_codes = df["oblast"].map(code_of).to_numpy()

        # --- 1. Start/end of every alert -> "hour index" ---
        # astype('datetime64[h]') truncates the time to a whole hour and does not
        # depend on the internal date resolution (microseconds in pandas 3.0).
        start_naive = df["start_local"]
        fin_local = df["end_local"]
        cap = start_naive + pd.Timedelta(hours=self.MAX_DURATION_HOURS)
        fin_capped = pd.Series(np.minimum(fin_local.to_numpy(), cap.to_numpy()))

        start_idx = start_naive.to_numpy().astype("datetime64[h]").astype("int64")
        end_idx = fin_capped.to_numpy().astype("datetime64[h]").astype("int64")

        # --- 2. "Unroll" every alert into the list of hours it covers ---
        counts = np.clip(end_idx - start_idx + 1, 1, None)
        total = int(counts.sum())
        oblast_rep = np.repeat(oblast_codes, counts)
        base = np.repeat(start_idx, counts)
        within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        pos_hour_idx = base + within

        BIG = 1_000_000_000
        pos_keys = np.unique(oblast_rep.astype(np.int64) * BIG + pos_hour_idx)

        # --- 3. Full grid: all oblasts × all hours of the period ---
        min_idx, max_idx = int(start_idx.min()), int(end_idx.max())
        all_hours = np.arange(min_idx, max_idx + 1)
        n_obl = len(oblast_list)
        grid_oblast = np.repeat(np.arange(n_obl), len(all_hours))
        grid_hour = np.tile(all_hours, n_obl)
        grid_keys = grid_oblast.astype(np.int64) * BIG + grid_hour
        alert = np.isin(grid_keys, pos_keys).astype(np.int8)

        # --- 4. Time-based features ---
        ts = pd.DatetimeIndex(grid_hour.astype("int64").astype("datetime64[h]"))
        grid = pd.DataFrame({
            "oblast": np.array(oblast_list)[grid_oblast],
            "ts": ts, "alert": alert,
            "hour": ts.hour, "day_of_week": ts.dayofweek,
            "month": ts.month, "year": ts.year, "date": ts.normalize(),
        })

        # --- 5. "Recent activity" features ---
        return self._add_recent_features(grid, oblast_list)

    def _add_recent_features(self, grid: pd.DataFrame, oblast_list: list) -> pd.DataFrame:
        """
        Adds per-day/oblast features:
          recent_cnt — number of alerts over the PREVIOUS N days;
          recent_dur — average duration over the previous N days.
        shift(1) guarantees only the past is used (no leakage).
        """
        df = self.df
        base = pd.DataFrame({
            "oblast": df["oblast"].to_numpy(), "date": df["date"].to_numpy(),
            "dur": df["duration_min"].to_numpy(), "ok": df["duration_ok"].to_numpy(),
        })
        cnt_daily = base.groupby(["oblast", "date"]).size().rename("cnt")
        dur_daily = base[base["ok"]].groupby(["oblast", "date"])["dur"].mean().rename("dur")
        daily = pd.concat([cnt_daily, dur_daily], axis=1)

        full_dates = pd.date_range(grid["date"].min(), grid["date"].max(), freq="D")
        idx = pd.MultiIndex.from_product([oblast_list, full_dates], names=["oblast", "date"])
        daily = daily.reindex(idx)
        daily["cnt"] = daily["cnt"].fillna(0.0)
        daily["dur"] = daily.groupby(level=0)["dur"].ffill().fillna(0.0)

        n = self.rolling_days
        daily["recent_cnt"] = (daily.groupby(level=0)["cnt"]
                               .transform(lambda s: s.rolling(n, min_periods=1).sum().shift(1))
                               .fillna(0.0))
        daily["recent_dur"] = (daily.groupby(level=0)["dur"]
                               .transform(lambda s: s.rolling(n, min_periods=1).mean().shift(1))
                               .fillna(0.0))

        recent = daily.reset_index()[["oblast", "date", "recent_cnt", "recent_dur"]]
        grid = grid.merge(recent, on=["oblast", "date"], how="left")
        grid[["recent_cnt", "recent_dur"]] = grid[["recent_cnt", "recent_dur"]].fillna(0.0)
        return grid

    # =================================================================
    # Slices for statistics and charts
    # =================================================================
    @property
    def oblasts(self) -> list:
        """Sorted list of oblasts present in the data."""
        return sorted(self.df["oblast"].unique())

    def daily_counts(self, oblast: str) -> pd.Series:
        """Number of alerts per day for the selected oblast."""
        sub = self.df[self.df["oblast"] == oblast]
        s = sub.groupby("date").size().sort_index().asfreq("D", fill_value=0)
        s.name = "alerts"
        return s

    def oblast_stats(self) -> pd.DataFrame:
        """Summary over all oblasts: total alerts, share of "alert" hours, avg duration."""
        total = self.df.groupby("oblast").size().rename("alerts")
        base_rate = self.grid.groupby("oblast")["alert"].mean().rename("hourly_rate")
        avg_dur = (self.df[self.df["duration_ok"]].groupby("oblast")["duration_min"]
                   .mean().rename("avg_duration_min"))
        out = pd.concat([total, base_rate, avg_dur], axis=1).fillna(0.0)
        return out.sort_values("alerts", ascending=False)

    def region_summary(self, oblast: str) -> dict:
        """Short statistics for a single oblast — used by the UI cards."""
        sub = self.df[self.df["oblast"] == oblast]
        g = self.grid[self.grid["oblast"] == oblast]
        return {
            "total_alerts": int(len(sub)),
            "days": int(sub["date"].nunique()),
            "avg_per_day": float(self.daily_counts(oblast).mean()),
            "median_duration": float(sub[sub["duration_ok"]]["duration_min"].median()),
            "hourly_rate": float(g["alert"].mean()),
            "peak_hour": int(g.groupby("hour")["alert"].mean().idxmax()),
        }
