# -*- coding: utf-8 -*-
"""
forecasting.py
==============
AlertForecaster class — the machine-learning model and probabilistic forecast.

The task is BINARY CLASSIFICATION:
    0 — no alert during this hour
    1 — an alert is active during this hour

Algorithm: Gradient Boosting (HistGradientBoostingClassifier) — a fast and
accurate gradient boosting that works with the categorical "oblast" feature
directly and outputs a probability via predict_proba().

The whole trained object is saved to models/model.pkl with joblib and simply
loaded again on subsequent runs.
"""

import os
import datetime as dt

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "model.pkl"
)


class AlertForecaster:
    """Alert-probability model plus reference tables used for explanations."""

    # Features the model is trained on
    FEATURES = ["hour", "day_of_week", "month", "year",
                "recent_cnt", "recent_dur", "oblast"]

    MONTHS_EN = {1: "January", 2: "February", 3: "March", 4: "April",
                 5: "May", 6: "June", 7: "July", 8: "August",
                 9: "September", 10: "October", 11: "November", 12: "December"}

    DAYS_EN = ["on Monday", "on Tuesday", "on Wednesday", "on Thursday",
               "on Friday", "on Saturday", "on Sunday"]

    def __init__(self):
        self.model = None
        self.oblast_list = None
        self.auc = None
        # reference tables are filled in train()
        self.rate_hour = self.rate_month = self.rate_dow = {}
        self.base_rate = self.month_avg_rate = self.dow_avg_rate = {}
        self.monthly_recent = self.oblast_recent = {}
        self.avg_duration = self.monthly_daily_count = self.oblast_daily_count = {}
        self.min_date = self.max_date = None

    # =================================================================
    # Training
    # =================================================================
    def train(self, grid: pd.DataFrame, df_clean: pd.DataFrame) -> "AlertForecaster":
        """Trains the classifier and builds the reference tables. Returns self."""
        self.oblast_list = sorted(df_clean["oblast"].unique())

        data = grid.sort_values("ts").reset_index(drop=True)
        X = data[self.FEATURES].copy()
        X["oblast"] = pd.Categorical(X["oblast"], categories=self.oblast_list)
        y = data["alert"].astype(int)

        # Split by TIME: train on the past, validate on the more recent part.
        cut = int(len(data) * 0.8)
        self.model = HistGradientBoostingClassifier(
            categorical_features="from_dtype",
            max_iter=200, learning_rate=0.1,
            max_leaf_nodes=63, random_state=42)
        self.model.fit(X.iloc[:cut], y.iloc[:cut])
        self.auc = float(roc_auc_score(y.iloc[cut:],
                                       self.model.predict_proba(X.iloc[cut:])[:, 1]))

        # Final model — trained on all data (for production forecasting)
        self.model.fit(X, y)

        self._build_reference_tables(grid, df_clean)
        return self

    def _build_reference_tables(self, grid: pd.DataFrame, df_clean: pd.DataFrame) -> None:
        """Tables to fill in features of a future date and to build explanations."""
        rate_hour = grid.groupby(["oblast", "hour"])["alert"].mean()
        rate_month = grid.groupby(["oblast", "month"])["alert"].mean()
        rate_dow = grid.groupby(["oblast", "day_of_week"])["alert"].mean()

        self.rate_hour = rate_hour.to_dict()
        self.rate_month = rate_month.to_dict()
        self.rate_dow = rate_dow.to_dict()
        self.base_rate = grid.groupby("oblast")["alert"].mean().to_dict()
        self.month_avg_rate = rate_month.groupby(level=0).mean().to_dict()
        self.dow_avg_rate = rate_dow.groupby(level=0).mean().to_dict()

        # Seasonal estimate of the recent-* features (mean by oblast+month).
        monthly = grid.groupby(["oblast", "month"])[["recent_cnt", "recent_dur"]].mean()
        oblast_r = grid.groupby("oblast")[["recent_cnt", "recent_dur"]].mean()
        self.monthly_recent = {k: (float(v["recent_cnt"]), float(v["recent_dur"]))
                               for k, v in monthly.iterrows()}
        self.oblast_recent = {k: (float(v["recent_cnt"]), float(v["recent_dur"]))
                              for k, v in oblast_r.iterrows()}

        self.avg_duration = (df_clean[df_clean["duration_ok"]]
                             .groupby("oblast")["duration_min"].mean().to_dict())

        # Expected number of alerts per day (mean by oblast+month and by oblast)
        daily_cnt = df_clean.groupby(["oblast", "date"]).size().reset_index(name="c")
        daily_cnt["month"] = daily_cnt["date"].dt.month
        self.monthly_daily_count = daily_cnt.groupby(["oblast", "month"])["c"].mean().to_dict()
        self.oblast_daily_count = daily_cnt.groupby("oblast")["c"].mean().to_dict()

        self.min_date = grid["date"].min()
        self.max_date = grid["date"].max()

    # =================================================================
    # Save / load / "train or load"
    # =================================================================
    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "AlertForecaster":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("Model file is not compatible with AlertForecaster")
        return obj

    @classmethod
    def get_or_train(cls, grid: pd.DataFrame, df_clean: pd.DataFrame,
                     path: str = MODEL_PATH, force: bool = False) -> "AlertForecaster":
        """Loads a ready model or trains a new one and saves it."""
        if os.path.exists(path) and not force:
            try:
                return cls.load(path)
            except Exception:
                pass  # outdated format / corrupted file — retrain
        obj = cls().train(grid, df_clean)
        obj.save(path)
        return obj

    # =================================================================
    # Probability forecast
    # =================================================================
    def _recent_for(self, oblast: str, month: int):
        """recent_* features: seasonal estimate -> fall back to oblast mean."""
        if (oblast, month) in self.monthly_recent:
            return self.monthly_recent[(oblast, month)]
        return self.oblast_recent.get(oblast, (0.0, 0.0))

    def _feature_frame(self, oblast: str, moments) -> pd.DataFrame:
        """Feature table for one or several moments in time."""
        rows = []
        for m in moments:
            rc, rd = self._recent_for(oblast, m.month)
            rows.append({"hour": m.hour, "day_of_week": m.weekday(),
                         "month": m.month, "year": m.year,
                         "recent_cnt": rc, "recent_dur": rd, "oblast": oblast})
        X = pd.DataFrame(rows)[self.FEATURES]
        X["oblast"] = pd.Categorical(X["oblast"], categories=self.oblast_list)
        return X

    def predict_proba(self, oblast: str, moment: dt.datetime) -> float:
        """Alert probability (0..1) for a specific moment in time."""
        X = self._feature_frame(oblast, [moment])
        return float(self.model.predict_proba(X)[0, 1])

    def predict_day_curve(self, oblast: str, day: dt.date) -> np.ndarray:
        """Alert probability for each of the 24 hours of the selected day."""
        moments = [dt.datetime(day.year, day.month, day.day, h) for h in range(24)]
        X = self._feature_frame(oblast, moments)
        return self.model.predict_proba(X)[:, 1]

    def expected_daily_count(self, oblast: str, day: dt.date) -> float:
        """Expected number of alerts for the day (seasonal estimate by month)."""
        val = self.monthly_daily_count.get((oblast, day.month))
        if val is None:
            val = self.oblast_daily_count.get(oblast, 0.0)
        return float(val)

    @staticmethod
    def risk_level(prob: float) -> dict:
        """Maps a probability to a risk level and a color for the UI."""
        if prob >= 0.66:
            return {"label": "High", "color": "#dc2626", "emoji": "🔴"}
        if prob >= 0.33:
            return {"label": "Medium", "color": "#f59e0b", "emoji": "🟠"}
        return {"label": "Low", "color": "#16a34a", "emoji": "🟢"}

    def explain(self, oblast: str, moment: dt.datetime) -> list:
        """Human-readable reasons for the forecast (compared to the typical level)."""
        hour, month, dow = moment.hour, moment.month, moment.weekday()
        base = self.base_rate.get(oblast, 0.0)
        reasons = []

        r_hour = self.rate_hour.get((oblast, hour), base)
        if base > 0 and r_hour >= base * 1.12:
            reasons.append(f"in this region alerts around {hour:02d}:00 happen more often than usual")
        elif base > 0 and r_hour <= base * 0.88:
            reasons.append(f"around {hour:02d}:00 alerts here are relatively rare")

        r_month = self.rate_month.get((oblast, month), base)
        m_avg = self.month_avg_rate.get(oblast, base)
        if m_avg > 0 and r_month >= m_avg * 1.1:
            reasons.append(f"historically, alert frequency in {self.MONTHS_EN[month]} is above average")
        elif m_avg > 0 and r_month <= m_avg * 0.9:
            reasons.append(f"in {self.MONTHS_EN[month]} alerts were recorded less often than usual")

        r_dow = self.rate_dow.get((oblast, dow), base)
        d_avg = self.dow_avg_rate.get(oblast, base)
        if d_avg > 0 and r_dow >= d_avg * 1.1:
            reasons.append(f"{self.DAYS_EN[dow]} there have been more alerts in similar periods")

        rc, _ = self._recent_for(oblast, month)
        if rc >= 25:
            reasons.append("this season the region recorded many alerts in a row")

        avg_dur = self.avg_duration.get(oblast)
        if avg_dur:
            reasons.append(f"average alert duration in the region ≈ {avg_dur:.0f} min")

        if not reasons:
            reasons.append("the forecast is based on the historical alert frequency in the region")
        return reasons[:4]
