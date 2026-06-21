# -*- coding: utf-8 -*-
"""
forecasting.py
==============
Класс AlertForecaster — модель машинного обучения и вероятностный прогноз.

Задача — БИНАРНАЯ КЛАССИФИКАЦИЯ:
    0 — в этот час тревоги нет
    1 — в этот час тревога есть

Алгоритм: Gradient Boosting (HistGradientBoostingClassifier) — быстрый и точный
градиентный бустинг, который работает с категориальным признаком «область»
напрямую и выдаёт вероятность через predict_proba().

Обученный объект целиком сохраняется в models/model.pkl через joblib и при
следующих запусках просто загружается.
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
    """Модель прогноза вероятности тревоги + справочные таблицы для объяснений."""

    # Признаки, на которых учится модель
    FEATURES = ["hour", "day_of_week", "month", "year",
                "recent_cnt", "recent_dur", "oblast"]

    MONTHS_RU = {1: "январе", 2: "феврале", 3: "марте", 4: "апреле",
                 5: "мае", 6: "июне", 7: "июле", 8: "августе",
                 9: "сентябре", 10: "октябре", 11: "ноябре", 12: "декабре"}

    DAYS_RU = ["в понедельник", "во вторник", "в среду", "в четверг",
               "в пятницу", "в субботу", "в воскресенье"]

    def __init__(self):
        self.model = None
        self.oblast_list = None
        self.auc = None
        # справочные таблицы заполняются в train()
        self.rate_hour = self.rate_month = self.rate_dow = {}
        self.base_rate = self.month_avg_rate = self.dow_avg_rate = {}
        self.monthly_recent = self.oblast_recent = {}
        self.avg_duration = self.monthly_daily_count = self.oblast_daily_count = {}
        self.min_date = self.max_date = None

    # =================================================================
    # Обучение
    # =================================================================
    def train(self, grid: pd.DataFrame, df_clean: pd.DataFrame) -> "AlertForecaster":
        """Обучает классификатор и строит справочные таблицы. Возвращает self."""
        self.oblast_list = sorted(df_clean["oblast"].unique())

        data = grid.sort_values("ts").reset_index(drop=True)
        X = data[self.FEATURES].copy()
        X["oblast"] = pd.Categorical(X["oblast"], categories=self.oblast_list)
        y = data["alert"].astype(int)

        # Разделение по ВРЕМЕНИ: учимся на прошлом, проверяем на более позднем.
        cut = int(len(data) * 0.8)
        self.model = HistGradientBoostingClassifier(
            categorical_features="from_dtype",
            max_iter=200, learning_rate=0.1,
            max_leaf_nodes=63, random_state=42)
        self.model.fit(X.iloc[:cut], y.iloc[:cut])
        self.auc = float(roc_auc_score(y.iloc[cut:],
                                       self.model.predict_proba(X.iloc[cut:])[:, 1]))

        # Финальная модель — на всех данных (для боевого прогноза)
        self.model.fit(X, y)

        self._build_reference_tables(grid, df_clean)
        return self

    def _build_reference_tables(self, grid: pd.DataFrame, df_clean: pd.DataFrame) -> None:
        """Таблицы для подстановки признаков будущей даты и для объяснений."""
        rate_hour = grid.groupby(["oblast", "hour"])["alert"].mean()
        rate_month = grid.groupby(["oblast", "month"])["alert"].mean()
        rate_dow = grid.groupby(["oblast", "day_of_week"])["alert"].mean()

        self.rate_hour = rate_hour.to_dict()
        self.rate_month = rate_month.to_dict()
        self.rate_dow = rate_dow.to_dict()
        self.base_rate = grid.groupby("oblast")["alert"].mean().to_dict()
        self.month_avg_rate = rate_month.groupby(level=0).mean().to_dict()
        self.dow_avg_rate = rate_dow.groupby(level=0).mean().to_dict()

        # Сезонная оценка recent-признаков (среднее по область+месяц).
        monthly = grid.groupby(["oblast", "month"])[["recent_cnt", "recent_dur"]].mean()
        oblast_r = grid.groupby("oblast")[["recent_cnt", "recent_dur"]].mean()
        self.monthly_recent = {k: (float(v["recent_cnt"]), float(v["recent_dur"]))
                               for k, v in monthly.iterrows()}
        self.oblast_recent = {k: (float(v["recent_cnt"]), float(v["recent_dur"]))
                              for k, v in oblast_r.iterrows()}

        self.avg_duration = (df_clean[df_clean["duration_ok"]]
                             .groupby("oblast")["duration_min"].mean().to_dict())

        # Ожидаемое число тревог в день (среднее по область+месяц и по области)
        daily_cnt = df_clean.groupby(["oblast", "date"]).size().reset_index(name="c")
        daily_cnt["month"] = daily_cnt["date"].dt.month
        self.monthly_daily_count = daily_cnt.groupby(["oblast", "month"])["c"].mean().to_dict()
        self.oblast_daily_count = daily_cnt.groupby("oblast")["c"].mean().to_dict()

        self.min_date = grid["date"].min()
        self.max_date = grid["date"].max()

    # =================================================================
    # Сохранение / загрузка / «обучить или загрузить»
    # =================================================================
    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "AlertForecaster":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("Файл модели несовместим с AlertForecaster")
        return obj

    @classmethod
    def get_or_train(cls, grid: pd.DataFrame, df_clean: pd.DataFrame,
                     path: str = MODEL_PATH, force: bool = False) -> "AlertForecaster":
        """Загружает готовую модель или обучает новую и сохраняет."""
        if os.path.exists(path) and not force:
            try:
                return cls.load(path)
            except Exception:
                pass  # формат устарел/файл повреждён — переобучим
        obj = cls().train(grid, df_clean)
        obj.save(path)
        return obj

    # =================================================================
    # Прогноз вероятности
    # =================================================================
    def _recent_for(self, oblast: str, month: int):
        """Признаки recent_*: сезонная оценка -> среднее по области."""
        if (oblast, month) in self.monthly_recent:
            return self.monthly_recent[(oblast, month)]
        return self.oblast_recent.get(oblast, (0.0, 0.0))

    def _feature_frame(self, oblast: str, moments) -> pd.DataFrame:
        """Таблица признаков для одного или нескольких моментов времени."""
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
        """Вероятность тревоги (0..1) для конкретного момента времени."""
        X = self._feature_frame(oblast, [moment])
        return float(self.model.predict_proba(X)[0, 1])

    def predict_day_curve(self, oblast: str, day: dt.date) -> np.ndarray:
        """Вероятность тревоги по каждому из 24 часов выбранного дня."""
        moments = [dt.datetime(day.year, day.month, day.day, h) for h in range(24)]
        X = self._feature_frame(oblast, moments)
        return self.model.predict_proba(X)[:, 1]

    def expected_daily_count(self, oblast: str, day: dt.date) -> float:
        """Предполагаемое количество тревог за день (сезонная оценка по месяцу)."""
        val = self.monthly_daily_count.get((oblast, day.month))
        if val is None:
            val = self.oblast_daily_count.get(oblast, 0.0)
        return float(val)

    @staticmethod
    def risk_level(prob: float) -> dict:
        """Переводит вероятность в уровень риска и цвет для интерфейса."""
        if prob >= 0.66:
            return {"label": "Высокий", "color": "#dc2626", "emoji": "🔴"}
        if prob >= 0.33:
            return {"label": "Средний", "color": "#f59e0b", "emoji": "🟠"}
        return {"label": "Низкий", "color": "#16a34a", "emoji": "🟢"}

    def explain(self, oblast: str, moment: dt.datetime) -> list:
        """Человекочитаемые причины прогноза (сравнение с типичным уровнем)."""
        hour, month, dow = moment.hour, moment.month, moment.weekday()
        base = self.base_rate.get(oblast, 0.0)
        reasons = []

        r_hour = self.rate_hour.get((oblast, hour), base)
        if base > 0 and r_hour >= base * 1.12:
            reasons.append(f"в этом регионе тревоги около {hour:02d}:00 случаются чаще обычного")
        elif base > 0 and r_hour <= base * 0.88:
            reasons.append(f"в районе {hour:02d}:00 тревоги здесь относительно редки")

        r_month = self.rate_month.get((oblast, month), base)
        m_avg = self.month_avg_rate.get(oblast, base)
        if m_avg > 0 and r_month >= m_avg * 1.1:
            reasons.append(f"исторически в {self.MONTHS_RU[month]} частота тревог выше средней")
        elif m_avg > 0 and r_month <= m_avg * 0.9:
            reasons.append(f"в {self.MONTHS_RU[month]} тревоги фиксировались реже обычного")

        r_dow = self.rate_dow.get((oblast, dow), base)
        d_avg = self.dow_avg_rate.get(oblast, base)
        if d_avg > 0 and r_dow >= d_avg * 1.1:
            reasons.append(f"{self.DAYS_RU[dow]} в похожие периоды тревог было больше")

        rc, _ = self._recent_for(oblast, month)
        if rc >= 25:
            reasons.append("в этот сезон в регионе фиксировалось много тревог подряд")

        avg_dur = self.avg_duration.get(oblast)
        if avg_dur:
            reasons.append(f"средняя длительность тревоги в регионе ≈ {avg_dur:.0f} мин")

        if not reasons:
            reasons.append("прогноз построен по исторической частоте тревог в регионе")
        return reasons[:4]
