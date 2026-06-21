# -*- coding: utf-8 -*-
"""
data_processing.py
==================
Класс DataProcessor — загрузка, очистка и подготовка данных воздушных тревог.

Главная идея:
  Исходный CSV — это ЛОГ СОБЫТИЙ (у каждой тревоги есть начало и конец).
  Чтобы обучить модель «будет ли тревога в такой-то час», нужен другой формат —
  ПОЧАСОВАЯ СЕТКА: для каждой пары (область, конкретный час) ставим метку 1,
  если в этот час шла тревога, и 0 — если нет.

  Метод build() как раз превращает лог событий в такую сетку и добавляет
  признаки для машинного обучения.
"""

import os
import numpy as np
import pandas as pd

# Путь к данным по умолчанию (папка data/ лежит рядом с папкой src/)
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "official_data_en.csv"
)


class DataProcessor:
    """
    Загружает и готовит данные. После вызова prepare() доступны:
      • self.df   — очищенный лог событий (одна строка = одна тревога);
      • self.grid — почасовая сетка с метками 0/1 и признаками.
    """

    # --- Настройки обработки ---
    LOCAL_TZ = "Europe/Kyiv"      # в файле время в UTC, переводим в киевское
    MAX_DURATION_HOURS = 24       # тревоги длиннее суток считаем выбросами
    ROLLING_DAYS = 7              # окно для признака «тревог за последние N дней»

    def __init__(self, data_path: str = DEFAULT_DATA_PATH, rolling_days: int | None = None):
        self.data_path = data_path
        self.rolling_days = rolling_days or self.ROLLING_DAYS
        self.df: pd.DataFrame | None = None     # очищенные события
        self.grid: pd.DataFrame | None = None   # почасовая сетка

    # =================================================================
    # Публичный конвейер подготовки
    # =================================================================
    def prepare(self) -> "DataProcessor":
        """Загрузка + очистка + построение почасовой сетки. Возвращает self."""
        self.df = self._clean(self._load_raw())
        self.grid = self._build_hourly_dataset()
        return self

    # =================================================================
    # Загрузка и очистка
    # =================================================================
    def _load_raw(self) -> pd.DataFrame:
        """Читает CSV (только нужные колонки)."""
        cols = ["oblast", "level", "started_at", "finished_at"]
        return pd.read_csv(self.data_path, usecols=cols)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка данных:
          - парсинг дат (UTC -> местное время);
          - удаление пропусков и некорректных интервалов;
          - расчёт длительности тревоги;
          - пометка выбросов по длительности;
          - вспомогательные колонки (местное время начала/конца, дата, час).
        """
        df = df.copy()

        # Даты: utc=True приводит всё к единому поясу; format ускоряет парсинг.
        df["started_at"] = pd.to_datetime(df["started_at"], utc=True,
                                          format="ISO8601", errors="coerce")
        df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True,
                                           format="ISO8601", errors="coerce")

        # Пропуски и некорректные интервалы (конец раньше начала)
        df = df.dropna(subset=["started_at", "finished_at"])
        df = df[df["finished_at"] >= df["started_at"]].copy()

        # Длительность в минутах
        df["duration_min"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60.0
        # Метка выброса (для расчёта средней длительности такие исключаем)
        df["duration_ok"] = df["duration_min"] <= self.MAX_DURATION_HOURS * 60
        # «Обрезанная» длительность: для суммарных метрик не даём выбросам доминировать
        df["duration_capped"] = df["duration_min"].clip(upper=self.MAX_DURATION_HOURS * 60)

        # Местное (наивное, без tz) время начала и конца каждой тревоги.
        df["start_local"] = df["started_at"].dt.tz_convert(self.LOCAL_TZ).dt.tz_localize(None)
        df["end_local"] = df["finished_at"].dt.tz_convert(self.LOCAL_TZ).dt.tz_localize(None)
        df["date"] = df["start_local"].dt.normalize()   # наивная дата (без tz)
        df["hour"] = df["start_local"].dt.hour
        df["weekday"] = df["start_local"].dt.dayofweek
        return df.reset_index(drop=True)

    # =================================================================
    # Превращение лога событий в почасовую сетку с метками 0/1
    # =================================================================
    def _build_hourly_dataset(self) -> pd.DataFrame:
        """
        Возвращает DataFrame, где одна строка = (область, конкретный час).
        Колонки: oblast, ts, alert (0/1), hour, day_of_week, month, year, date,
                 recent_cnt, recent_dur.
        """
        df = self.df
        oblast_list = self.oblasts
        code_of = {name: i for i, name in enumerate(oblast_list)}
        oblast_codes = df["oblast"].map(code_of).to_numpy()

        # --- 1. Начало/конец каждой тревоги -> «номер часа» ---
        # astype('datetime64[h]') обрезает время до целого часа и не зависит
        # от внутреннего разрешения дат (в pandas 3.0 это микросекунды).
        start_naive = df["start_local"]
        fin_local = df["end_local"]
        cap = start_naive + pd.Timedelta(hours=self.MAX_DURATION_HOURS)
        fin_capped = pd.Series(np.minimum(fin_local.to_numpy(), cap.to_numpy()))

        start_idx = start_naive.to_numpy().astype("datetime64[h]").astype("int64")
        end_idx = fin_capped.to_numpy().astype("datetime64[h]").astype("int64")

        # --- 2. «Разворачиваем» каждую тревогу в список затронутых часов ---
        counts = np.clip(end_idx - start_idx + 1, 1, None)
        total = int(counts.sum())
        oblast_rep = np.repeat(oblast_codes, counts)
        base = np.repeat(start_idx, counts)
        within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        pos_hour_idx = base + within

        BIG = 1_000_000_000
        pos_keys = np.unique(oblast_rep.astype(np.int64) * BIG + pos_hour_idx)

        # --- 3. Полная сетка: все области × все часы периода ---
        min_idx, max_idx = int(start_idx.min()), int(end_idx.max())
        all_hours = np.arange(min_idx, max_idx + 1)
        n_obl = len(oblast_list)
        grid_oblast = np.repeat(np.arange(n_obl), len(all_hours))
        grid_hour = np.tile(all_hours, n_obl)
        grid_keys = grid_oblast.astype(np.int64) * BIG + grid_hour
        alert = np.isin(grid_keys, pos_keys).astype(np.int8)

        # --- 4. Признаки из времени ---
        ts = pd.DatetimeIndex(grid_hour.astype("int64").astype("datetime64[h]"))
        grid = pd.DataFrame({
            "oblast": np.array(oblast_list)[grid_oblast],
            "ts": ts, "alert": alert,
            "hour": ts.hour, "day_of_week": ts.dayofweek,
            "month": ts.month, "year": ts.year, "date": ts.normalize(),
        })

        # --- 5. Признаки «история активности» ---
        return self._add_recent_features(grid, oblast_list)

    def _add_recent_features(self, grid: pd.DataFrame, oblast_list: list) -> pd.DataFrame:
        """
        Добавляет признаки на каждый день/область:
          recent_cnt — тревог за ПРЕДЫДУЩИЕ N дней;
          recent_dur — средняя длительность за предыдущие N дней.
        shift(1) гарантирует использование только прошлого (нет «утечки»).
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
    # Срезы для статистики и графиков
    # =================================================================
    @property
    def oblasts(self) -> list:
        """Отсортированный список областей из данных."""
        return sorted(self.df["oblast"].unique())

    def daily_counts(self, oblast: str) -> pd.Series:
        """Количество тревог по дням для выбранной области."""
        sub = self.df[self.df["oblast"] == oblast]
        s = sub.groupby("date").size().sort_index().asfreq("D", fill_value=0)
        s.name = "alerts"
        return s

    def oblast_stats(self) -> pd.DataFrame:
        """Сводка по всем областям: всего тревог, доля «тревожных» часов, ср. длит."""
        total = self.df.groupby("oblast").size().rename("alerts")
        base_rate = self.grid.groupby("oblast")["alert"].mean().rename("hourly_rate")
        avg_dur = (self.df[self.df["duration_ok"]].groupby("oblast")["duration_min"]
                   .mean().rename("avg_duration_min"))
        out = pd.concat([total, base_rate, avg_dur], axis=1).fillna(0.0)
        return out.sort_values("alerts", ascending=False)

    def region_summary(self, oblast: str) -> dict:
        """Короткая статистика по одной области — для карточек интерфейса."""
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
