# 🛡️ Ukraine Air-Raid Alerts — Analytics System

An interactive **Streamlit** application for analyzing real historical air-raid
alert data in Ukraine (≈271,000 events, 2022–2026). The app is split into
**three independent modes** (tabs at the top).

---

## 🧭 Three modes

### 1. 🔮 30-day forecast
A probabilistic forecast powered by a machine-learning model.
- The date is restricted to the **"today → +30 days"** range (determined by the
  computer's system date).
- You choose an oblast, a date and a time.
- Result: **alert probability (%)**, a risk level
  (🟢 low / 🟠 medium / 🔴 high), the **estimated number of alerts for the day**,
  reasons, and a probability-by-hour chart.

### 2. 🔎 Historical check
Verifies a **fact** against the data (no forecast).
- Date range: from the **start of the invasion (2022-02-24)** to the last date
  in the CSV.
- The app checks whether the selected moment falls between an alert's
  `started_at` and `finished_at` in that oblast.
- Result: 🔴 **there was an alert** (with start time, end time and duration)
  or 🟢 **there was no alert**.

### 3. 🔥 Busiest days
For each oblast — its **hardest day** over the whole period.
- Criterion: **time under alert** during a day (hours when the oblast had at
  least one active alert).
- Shows: a table for all oblasts (sorted from the busiest), a chart of the
  top-10 busiest days, an oblast comparison, and an activity map.

---

## 📂 Project structure

```
air_alert_project/
├── app.py                     # AlertApp class — Streamlit UI (3 tabs)
├── data/
│   └── official_data_en.csv   # source data
├── models/
│   └── model.pkl              # trained model (created automatically)
├── src/
│   ├── data_processing.py     # DataProcessor class — data and the hourly grid
│   ├── forecasting.py         # AlertForecaster class — model and forecast
│   ├── historical_analysis.py # HistoricalAnalyzer class — fact check, busiest days
│   └── visualization.py       # Visualizer class — Plotly charts
├── outputs/                   # reserved for exports
├── requirements.txt
└── README.md
```

### 🏛️ Architecture (OOP)
The project is built from classes with a clear separation of concerns:

| Class | File | Responsibility |
|---|---|---|
| `DataProcessor` | `data_processing.py` | loading, cleaning, hourly grid, features, statistics |
| `AlertForecaster` | `forecasting.py` | training/loading the model, `predict_proba`, explanations |
| `HistoricalAnalyzer` | `historical_analysis.py` | alert fact check, busiest days |
| `Visualizer` | `visualization.py` | building interactive charts |
| `AlertApp` | `app.py` | wires the objects together and renders the UI (3 tabs) |

The trained model is saved as a whole `AlertForecaster` object via `joblib`
(`AlertForecaster.get_or_train(...)`), and is loaded from disk on startup.

---

## 🚀 Running

Requires **Python 3.10+** (tested on 3.14).

```bash
# 1. Go to the project folder
cd air_alert_project

# 2. (recommended) virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

A browser opens at `http://localhost:8501`.

> **The first run** builds the hourly grid and trains the model (~1–2 minutes),
> then the model is saved to `models/model.pkl` and subsequent runs start
> instantly. To retrain — delete `models/model.pkl`.

### Open in PyCharm / VS Code
Open the `air_alert_project` folder as a project, select an interpreter with the
dependencies installed, and run `streamlit run app.py` in the terminal.
In PyCharm you can create a **Python** run configuration: *module name* =
`streamlit`, *parameters* = `run app.py`, working directory = project root.

---

## 🧠 How the forecast works (mode 1)

1. **From a log to an hourly grid.** The source CSV is a list of events
   (each alert has a start and an end). We turn it into a table
   "(oblast, specific hour) → was there an alert (0/1)" — ~900,000 observations.
2. **Features:** `hour`, `day_of_week`, `month`, `year`, `oblast`,
   `recent_cnt` (alerts in the last 7 days), `recent_dur` (average duration over
   7 days).
3. **Model:** `HistGradientBoostingClassifier` (gradient boosting, scikit-learn).
   Binary classification `0/1`, probability via `predict_proba()`. Quality on a
   time-held-out set is ROC-AUC ≈ 0.90.
4. **Future date.** Real recent activity cannot be known for the future, so the
   `recent_*` features are taken as the **seasonal average** (by month).
5. **Risk level:** `p < 33%` — 🟢 low, `33–66%` — 🟠 medium, `> 66%` — 🔴 high.

---

## ⚙️ Optimization and performance

- **Streamlit caching** (`@st.cache_resource`): data, the hourly grid, the model
  and summary tables are loaded once per session.
- **Date preprocessing** at load time (`format="ISO8601"`, UTC → local
  conversion, precomputed `start_local` / `end_local`).
- **Vectorized computations** (numpy) for the hourly grid and the
  "busiest days" calculation — no slow loops.
- The ready model is cached to disk (`joblib`).

---

## ⚠️ Limitations

- The model reflects **historical patterns**, not the real military situation,
  and **does not predict specific attacks**. It is an analytical tool, not an
  alerting system.
- The future forecast relies on **seasonal averages**, since actual recent
  activity cannot be known in advance.
- External factors (decisions of the parties, weather, target types) are not
  taken into account.
- In the data, alerts of different levels (`oblast` / `raion` / `hromada`)
  overlap each other, so mode 3 uses **time under alert** (union of intervals,
  ≤ 24 h/day) rather than a plain sum of durations (which would give unrealistic
  hundreds of hours per day).
- Individual alert durations are sometimes overstated (outliers > 24 h are
  flagged and capped).

> For real alert signals, rely only on official alerting sources.

---

## 🧰 Stack

`Python` · `Streamlit` · `pandas` · `numpy` · `scikit-learn` · `Plotly` · `joblib`
