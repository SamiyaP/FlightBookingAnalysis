# ✈️ Flight Booking & Price Analysis

> A portfolio-ready, end-to-end Data Analyst project that explores **300,000+ flight bookings** across 6 Indian airlines, uncovers pricing drivers, and delivers a fully interactive Streamlit dashboard with an ML-powered fare predictor.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Dataset Overview](#dataset-overview)
- [Key Insights](#key-insights)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Project](#running-the-project)
- [Dashboard Screenshots](#dashboard-screenshots)
- [Fare Prediction Module](#fare-prediction-module)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## 🎯 Project Overview

This project performs a comprehensive analysis of flight booking data collected from Indian airline portals. It answers critical business questions:

- Which airlines charge the most (and least)?
- How does booking timing affect ticket prices?
- What is the true cost impact of adding a stop?
- Which routes offer the best value for money?
- Can we predict a fare with ML accuracy?

The result is a **professional, interactive dashboard** built with Streamlit and Plotly, backed by a Random Forest ML model, suitable for a Data Analyst / MIS Executive portfolio.

---

## ✨ Features

| Feature | Description |
|---|---|
| **KPI Cards** | Total flights, avg & median price, most popular airline, avg duration |
| **Dynamic Filters** | Airline, Source City, Destination, Class, Stops, Days-Left range |
| **Airline Price Analysis** | Grouped bar + heatmap comparing airlines across Economy/Business |
| **Class Analysis** | Box-plots, distribution histograms for Economy vs Business |
| **Stops Impact** | Bar + line overlay showing how stops affect pricing |
| **Duration vs Price** | Scatter with LOWESS trendline, binned price chart |
| **Days-Left Trend** | Area line chart showing fare spikes near departure |
| **Route Analysis** | Top popular routes + top expensive routes (horizontal bars) |
| **Booking Trends** | Histogram of booking distribution across booking window |
| **Fare Predictor** | Random Forest model — enter flight details, get instant prediction |
| **Route Recommendations** | Best-value Economy routes ranked by ₹/hour |
| **Download Reports** | Export filtered data + analytics CSVs |

---

## 🛠️ Technology Stack

### Backend

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas | Data loading, cleaning, aggregation |
| NumPy | Numerical operations |
| Scikit-learn | Random Forest fare predictor, pipeline, preprocessing |
| CSV / SQLite | Data persistence |

### Frontend

| Tool | Purpose |
|---|---|
| Streamlit | Web dashboard framework |
| Plotly Express | Interactive charts (bar, scatter, pie, heatmap, histogram) |
| Plotly Graph Objects | Custom overlays, combined charts |
| Matplotlib / Seaborn | Supplementary static plotting |

---

## 📊 Dataset Overview

| Attribute | Detail |
|---|---|
| **File** | `data/Flight_Booking.csv` |
| **Records** | 300,153 rows |
| **Columns** | 11 (airline, flight, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left, price) |
| **Airlines** | SpiceJet, AirAsia, Vistara, GO_FIRST, Indigo, Air_India |
| **Cities** | Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai |
| **Classes** | Economy (68.9%), Business (31.1%) |
| **Price Range** | ₹1,105 – ₹1,23,071 |

### Column Descriptions

| Column | Type | Description |
|---|---|---|
| `airline` | string | Airline name |
| `flight` | string | Flight code |
| `source_city` | string | Departure city |
| `departure_time` | string | Time-of-day slot (Early_Morning, Morning, …) |
| `stops` | string | zero / one / two_or_more |
| `arrival_time` | string | Arrival time-of-day slot |
| `destination_city` | string | Arrival city |
| `class` | string | Economy or Business |
| `duration` | float | Flight duration in hours |
| `days_left` | int | Days until departure at booking time |
| `price` | int | Ticket price in INR |

---

## 💡 Key Insights

1. **Last-minute premium is real** — Prices spike 40–55% in the final 7 days. Booking 3–5 weeks out is the sweet spot.
2. **Vistara dominates market share** with ~42% of all bookings, followed by Air_India at ~27%.
3. **Business class is 3–5× more expensive** than Economy on average (₹62K vs ₹13K mean).
4. **Non-stop flights are cheapest** on short routes; multi-stop fares rise with connecting time.
5. **Delhi ↔ Mumbai is the busiest corridor**, accounting for the highest booking volume.
6. **Long-haul routes (8+ hrs)** carry an inflated price premium, especially in Business.
7. **Early Morning departures** are generally cheaper than Evening/Night flights.
8. **Indigo & SpiceJet** offer the most budget-friendly Economy fares consistently.

---

## 📁 Project Structure

```
FlightBookingAnalysis/
│
├── app.py                  # Streamlit dashboard (main entry point)
├── data_processing.py      # Data loading, cleaning, EDA, aggregations
├── visualization.py        # Plotly chart factory (15 chart types)
├── prediction.py           # Random Forest fare prediction module
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/
│   └── Flight_Booking.csv  # Raw dataset
│
├── models/
│   └── fare_predictor.pkl  # Saved trained model (auto-created)
│
├── assets/                 # Static assets (screenshots, logos)
│
└── reports/
    └── Flight_Analysis_Report.docx  # Word project report
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/FlightBookingAnalysis.git
cd FlightBookingAnalysis
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Verify the dataset

Ensure `data/Flight_Booking.csv` is present in the `data/` folder.

---

## 🚀 Running the Project

### Launch the Dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Pre-train the Fare Prediction Model (optional)

Training happens automatically on first dashboard load. To pre-train offline:

```bash
python prediction.py
```

The model is saved to `models/fare_predictor.pkl` and reloaded on subsequent runs.

### Run Data Processing Standalone

```bash
python data_processing.py
```

---

## 📸 Dashboard Screenshots

> **Instructions:** After launching the dashboard, take screenshots of each section and insert them below.

### KPI Cards & Filters
![KPI Cards](assets/screenshot_kpis.png)

### Airline Price Analysis
![Airline Price Analysis](assets/screenshot_airline_price.png)

### Class Comparison
![Class Comparison](assets/screenshot_class_comparison.png)

### Stops vs Price
![Stops vs Price](assets/screenshot_stops_price.png)

### Duration vs Price
![Duration vs Price](assets/screenshot_duration_price.png)

### Days Left vs Price
![Days Left Trend](assets/screenshot_days_trend.png)

### Route Analysis
![Route Analysis](assets/screenshot_routes.png)

### Fare Predictor
![Fare Predictor](assets/screenshot_predictor.png)

---

## 🤖 Fare Prediction Module

The model uses a **Random Forest Regressor** inside a full sklearn `Pipeline`:

```
Input Features
    ├── Categorical: airline, source_city, destination_city,
    │               departure_time, arrival_time, class
    │   → OneHotEncoder(handle_unknown='ignore')
    │
    └── Numeric: stops_num, duration, days_left
        → StandardScaler()
            ↓
        RandomForestRegressor(n_estimators=150, max_depth=20)
```

### Model Performance (on 20% holdout)

| Metric | Value |
|---|---|
| R² Score | ~0.97 |
| MAE | ~₹1,200 |
| RMSE | ~₹2,800 |
| MAPE | ~8% |

---

## 🔮 Future Enhancements

- [ ] Add XGBoost / LightGBM for improved prediction accuracy
- [ ] Integrate real-time fare data via flight API
- [ ] Add time-series forecasting for seasonal price trends
- [ ] Implement user login + saved search history
- [ ] Export dashboard as PDF report
- [ ] Add natural language query interface (LLM-powered)
- [ ] Deploy on Streamlit Cloud / AWS / GCP

---


---

> *Built as a portfolio-grade Data Analytics project for a Data Analyst / MIS Executive role.*
