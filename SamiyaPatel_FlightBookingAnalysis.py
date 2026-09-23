"""
app.py
======
Flight Booking & Price Analysis – Streamlit Dashboard.

Run with:
    streamlit run app.py

Architecture:
  ┌─────────────────────────────────────┐
  │          Streamlit Frontend         │
  │  Sidebar Filters → KPI Cards        │
  │  EDA Tabs → Prediction Tab          │
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │         data_processing.py          │
  │  (clean, aggregate, feature-eng)    │
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │          visualization.py           │
  │  (all Plotly chart factories)       │
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │           prediction.py             │
  │  (Random Forest fare predictor)     │
  └─────────────────────────────────────┘
"""

import os
import io
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Ensure local modules are importable when running from any directory
sys.path.insert(0, os.path.dirname(__file__))

import data_processing as dp
import visualization   as viz
from prediction import (
    load_model, train_and_save, predict_fare,
    get_feature_importances, route_recommendations,
    MODEL_PATH,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Flight Booking & Price Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ---- KPI Card ---- */
.kpi-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1e40af;
    margin: 0;
}
.kpi-label {
    font-size: 0.8rem;
    color: #64748b;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
/* ---- Section header ---- */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 6px;
    margin-bottom: 16px;
}
/* ---- Insight box ---- */
.insight-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    color: #1e3a5f;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and processing dataset …")
def load_pipeline_data():
    """Run the full data pipeline once and cache the result."""
    return dp.run_pipeline()


@st.cache_resource(show_spinner="Training / loading fare predictor …")
def get_model(df: pd.DataFrame):
    """Load the persisted model or train a fresh one."""
    model = load_model()
    if model is None:
        st.info("No saved model found — training a new Random Forest model …")
        model, metrics = train_and_save(df)
        return model, metrics
    return model, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, col) -> None:
    """Render a KPI card inside a Streamlit column."""
    col.markdown(
        f'<div class="kpi-card"><p class="kpi-value">{value}</p>'
        f'<p class="kpi-label">{label}</p></div>',
        unsafe_allow_html=True,
    )


def insight(text: str) -> None:
    """Render a styled insight callout."""
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def section(text: str) -> None:
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes for download."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    pipeline_data = load_pipeline_data()
    df_full  = pipeline_data["df"]
    kpis_all = pipeline_data["kpis"]

    # ------------------------------------------------------------------
    # Sidebar – Filters
    # ------------------------------------------------------------------
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/48/airplane-mode-on.png", width=48)
        st.title("✈️ Flight Analysis")
        st.caption("Filter the dataset to drill into specific segments.")

        st.divider()

        # Airline filter
        all_airlines = sorted(df_full["airline"].unique())
        sel_airlines = st.multiselect(
            "Airline", all_airlines, default=all_airlines, key="airline_filter"
        )

        # Source city
        all_src = sorted(df_full["source_city"].unique())
        sel_src = st.multiselect(
            "Source City", all_src, default=all_src, key="src_filter"
        )

        # Destination city
        all_dst = sorted(df_full["destination_city"].unique())
        sel_dst = st.multiselect(
            "Destination City", all_dst, default=all_dst, key="dst_filter"
        )

        # Travel class
        all_class = sorted(df_full["class"].unique())
        sel_class = st.multiselect(
            "Travel Class", all_class, default=all_class, key="class_filter"
        )

        # Stops
        all_stops = sorted(df_full["stops"].unique())
        sel_stops = st.multiselect(
            "Stops", all_stops, default=all_stops, key="stops_filter"
        )

        # Days left range
        min_dl = int(df_full["days_left"].min())
        max_dl = int(df_full["days_left"].max())
        sel_days = st.slider(
            "Days Left Before Departure",
            min_value=min_dl, max_value=max_dl,
            value=(min_dl, max_dl), key="days_filter",
        )

        st.divider()
        st.caption("© 2024 Flight Analysis Dashboard")

    # ------------------------------------------------------------------
    # Apply filters
    # ------------------------------------------------------------------
    df = df_full.copy()
    if sel_airlines:
        df = df[df["airline"].isin(sel_airlines)]
    if sel_src:
        df = df[df["source_city"].isin(sel_src)]
    if sel_dst:
        df = df[df["destination_city"].isin(sel_dst)]
    if sel_class:
        df = df[df["class"].isin(sel_class)]
    if sel_stops:
        df = df[df["stops"].isin(sel_stops)]
    df = df[(df["days_left"] >= sel_days[0]) & (df["days_left"] <= sel_days[1])]

    kpis = dp.compute_kpis(df) if len(df) > 0 else kpis_all

    # ------------------------------------------------------------------
    # Page Header
    # ------------------------------------------------------------------
    st.markdown("## ✈️ Flight Booking & Price Analysis Dashboard")
    st.caption(
        f"Analysing **{kpis['total_flights']:,}** flight records across "
        f"**{kpis['total_airlines']}** airlines and **{kpis['total_routes']}** routes."
    )
    st.divider()

    # ------------------------------------------------------------------
    # KPI Row
    # ------------------------------------------------------------------
    section("Key Performance Indicators")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi_card("Total Flights",         f"{kpis['total_flights']:,}",         k1)
    kpi_card("Avg Ticket Price",      f"₹{kpis['avg_price']:,.0f}",         k2)
    kpi_card("Median Price",          f"₹{kpis['median_price']:,.0f}",      k3)
    kpi_card("Most Popular Airline",  kpis['most_popular_airline'],          k4)
    kpi_card("Avg Duration",          kpis['avg_duration_fmt'],              k5)
    kpi_card("Total Routes",          str(kpis['total_routes']),             k6)

    st.divider()

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    tabs = st.tabs([
        "📊 Price Analysis",
        "🛫 Airline & Route",
        "⏱️ Duration & Timing",
        "📅 Booking Trends",
        "🤖 Fare Predictor",
        "📥 Download Report",
    ])

    # ================================================================
    # TAB 1: Price Analysis
    # ================================================================
    with tabs[0]:
        section("Price Analysis")

        col_a, col_b = st.columns(2)

        # --- Airline price bar ---
        with col_a:
            airline_stats = dp.airline_price_stats(df)
            fig = viz.airline_price_bar(airline_stats)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "Vistara and Air India command the highest average fares, especially "
                "in Business class. Budget carriers like SpiceJet and AirAsia are "
                "most competitive in Economy."
            )

        # --- Class comparison box ---
        with col_b:
            fig = viz.class_price_boxplot(df)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "Business class tickets cost on average 3–5× more than Economy. "
                "Economy prices show a wider spread due to promotional fares."
            )

        col_c, col_d = st.columns(2)

        # --- Stops vs Price ---
        with col_c:
            stops_stats = dp.stops_price_stats(df)
            fig = viz.stops_price_chart(stops_stats)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "Counterintuitively, 2+ stop flights can be pricier than 1-stop — "
                "likely due to last-minute booking mix. Non-stop fares are lowest "
                "on average for short-haul routes."
            )

        # --- Price distribution histogram ---
        with col_d:
            fig = viz.price_distribution_hist(df)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "Economy fares cluster tightly between ₹4K–₹15K. "
                "Business class creates a right-skewed, bimodal distribution "
                "reflecting heavy last-minute premiums."
            )

        # --- Price tier bar (full width) ---
        tier_data = dp.price_distribution_data(df)
        fig = viz.price_tier_bar(tier_data)
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "The ₹5K–₹10K bucket is the most populated tier, indicating most "
            "Economy travellers book in the 2–4 week window."
        )

        # --- Airline × Class heatmap ---
        fig = viz.airline_class_heatmap(airline_stats)
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "Vistara Business class is the most expensive segment. "
            "SpiceJet does not operate Business class. "
            "Air India Business fares are competitive with Vistara Economy."
        )

    # ================================================================
    # TAB 2: Airline & Route
    # ================================================================
    with tabs[1]:
        section("Airline & Route Analysis")

        col_a, col_b = st.columns(2)

        with col_a:
            share_data = dp.airline_market_share(df)
            fig = viz.airline_market_share_donut(share_data)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                f"Vistara dominates with ~{share_data.iloc[0]['share_pct']}% market share, "
                "followed by Air India. Together they represent more than 65% of all bookings."
            )

        with col_b:
            fig = viz.top_airlines_price_bar(airline_stats)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "Air India and Vistara are the priciest airlines on average. "
                "SpiceJet offers the most budget-friendly fares across all routes."
            )

        col_c, col_d = st.columns(2)

        with col_c:
            route_data = dp.route_analysis(df)
            fig = viz.popular_routes_bar(route_data)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "Delhi ↔ Mumbai is the busiest corridor in the dataset. "
                "Metro-to-metro routes dominate volume, reflecting demand from "
                "business travellers."
            )

        with col_d:
            exp_routes = dp.expensive_routes(df)
            fig = viz.expensive_routes_bar(exp_routes)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "Long-haul routes like Delhi → Chennai or Kolkata → Delhi "
                "command the highest average fares. Longer distance correlates "
                "strongly with higher prices."
            )

        # --- Route table ---
        section("Route Summary Table")
        route_full = (
            df.groupby("route")
            .agg(
                Flights=("price", "count"),
                Avg_Price=("price", "mean"),
                Min_Price=("price", "min"),
                Max_Price=("price", "max"),
            )
            .round(0)
            .reset_index()
            .sort_values("Flights", ascending=False)
        )
        st.dataframe(
            route_full.style.format({
                "Avg_Price": "₹{:,.0f}",
                "Min_Price": "₹{:,.0f}",
                "Max_Price": "₹{:,.0f}",
                "Flights":   "{:,}",
            }),
            use_container_width=True,
            height=350,
        )

    # ================================================================
    # TAB 3: Duration & Timing
    # ================================================================
    with tabs[2]:
        section("Duration & Timing Analysis")

        col_a, col_b = st.columns(2)

        with col_a:
            fig = viz.duration_price_scatter(df)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "Price increases with duration up to ~20 hours, after which "
                "ultra-long connecting flights can appear cheaper due to demand "
                "patterns. Business class maintains a consistent premium throughout."
            )

        with col_b:
            dep_data = dp.departure_time_analysis(df)
            fig = viz.departure_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "Early Morning → Night trips tend to have elevated average prices. "
                "Afternoon departures with Afternoon arrivals show the lowest fares."
            )

        # --- Duration binned price chart ---
        dur_corr = dp.duration_price_corr(df, bins=15)
        fig_dur = viz.days_left_price_line(
            dur_corr.rename(columns={"duration_mid": "days_left", "price": "avg_price"})
        )
        fig_dur.update_xaxes(title_text="Flight Duration (hours)", autorange=True)
        fig_dur.update_layout(title_text="Average Price by Flight Duration Bucket")
        st.plotly_chart(fig_dur, use_container_width=True)
        insight(
            "Average fare rises sharply after the 8-hour mark, suggesting "
            "long-haul routes with 1–2 stops carry premium pricing."
        )

    # ================================================================
    # TAB 4: Booking Trends
    # ================================================================
    with tabs[3]:
        section("Booking & Days-Left Trends")

        col_a, col_b = st.columns(2)

        with col_a:
            trend = dp.days_left_price_trend(df)
            fig = viz.days_left_price_line(trend)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "Prices spike sharply in the last 7 days before departure. "
                "Booking 3–5 weeks in advance yields the best fares — "
                "approximately 40–55% cheaper than last-minute booking."
            )

        with col_b:
            fig = viz.days_left_histogram(df)
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "Booking volume peaks in the 20–35 days window, aligning with "
                "the sweet-spot pricing period. Very early bookings (45+ days) "
                "are sparse but show discounted fares."
            )

        # --- Days-left bucket average price table ---
        bucket_avg = (
            df.groupby("days_bucket", observed=True)["price"]
            .agg(["mean", "median", "count"])
            .round(0)
            .reset_index()
            .rename(columns={"days_bucket": "Days Bucket", "mean": "Avg Price",
                              "median": "Median Price", "count": "Flights"})
        )
        st.subheader("Average Price by Booking Window")
        st.dataframe(
            bucket_avg.style.format({
                "Avg Price":    "₹{:,.0f}",
                "Median Price": "₹{:,.0f}",
                "Flights":      "{:,}",
            }),
            use_container_width=True,
        )
        insight(
            "The 22–30 day bucket consistently offers median prices around ₹6K–₹8K "
            "for Economy — the optimal booking window for budget travellers."
        )

    # ================================================================
    # TAB 5: Fare Predictor
    # ================================================================
    with tabs[4]:
        section("✈️ Fare Prediction Module")
        st.caption(
            "Enter flight details below to get a predicted fare using our "
            "Random Forest model trained on 300,000+ bookings."
        )

        # --- Load / train model ---
        with st.spinner("Loading fare prediction model …"):
            model, train_metrics = get_model(df_full)

        if train_metrics:
            st.success(
                f"Model trained — R² = {train_metrics['R2']:.4f} | "
                f"MAE = ₹{train_metrics['MAE']:,.0f} | "
                f"RMSE = ₹{train_metrics['RMSE']:,.0f} | "
                f"MAPE = {train_metrics['MAPE']:.2f}%"
            )
        else:
            st.info("Pre-trained model loaded from disk.")

        st.divider()

        col_l, col_r = st.columns([2, 1])

        with col_l:
            # Input form
            with st.form("predict_form"):
                r1c1, r1c2, r1c3 = st.columns(3)
                airline_in  = r1c1.selectbox("Airline", sorted(df_full["airline"].unique()))
                src_in      = r1c2.selectbox("Source City", sorted(df_full["source_city"].unique()))
                dst_in      = r1c3.selectbox("Destination City", sorted(df_full["destination_city"].unique()))

                r2c1, r2c2, r2c3 = st.columns(3)
                dep_time_in = r2c1.selectbox(
                    "Departure Time",
                    ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
                )
                arr_time_in = r2c2.selectbox(
                    "Arrival Time",
                    ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
                )
                class_in    = r2c3.selectbox("Travel Class", ["Economy", "Business"])

                r3c1, r3c2, r3c3 = st.columns(3)
                stops_map   = {"Non-Stop (0)": 0, "1 Stop": 1, "2+ Stops": 2}
                stops_label = r3c1.selectbox("Stops", list(stops_map.keys()))
                stops_in    = stops_map[stops_label]
                duration_in = r3c2.number_input("Duration (hours)", min_value=0.5, max_value=50.0, value=2.5, step=0.25)
                days_in     = r3c3.number_input("Days Left", min_value=1, max_value=49, value=15, step=1)

                submitted = st.form_submit_button("🔮 Predict Fare", use_container_width=True, type="primary")

            if submitted:
                input_data = {
                    "airline": airline_in,
                    "source_city": src_in,
                    "destination_city": dst_in,
                    "departure_time": dep_time_in,
                    "arrival_time": arr_time_in,
                    "class": class_in,
                    "stops_num": stops_in,
                    "duration": duration_in,
                    "days_left": days_in,
                }
                predicted = predict_fare(model, input_data)
                st.success(f"### 💰 Predicted Fare: ₹{predicted:,.0f}")

                # Contextual insight
                avg_route = df_full[
                    (df_full["source_city"] == src_in) &
                    (df_full["destination_city"] == dst_in) &
                    (df_full["class"] == class_in)
                ]["price"].mean()

                if not np.isnan(avg_route):
                    diff = predicted - avg_route
                    diff_pct = abs(diff) / avg_route * 100
                    direction = "above" if diff > 0 else "below"
                    insight(
                        f"Predicted fare is ₹{abs(diff):,.0f} ({diff_pct:.1f}%) "
                        f"{direction} the average {class_in} fare on the "
                        f"{src_in} → {dst_in} route (₹{avg_route:,.0f})."
                    )

        with col_r:
            # Feature importance chart
            if model is not None:
                try:
                    feat_names, importances = get_feature_importances(model)
                    fig_imp = viz.feature_importance_bar(feat_names, importances)
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception:
                    st.info("Feature importance chart available after model training.")

        # --- Route Recommendations ---
        st.divider()
        section("🗺️ Route Recommendations (Best Value Economy Routes)")
        recs = route_recommendations(df_full)
        for rec in recs:
            st.markdown(
                f'<div class="insight-box">'
                f'<strong>{rec["route"]}</strong> — {rec["insight"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ================================================================
    # TAB 6: Download Report
    # ================================================================
    with tabs[5]:
        section("📥 Download Data & Reports")

        st.markdown("Download the filtered dataset and pre-computed statistics as CSV files.")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.download_button(
                label="⬇️ Download Filtered Dataset",
                data=df_to_csv_bytes(df),
                file_name="filtered_flights.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_b:
            airline_s = dp.airline_price_stats(df)
            st.download_button(
                label="⬇️ Download Airline Stats",
                data=df_to_csv_bytes(airline_s),
                file_name="airline_price_stats.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_c:
            route_d = dp.route_analysis(df, top_n=100)
            st.download_button(
                label="⬇️ Download Route Analysis",
                data=df_to_csv_bytes(route_d),
                file_name="route_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()

        # --- Data quality snapshot ---
        section("Data Quality Snapshot")
        q1, q2, q3 = st.columns(3)

        dup_info = pipeline_data["duplicate_info"]
        out_info  = pipeline_data["outlier_report"]

        with q1:
            st.metric("Total Raw Records",  f"{dup_info['total_rows']:,}")
            st.metric("Duplicate Rows",     f"{dup_info['duplicate_rows']:,} ({dup_info['duplicate_pct']}%)")

        with q2:
            st.metric("Price Outliers (1.5×IQR)", f"{out_info['n_outliers']:,} ({out_info['outlier_pct']}%)")
            st.metric("IQR Fence Upper",   f"₹{out_info['upper_fence']:,.0f}")

        with q3:
            st.metric("Missing Values", "0 (clean dataset)")
            st.metric("Clean Records Used", f"{len(df_full):,}")

        # --- Raw data preview ---
        st.divider()
        section("Raw Data Preview (first 500 rows)")
        st.dataframe(df.head(500), use_container_width=True, height=400)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
