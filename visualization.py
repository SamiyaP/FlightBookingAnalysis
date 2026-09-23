"""
visualization.py
================
Flight Booking Analysis – Reusable Plotly Chart Factory.

Every public function accepts a pre-processed DataFrame (or aggregated
stats DataFrame from data_processing.py) and returns a plotly.graph_objects.Figure.
All figures use a consistent colour palette and theme so the dashboard
looks uniform.

Usage:
    from visualization import *
    fig = airline_price_bar(airline_stats_df)
    fig.show()
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Theme / Palette
# ---------------------------------------------------------------------------

PALETTE = px.colors.qualitative.Set2          # up to 8 distinct colours
BLUE    = "#3B82F6"
ORANGE  = "#F97316"
GREEN   = "#22C55E"
PURPLE  = "#8B5CF6"
RED     = "#EF4444"
TEAL    = "#14B8A6"
YELLOW  = "#EAB308"
PINK    = "#EC4899"

CLASS_COLORS = {"Economy": BLUE, "Business": ORANGE}

PLOTLY_TEMPLATE = "plotly_white"

LAYOUT_DEFAULTS = dict(
    template=PLOTLY_TEMPLATE,
    font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#1f2937"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def _apply_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply standard layout settings to any figure."""
    fig.update_layout(title_text=title, title_font_size=15, **LAYOUT_DEFAULTS)
    return fig


# ---------------------------------------------------------------------------
# 1. Airline-wise Price Analysis (grouped bar)
# ---------------------------------------------------------------------------

def airline_price_bar(df_stats: pd.DataFrame) -> go.Figure:
    """
    Grouped bar: average price per airline, split by Economy / Business class.

    Parameters
    ----------
    df_stats : output of data_processing.airline_price_stats()
    """
    fig = px.bar(
        df_stats,
        x="airline",
        y="avg_price",
        color="class",
        barmode="group",
        color_discrete_map=CLASS_COLORS,
        labels={"avg_price": "Average Price (INR)", "airline": "Airline", "class": "Class"},
        text="avg_price",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickprefix="₹", tickformat=",")
    return _apply_defaults(fig, "Airline-wise Average Ticket Price by Class")


# ---------------------------------------------------------------------------
# 2. Class-wise Price Distribution (box plot)
# ---------------------------------------------------------------------------

def class_price_boxplot(df: pd.DataFrame) -> go.Figure:
    """Box-plot of price distribution for Economy vs Business."""
    fig = px.box(
        df,
        x="class",
        y="price",
        color="class",
        color_discrete_map=CLASS_COLORS,
        points=False,
        labels={"price": "Ticket Price (INR)", "class": "Travel Class"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(tickprefix="₹", tickformat=",")
    return _apply_defaults(fig, "Price Distribution: Economy vs Business Class")


# ---------------------------------------------------------------------------
# 3. Stops vs Price (bar + line overlay)
# ---------------------------------------------------------------------------

def stops_price_chart(df_stops: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing average price per stop category with median overlay.

    Parameters
    ----------
    df_stops : output of data_processing.stops_price_stats()
    """
    stops_label_map = {"zero": "Non-Stop", "one": "1 Stop", "two_or_more": "2+ Stops"}
    df_stops = df_stops.copy()
    df_stops["stops_label"] = df_stops["stops"].map(stops_label_map).fillna(df_stops["stops"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_stops["stops_label"],
        y=df_stops["avg_price"],
        name="Avg Price",
        marker_color=BLUE,
        text=df_stops["avg_price"],
        texttemplate="₹%{text:,.0f}",
        textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=df_stops["stops_label"],
        y=df_stops["median_price"],
        mode="lines+markers",
        name="Median Price",
        line=dict(color=ORANGE, width=2.5),
        marker=dict(size=9),
    ))
    fig.update_yaxes(tickprefix="₹", tickformat=",", title_text="Price (INR)")
    fig.update_xaxes(title_text="Number of Stops")
    return _apply_defaults(fig, "Impact of Number of Stops on Ticket Price")


# ---------------------------------------------------------------------------
# 4. Duration vs Price (scatter with trendline)
# ---------------------------------------------------------------------------

def duration_price_scatter(df: pd.DataFrame, sample: int = 8000) -> go.Figure:
    """
    Scatter plot of flight duration vs price, coloured by class.
    Samples the dataframe to keep rendering fast.
    """
    df_s = df.sample(min(sample, len(df)), random_state=42)
    fig = px.scatter(
        df_s,
        x="duration",
        y="price",
        color="class",
        color_discrete_map=CLASS_COLORS,
        opacity=0.4,
        trendline="lowess",
        labels={"duration": "Flight Duration (hours)", "price": "Ticket Price (INR)", "class": "Class"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(tickprefix="₹", tickformat=",")
    return _apply_defaults(fig, "Flight Duration vs Ticket Price")


# ---------------------------------------------------------------------------
# 5. Days Left vs Price (line chart)
# ---------------------------------------------------------------------------

def days_left_price_line(df_trend: pd.DataFrame) -> go.Figure:
    """
    Line chart of average price vs days left before departure.

    Parameters
    ----------
    df_trend : output of data_processing.days_left_price_trend()
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_trend["days_left"],
        y=df_trend["avg_price"],
        mode="lines",
        fill="tozeroy",
        line=dict(color=TEAL, width=2.5),
        fillcolor="rgba(20,184,166,0.12)",
        name="Avg Price",
    ))
    fig.update_xaxes(title_text="Days Left Before Departure", autorange="reversed")
    fig.update_yaxes(tickprefix="₹", tickformat=",", title_text="Average Price (INR)")
    return _apply_defaults(fig, "Days Left Before Departure vs Average Ticket Price")


# ---------------------------------------------------------------------------
# 6. Top Popular Routes (horizontal bar)
# ---------------------------------------------------------------------------

def popular_routes_bar(df_routes: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of top routes by flight volume.

    Parameters
    ----------
    df_routes : output of data_processing.route_analysis()
    """
    fig = px.bar(
        df_routes.sort_values("flight_count"),
        x="flight_count",
        y="route",
        orientation="h",
        color="avg_price",
        color_continuous_scale="Blues",
        labels={"flight_count": "Number of Flights", "route": "Route", "avg_price": "Avg Price (INR)"},
        text="flight_count",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_coloraxes(colorbar_tickprefix="₹")
    return _apply_defaults(fig, "Most Popular Flight Routes (by Volume)")


# ---------------------------------------------------------------------------
# 7. Most Expensive Routes (bar)
# ---------------------------------------------------------------------------

def expensive_routes_bar(df_exp: pd.DataFrame) -> go.Figure:
    """
    Bar chart of top 10 most expensive routes by average price.

    Parameters
    ----------
    df_exp : output of data_processing.expensive_routes()
    """
    fig = px.bar(
        df_exp.sort_values("avg_price"),
        x="avg_price",
        y="route",
        orientation="h",
        color="avg_price",
        color_continuous_scale="Reds",
        labels={"avg_price": "Average Price (INR)", "route": "Route"},
        text="avg_price",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_coloraxes(colorbar_tickprefix="₹")
    return _apply_defaults(fig, "Most Expensive Flight Routes (Average Price)")


# ---------------------------------------------------------------------------
# 8. Airline Market Share (donut)
# ---------------------------------------------------------------------------

def airline_market_share_donut(df_share: pd.DataFrame) -> go.Figure:
    """
    Donut chart showing each airline's share of total flights.

    Parameters
    ----------
    df_share : output of data_processing.airline_market_share()
    """
    fig = px.pie(
        df_share,
        names="airline",
        values="flight_count",
        hole=0.45,
        color_discrete_sequence=PALETTE,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textposition="outside", textinfo="label+percent")
    return _apply_defaults(fig, "Airline Market Share (by Flight Volume)")


# ---------------------------------------------------------------------------
# 9. Price Distribution Histogram
# ---------------------------------------------------------------------------

def price_distribution_hist(df: pd.DataFrame) -> go.Figure:
    """Histogram of ticket prices, split by class."""
    fig = px.histogram(
        df,
        x="price",
        color="class",
        nbins=60,
        color_discrete_map=CLASS_COLORS,
        barmode="overlay",
        opacity=0.7,
        labels={"price": "Ticket Price (INR)", "class": "Class"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_xaxes(tickprefix="₹", tickformat=",")
    fig.update_yaxes(title_text="Number of Flights")
    return _apply_defaults(fig, "Ticket Price Distribution")


# ---------------------------------------------------------------------------
# 10. Price Tier Breakdown (bar)
# ---------------------------------------------------------------------------

def price_tier_bar(df_tier: pd.DataFrame) -> go.Figure:
    """
    Bar chart of flights per price tier bucket.

    Parameters
    ----------
    df_tier : output of data_processing.price_distribution_data()
    """
    fig = px.bar(
        df_tier,
        x="price_tier",
        y="count",
        color="price_tier",
        color_discrete_sequence=PALETTE,
        labels={"price_tier": "Price Tier (INR)", "count": "Number of Flights"},
        text="count",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(showlegend=False)
    return _apply_defaults(fig, "Flight Booking Count by Price Tier")


# ---------------------------------------------------------------------------
# 11. Departure Time vs Price (heatmap)
# ---------------------------------------------------------------------------

def departure_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap: departure time vs arrival time, cell = mean price.
    """
    pivot = (
        df.groupby(["departure_time", "arrival_time"])["price"]
        .mean()
        .round(0)
        .unstack(fill_value=0)
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdBu_r",
        labels=dict(x="Arrival Time", y="Departure Time", color="Avg Price (INR)"),
        aspect="auto",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_coloraxes(colorbar_tickprefix="₹")
    return _apply_defaults(fig, "Average Price by Departure × Arrival Time Slot")


# ---------------------------------------------------------------------------
# 12. Booking Trends: Days-left Histogram
# ---------------------------------------------------------------------------

def days_left_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram showing booking density across days-left buckets."""
    fig = px.histogram(
        df,
        x="days_left",
        nbins=49,
        color_discrete_sequence=[PURPLE],
        labels={"days_left": "Days Left Before Departure", "count": "Number of Bookings"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(title_text="Number of Bookings")
    return _apply_defaults(fig, "Booking Distribution by Days Left Before Departure")


# ---------------------------------------------------------------------------
# 13. Airline vs Class Heatmap (avg price)
# ---------------------------------------------------------------------------

def airline_class_heatmap(df_stats: pd.DataFrame) -> go.Figure:
    """
    Heatmap: airline vs class, cell = average price.

    Parameters
    ----------
    df_stats : output of data_processing.airline_price_stats()
    """
    pivot = df_stats.pivot(index="airline", columns="class", values="avg_price").fillna(0)
    fig = px.imshow(
        pivot,
        color_continuous_scale="Viridis",
        labels=dict(x="Class", y="Airline", color="Avg Price (INR)"),
        text_auto=",.0f",
        aspect="auto",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_coloraxes(colorbar_tickprefix="₹")
    return _apply_defaults(fig, "Average Ticket Price: Airline × Class Heatmap")


# ---------------------------------------------------------------------------
# 14. Top Airlines by Average Price (horizontal bar)
# ---------------------------------------------------------------------------

def top_airlines_price_bar(df_stats: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar: airlines ranked by overall average price.

    Parameters
    ----------
    df_stats : output of data_processing.airline_price_stats()
    """
    overall = (
        df_stats.groupby("airline")["avg_price"]
        .mean()
        .round(0)
        .reset_index()
        .sort_values("avg_price")
    )
    fig = px.bar(
        overall,
        x="avg_price",
        y="airline",
        orientation="h",
        color="avg_price",
        color_continuous_scale="Teal",
        labels={"avg_price": "Average Price (INR)", "airline": "Airline"},
        text="avg_price",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_coloraxes(colorbar_tickprefix="₹")
    return _apply_defaults(fig, "Airlines Ranked by Average Ticket Price")


# ---------------------------------------------------------------------------
# 15. Fare Prediction Feature Importance (bar)
# ---------------------------------------------------------------------------

def feature_importance_bar(feature_names: list, importances: list) -> go.Figure:
    """
    Horizontal bar chart of feature importances from the ML model.

    Parameters
    ----------
    feature_names : list of str
    importances   : list of float (same length as feature_names)
    """
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance").tail(20)   # show top-20 features

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Oranges",
        labels={"importance": "Feature Importance", "feature": "Feature"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(showlegend=False)
    return _apply_defaults(fig, "Top 20 Feature Importances (Fare Prediction Model)")
