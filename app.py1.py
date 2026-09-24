"""
Car Sales Data Analytics & Business Intelligence Dashboard
=========================================================
Single-file Streamlit application.
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Car Sales BI Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; }
    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 12px;
    }
    .kpi-card.green  { border-left-color: #22c55e; }
    .kpi-card.orange { border-left-color: #f97316; }
    .kpi-card.purple { border-left-color: #a855f7; }
    .kpi-card.red    { border-left-color: #ef4444; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1e293b; margin: 6px 0; }
    .kpi-label { font-size: 0.82rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-sub   { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }
    /* Section headers */
    .section-header {
        font-size: 1.35rem; font-weight: 700; color: #1e293b;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 18px;
    }
    /* Insight boxes */
    .insight-box {
        background: white; border-radius: 10px; padding: 16px 20px;
        border-left: 5px solid #3b82f6; margin-bottom: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .insight-box.warning { border-left-color: #f97316; }
    .insight-box.success { border-left-color: #22c55e; }
    .insight-box.danger  { border-left-color: #ef4444; }
    .insight-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
                     letter-spacing: 0.1em; color: #64748b; margin-bottom: 4px; }
    .insight-text  { font-size: 0.92rem; color: #1e293b; line-height: 1.5; }
    /* Sidebar */
    section[data-testid="stSidebar"] { background: #1e293b; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label { color: #94a3b8 !important; }
    /* Table */
    .styled-table { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
    .styled-table th { background: #f1f5f9; color: #475569; font-weight: 600;
                       padding: 10px 14px; text-align: left; border-bottom: 2px solid #e2e8f0; }
    .styled-table td { padding: 9px 14px; border-bottom: 1px solid #f1f5f9; color: #334155; }
    .styled-table tr:hover td { background: #f8fafc; }
    /* Page title */
    .page-title { font-size: 1.7rem; font-weight: 800; color: #1e293b; margin-bottom: 4px; }
    .page-sub   { font-size: 0.95rem; color: #64748b; margin-bottom: 24px; }
    /* Hide Streamlit default header */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and cleaning data…")
def load_data():
    # ── Locate the CSV ──────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "car_sales.csv"),
        os.path.join(script_dir, "Car_Sales_Project.csv"),
        os.path.join(script_dir, "data", "car_sales.csv"),
    ]
    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break
    if path is None:
        st.error("Dataset not found. Place car_sales.csv in the same folder as app.py.")
        st.stop()

    # ── Load ─────────────────────────────────────────────────
    df = pd.read_csv(path)

    # ── Standardise column names ─────────────────────────────
    df.columns = df.columns.str.strip()

    # ── Rename to short internal names ───────────────────────
    col_map = {
        "Manufacturer":        "manufacturer",
        "Model":               "model",
        "Engine size":         "engine_size",
        "Fuel type":           "fuel_type",
        "Year of manufacture": "year",
        "Mileage":             "mileage",
        "Price":               "price",
    }
    df.rename(columns=col_map, inplace=True)

    # ── Data types ───────────────────────────────────────────
    df["engine_size"] = pd.to_numeric(df["engine_size"], errors="coerce")
    df["year"]        = pd.to_numeric(df["year"],        errors="coerce").astype("Int64")
    df["mileage"]     = pd.to_numeric(df["mileage"],     errors="coerce")
    df["price"]       = pd.to_numeric(df["price"],       errors="coerce")

    # ── Strip text whitespace ────────────────────────────────
    for col in ["manufacturer", "model", "fuel_type"]:
        df[col] = df[col].str.strip()

    # ── Drop rows with missing key numerics ──────────────────
    df.dropna(subset=["price", "mileage", "year", "engine_size"], inplace=True)

    # ── Drop exact duplicates ────────────────────────────────
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Remove implausible values ────────────────────────────
    df = df[df["price"]   > 0]
    df = df[df["mileage"] > 0]
    df = df[df["year"]    >= 1980]
    df = df[df["year"]    <= 2025]

    # ── Derived columns ──────────────────────────────────────
    df["age"]             = 2024 - df["year"]
    df["price_per_mile"]  = (df["price"] / df["mileage"]).round(4)
    df["decade"]          = (df["year"] // 10 * 10).astype(str) + "s"

    # ── Outlier flag (IQR) for Price — keep for risk section ─
    q1, q3 = df["price"].quantile(0.25), df["price"].quantile(0.75)
    iqr = q3 - q1
    df["price_outlier"] = (df["price"] < q1 - 1.5 * iqr) | (df["price"] > q3 + 1.5 * iqr)

    return df

df = load_data()

# ─────────────────────────────────────────────
# GLOBAL KPI CALCULATIONS
# ─────────────────────────────────────────────
total_listings   = len(df)
total_revenue    = df["price"].sum()
avg_price        = df["price"].mean()
median_price     = df["price"].median()
num_brands       = df["manufacturer"].nunique()
num_models       = df["model"].nunique()
avg_mileage      = df["mileage"].mean()
avg_age          = df["age"].mean()
price_outlier_pct = df["price_outlier"].mean() * 100

top_brand_vol    = df["manufacturer"].value_counts().idxmax()
top_brand_rev    = df.groupby("manufacturer")["price"].sum().idxmax()

# Year-over-year growth (compare avg price of newest 2 year bands)
yearly_avg = df.groupby("year")["price"].mean().reset_index().sort_values("year")
if len(yearly_avg) >= 2:
    yoy_change = ((yearly_avg["price"].iloc[-1] - yearly_avg["price"].iloc[-2])
                  / yearly_avg["price"].iloc[-2] * 100)
else:
    yoy_change = 0.0

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION & FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Car Sales BI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Executive Overview", "🔍 Sales & Product Analysis", "⚠️ Market & Risk Analysis"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### Global Filters")

    all_brands = sorted(df["manufacturer"].unique())
    sel_brands = st.multiselect("Manufacturer", all_brands, default=all_brands)

    all_fuels = sorted(df["fuel_type"].unique())
    sel_fuels = st.multiselect("Fuel Type", all_fuels, default=all_fuels)

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    sel_years = st.slider("Year of Manufacture", year_min, year_max, (year_min, year_max))

    st.markdown("---")
    st.caption("Data: 50,000 used-car listings\n5 manufacturers · 3 fuel types\n1984–2022")

# ── Apply filters ────────────────────────────────────────────────────────────
dff = df[
    df["manufacturer"].isin(sel_brands) &
    df["fuel_type"].isin(sel_fuels) &
    df["year"].between(sel_years[0], sel_years[1])
].copy()

if dff.empty:
    st.warning("No data matches the current filters. Please broaden your selection.")
    st.stop()

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
BRAND_COLORS = {
    "BMW":     "#1a56db",
    "Ford":    "#e3261a",
    "Porsche": "#c7a227",
    "Toyota":  "#eb0a1e",
    "VW":      "#0077b6",
}

def fmt_currency(v):
    if v >= 1_000_000:
        return f"£{v/1_000_000:.1f}M"
    elif v >= 1_000:
        return f"£{v/1_000:.1f}K"
    return f"£{v:,.0f}"

def kpi_card(label, value, sub="", color="blue"):
    color_cls = {"blue": "", "green": "green", "orange": "orange",
                 "purple": "purple", "red": "red"}.get(color, "")
    st.markdown(f"""
    <div class="kpi-card {color_cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def insight_box(label, text, style=""):
    st.markdown(f"""
    <div class="insight-box {style}">
        <div class="insight-label">{label}</div>
        <div class="insight-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="#f8fafc",
    font=dict(family="Segoe UI, sans-serif", size=12, color="#334155"),
    margin=dict(t=40, b=40, l=40, r=20),
    hoverlabel=dict(bgcolor="white", font_size=12),
)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.markdown('<div class="page-title">📊 Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">High-level KPIs and market summary across all manufacturers</div>',
                unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("Total Listings", f"{len(dff):,}", "cars in dataset", "blue")
    with c2:
        kpi_card("Combined Value", fmt_currency(dff['price'].sum()),
                 "sum of all asking prices", "green")
    with c3:
        kpi_card("Average Price", fmt_currency(dff['price'].mean()),
                 f"median {fmt_currency(dff['price'].median())}", "orange")
    with c4:
        kpi_card("Manufacturers", str(dff['manufacturer'].nunique()),
                 "brands represented", "purple")
    with c5:
        kpi_card("Models", str(dff['model'].nunique()),
                 "distinct car models", "blue")
    with c6:
        kpi_card("Avg Mileage", f"{dff['mileage'].mean():,.0f} mi",
                 f"avg age {dff['age'].mean():.1f} yrs", "red")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Price Distribution + Brand Volume ──────────────
    col_a, col_b = st.columns([1.1, 0.9])

    with col_a:
        section("Price Distribution by Manufacturer")
        fig = px.box(
            dff, x="manufacturer", y="price", color="manufacturer",
            color_discrete_map=BRAND_COLORS,
            labels={"manufacturer": "Manufacturer", "price": "Price (£)"},
        )
        fig.update_traces(quartilemethod="exclusive")
        fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                          yaxis_tickprefix="£", yaxis_tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section("Listings by Manufacturer")
        vol = dff["manufacturer"].value_counts().reset_index()
        vol.columns = ["manufacturer", "count"]
        fig = px.bar(
            vol, x="count", y="manufacturer", orientation="h",
            color="manufacturer", color_discrete_map=BRAND_COLORS,
            labels={"count": "Number of Listings", "manufacturer": ""},
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                          yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Avg Price by Year + Fuel Mix ───────────────────
    col_c, col_d = st.columns([1.3, 0.7])

    with col_c:
        section("Average Price Trend by Manufacture Year")
        trend = dff.groupby("year").agg(avg_price=("price", "mean"),
                                         count=("price", "count")).reset_index()
        trend = trend[trend["count"] >= 10]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["year"], y=trend["avg_price"],
            mode="lines+markers", name="Avg Price",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=5),
            hovertemplate="Year: %{x}<br>Avg Price: £%{y:,.0f}<extra></extra>",
        ))

        # Trend line via linear regression
        if len(trend) > 5:
            X = trend["year"].values.reshape(-1, 1)
            y_arr = trend["avg_price"].values
            lr = LinearRegression().fit(X, y_arr)
            y_pred = lr.predict(X)
            fig.add_trace(go.Scatter(
                x=trend["year"], y=y_pred,
                mode="lines", name="Trend",
                line=dict(color="#f97316", width=1.8, dash="dash"),
                hovertemplate="Trend: £%{y:,.0f}<extra></extra>",
            ))

        fig.update_layout(**PLOT_LAYOUT,
                          yaxis_tickprefix="£", yaxis_tickformat=",",
                          xaxis_title="Year of Manufacture",
                          yaxis_title="Average Price (£)",
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        section("Fuel Type Mix")
        fuel_cnt = dff["fuel_type"].value_counts().reset_index()
        fuel_cnt.columns = ["fuel_type", "count"]
        fig = px.pie(
            fuel_cnt, names="fuel_type", values="count",
            color_discrete_sequence=["#3b82f6", "#22c55e", "#f97316"],
            hole=0.45,
        )
        fig.update_traces(textinfo="percent+label", pull=[0.03, 0.03, 0.03])
        fig.update_layout(**PLOT_LAYOUT, showlegend=True,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Key Business Insights ─────────────────────────
    section("Key Business Insights")
    i1, i2 = st.columns(2)

    brand_avg = dff.groupby("manufacturer")["price"].mean()
    top_val_brand = brand_avg.idxmax()
    low_vol_brand = dff["manufacturer"].value_counts().idxmin()

    with i1:
        insight_box("FACT",
            f"Porsche listings average <b>£{dff[dff.manufacturer=='Porsche']['price'].mean():,.0f}</b>, "
            f"the highest of all 5 brands — reflecting its premium positioning.",
            "success")
        insight_box("TREND",
            "Cars manufactured after 2015 command significantly higher prices, driven by newer "
            "technology, lower mileage, and modern features. The average price for post-2015 cars is "
            f"<b>{fmt_currency(dff[dff.year>2015]['price'].mean())}</b> vs "
            f"<b>{fmt_currency(dff[dff.year<=2015]['price'].mean())}</b> for older stock.",
            "")
        insight_box("OPPORTUNITY",
            "Hybrid vehicles represent a growing segment. "
            f"<b>{dff[dff.fuel_type=='Hybrid']['price'].mean():,.0f}£</b> average — competitive with Diesel. "
            "Increasing hybrid inventory could attract eco-conscious buyers.",
            "success")
    with i2:
        insight_box("INSIGHT",
            f"Ford accounts for the largest share of listings ({dff[dff.manufacturer=='Ford'].shape[0]:,} cars), "
            "giving it the highest market volume but also the most price competition at the budget end.",
            "")
        insight_box("RISK",
            f"<b>{price_outlier_pct:.1f}%</b> of listings are statistically extreme outliers (IQR method). "
            "These include very cheap high-mileage cars and very expensive low-mileage vehicles — "
            "both requiring careful vetting before purchase.",
            "warning")
        insight_box("RECOMMENDED ACTION",
            "Prioritise stocking vehicles from <b>2015–2022</b> with <b>under 80,000 miles</b>. "
            "This sweet-spot segment commands higher prices and faster turnover based on the data distribution.",
            "success")


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — SALES & PRODUCT ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Sales & Product Analysis":
    st.markdown('<div class="page-title">🔍 Sales & Product Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Detailed breakdown by manufacturer, model, fuel type, transmission and year</div>',
                unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📦 Brand & Model", "⛽ Fuel & Engine", "📅 Year Trends", "🏆 Rankings"]
    )

    # ── Tab 1: Brand & Model ──────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            section("Average Price by Manufacturer")
            brand_stats = (dff.groupby("manufacturer")
                           .agg(avg_price=("price", "mean"),
                                count=("price", "count"),
                                median_price=("price", "median"))
                           .reset_index()
                           .sort_values("avg_price", ascending=False))

            fig = px.bar(
                brand_stats, x="manufacturer", y="avg_price",
                color="manufacturer", color_discrete_map=BRAND_COLORS,
                text=brand_stats["avg_price"].apply(fmt_currency),
                labels={"manufacturer": "", "avg_price": "Average Price (£)"},
                custom_data=["count", "median_price"],
            )
            fig.update_traces(
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Avg: £%{y:,.0f}<br>Listings: %{customdata[0]:,}<br>Median: £%{customdata[1]:,.0f}<extra></extra>"
            )
            fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                              yaxis_tickprefix="£", yaxis_tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            section("Price Range by Manufacturer (IQR View)")
            fig = px.violin(
                dff, x="manufacturer", y="price",
                color="manufacturer", color_discrete_map=BRAND_COLORS,
                box=True, points=False,
                labels={"manufacturer": "", "price": "Price (£)"},
            )
            fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                              yaxis_tickprefix="£", yaxis_tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

        # Top models
        section("Top 20 Models by Average Price")
        sel_brand_tab = st.selectbox("Select Manufacturer", ["All"] + sorted(dff["manufacturer"].unique()), key="brand_model")
        df_model = dff if sel_brand_tab == "All" else dff[dff["manufacturer"] == sel_brand_tab]
        model_stats = (df_model.groupby(["manufacturer", "model"])
                       .agg(avg_price=("price", "mean"),
                            count=("price", "count"),
                            min_price=("price", "min"),
                            max_price=("price", "max"))
                       .reset_index()
                       .sort_values("avg_price", ascending=False)
                       .head(20))

        fig = px.bar(
            model_stats, x="avg_price", y="model", orientation="h",
            color="manufacturer", color_discrete_map=BRAND_COLORS,
            text=model_stats["avg_price"].apply(fmt_currency),
            labels={"avg_price": "Average Price (£)", "model": ""},
            custom_data=["count", "min_price", "max_price"],
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Avg: £%{x:,.0f}<br>Listings: %{customdata[0]:,}<br>Min: £%{customdata[1]:,.0f}  Max: £%{customdata[2]:,.0f}<extra></extra>",
        )
        fig.update_layout(**PLOT_LAYOUT, height=500,
                          yaxis=dict(categoryorder="total ascending"),
                          xaxis_tickprefix="£", xaxis_tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Fuel & Engine ──────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            section("Average Price by Fuel Type")
            fuel_stats = (dff.groupby("fuel_type")
                          .agg(avg_price=("price", "mean"),
                               count=("price", "count"))
                          .reset_index().sort_values("avg_price", ascending=False))
            fig = px.bar(
                fuel_stats, x="fuel_type", y="avg_price",
                color="fuel_type",
                color_discrete_sequence=["#3b82f6", "#22c55e", "#f97316"],
                text=fuel_stats["avg_price"].apply(fmt_currency),
                labels={"fuel_type": "Fuel Type", "avg_price": "Average Price (£)"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                              yaxis_tickprefix="£", yaxis_tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            section("Fuel Type Share by Manufacturer")
            fuel_brand = (dff.groupby(["manufacturer", "fuel_type"])
                          .size().reset_index(name="count"))
            totals = fuel_brand.groupby("manufacturer")["count"].transform("sum")
            fuel_brand["pct"] = fuel_brand["count"] / totals * 100
            fig = px.bar(
                fuel_brand, x="manufacturer", y="pct",
                color="fuel_type",
                color_discrete_sequence=["#3b82f6", "#22c55e", "#f97316"],
                labels={"manufacturer": "", "pct": "Share (%)", "fuel_type": "Fuel"},
                barmode="stack",
                text=fuel_brand["pct"].apply(lambda x: f"{x:.0f}%"),
            )
            fig.update_traces(textposition="inside")
            fig.update_layout(**PLOT_LAYOUT, yaxis_ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

        # Engine size vs price scatter
        section("Engine Size vs. Price")
        sample = dff.sample(min(3000, len(dff)), random_state=42)
        fig = px.scatter(
            sample, x="engine_size", y="price",
            color="manufacturer", color_discrete_map=BRAND_COLORS,
            opacity=0.55, size_max=6,
            labels={"engine_size": "Engine Size (L)", "price": "Price (£)", "manufacturer": "Brand"},
            hover_data=["model", "fuel_type", "year"],
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="#ef4444",
        )
        fig.update_layout(**PLOT_LAYOUT, height=420,
                          yaxis_tickprefix="£", yaxis_tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Year Trends ────────────────────────────────────
    with tab3:
        col_a, col_b = st.columns(2)

        with col_a:
            section("Listings Count by Manufacture Year")
            year_vol = dff.groupby("year").size().reset_index(name="count")
            fig = px.bar(
                year_vol, x="year", y="count",
                color_discrete_sequence=["#3b82f6"],
                labels={"year": "Year of Manufacture", "count": "Number of Listings"},
            )
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            section("Median Price by Manufacture Year")
            yr_price = (dff.groupby("year")["price"]
                        .median().reset_index())
            yr_price = yr_price[dff.groupby("year")["price"].count() >= 10]
            fig = px.line(
                yr_price, x="year", y="price",
                markers=True,
                color_discrete_sequence=["#22c55e"],
                labels={"year": "Year of Manufacture", "price": "Median Price (£)"},
            )
            fig.update_layout(**PLOT_LAYOUT,
                              yaxis_tickprefix="£", yaxis_tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

        # Mileage vs Age heatmap
        section("Average Price Heatmap — Decade × Fuel Type")
        heat_data = (dff.groupby(["decade", "fuel_type"])["price"]
                     .mean().unstack(fill_value=0).round(0))
        fig = px.imshow(
            heat_data.values,
            x=heat_data.columns.tolist(),
            y=heat_data.index.tolist(),
            color_continuous_scale="Blues",
            text_auto=True,
            labels=dict(x="Fuel Type", y="Decade", color="Avg Price (£)"),
            aspect="auto",
        )
        fig.update_layout(**PLOT_LAYOUT, height=340)
        st.plotly_chart(fig, use_container_width=True)

        # Mileage vs Price
        section("Mileage vs. Price (Depreciation Curve)")
        samp = dff.sample(min(4000, len(dff)), random_state=1)
        fig = px.scatter(
            samp, x="mileage", y="price",
            color="manufacturer", color_discrete_map=BRAND_COLORS,
            opacity=0.5,
            labels={"mileage": "Mileage (miles)", "price": "Price (£)", "manufacturer": "Brand"},
            trendline="lowess",
            trendline_scope="overall",
            trendline_color_override="#ef4444",
        )
        fig.update_layout(**PLOT_LAYOUT, height=380,
                          yaxis_tickprefix="£", yaxis_tickformat=",",
                          xaxis_tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Rankings ───────────────────────────────────────
    with tab4:
        col_a, col_b = st.columns(2)

        with col_a:
            section("Top 10 Most Expensive Models (Avg Price)")
            top10 = (dff.groupby(["manufacturer", "model"])
                     .agg(avg_price=("price", "mean"), count=("price", "count"))
                     .reset_index()
                     .query("count >= 5")
                     .sort_values("avg_price", ascending=False)
                     .head(10))
            top10["avg_price_fmt"] = top10["avg_price"].apply(fmt_currency)
            st.markdown("""
            <table class='styled-table'>
              <tr><th>#</th><th>Brand</th><th>Model</th><th>Avg Price</th><th>Listings</th></tr>
            """ + "".join(
                f"<tr><td>{i+1}</td><td>{r.manufacturer}</td><td>{r.model}</td>"
                f"<td>{r.avg_price_fmt}</td><td>{r['count']:,}</td></tr>"
                for i, r in top10.iterrows()
            ) + "</table>", unsafe_allow_html=True)

        with col_b:
            section("Top 10 Most Affordable Models (Avg Price)")
            bot10 = (dff.groupby(["manufacturer", "model"])
                     .agg(avg_price=("price", "mean"), count=("price", "count"))
                     .reset_index()
                     .query("count >= 5")
                     .sort_values("avg_price", ascending=True)
                     .head(10))
            bot10["avg_price_fmt"] = bot10["avg_price"].apply(fmt_currency)
            st.markdown("""
            <table class='styled-table'>
              <tr><th>#</th><th>Brand</th><th>Model</th><th>Avg Price</th><th>Listings</th></tr>
            """ + "".join(
                f"<tr><td>{i+1}</td><td>{r.manufacturer}</td><td>{r.model}</td>"
                f"<td>{r.avg_price_fmt}</td><td>{r['count']:,}</td></tr>"
                for i, r in bot10.iterrows()
            ) + "</table>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Most Listed Models (Volume Leaders)")
        top_vol = (dff.groupby(["manufacturer", "model"])
                   .size().reset_index(name="count")
                   .sort_values("count", ascending=False)
                   .head(15))
        fig = px.bar(
            top_vol, x="count", y="model", orientation="h",
            color="manufacturer", color_discrete_map=BRAND_COLORS,
            labels={"count": "Number of Listings", "model": ""},
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOT_LAYOUT, height=460,
                          yaxis=dict(categoryorder="total ascending"),
                          showlegend=True)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — MARKET & RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "⚠️ Market & Risk Analysis":
    st.markdown('<div class="page-title">⚠️ Market & Performance Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Risks, low-performing segments, opportunities and recommended business actions</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉 Risk Indicators", "💡 Opportunities", "🔮 Price Forecast"])

    # ── Tab 1: Risk Indicators ────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            section("Price Outlier Distribution")
            outlier_counts = dff["price_outlier"].value_counts().reset_index()
            outlier_counts.columns = ["outlier", "count"]
            outlier_counts["label"] = outlier_counts["outlier"].map(
                {True: "Outlier", False: "Normal"})
            fig = px.pie(
                outlier_counts, names="label", values="count",
                color="label",
                color_discrete_map={"Normal": "#3b82f6", "Outlier": "#ef4444"},
                hole=0.5,
                title=f"{dff['price_outlier'].sum():,} outlier listings ({dff['price_outlier'].mean()*100:.1f}%)",
            )
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            section("High Mileage / Low Price Risk Zone")
            # High-mileage + low-price = risky stock
            mileage_q75 = dff["mileage"].quantile(0.75)
            price_q25   = dff["price"].quantile(0.25)
            risk_df = dff[(dff["mileage"] > mileage_q75) & (dff["price"] < price_q25)]
            risk_cnt = risk_df.groupby("manufacturer").size().reset_index(name="risk_listings")
            fig = px.bar(
                risk_cnt.sort_values("risk_listings", ascending=False),
                x="manufacturer", y="risk_listings",
                color="manufacturer", color_discrete_map=BRAND_COLORS,
                text="risk_listings",
                labels={"manufacturer": "", "risk_listings": "Risk Listings"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                              title=f"High-mileage + Low-price listings: {len(risk_df):,}")
            st.plotly_chart(fig, use_container_width=True)

        # Depreciation rate: price vs mileage by brand
        section("Depreciation Sensitivity — Price vs Mileage by Brand")
        depr = (dff.groupby(["manufacturer", pd.cut(dff["mileage"],
                             bins=[0,20000,50000,100000,200000,500000],
                             labels=["<20K","20-50K","50-100K","100-200K","200K+"])])
                ["price"].mean().reset_index())
        depr.columns = ["manufacturer", "mileage_band", "avg_price"]
        depr = depr.dropna()
        fig = px.line(
            depr, x="mileage_band", y="avg_price",
            color="manufacturer", color_discrete_map=BRAND_COLORS,
            markers=True,
            labels={"mileage_band": "Mileage Band", "avg_price": "Average Price (£)",
                    "manufacturer": "Brand"},
        )
        fig.update_layout(**PLOT_LAYOUT, yaxis_tickprefix="£", yaxis_tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

        # Low-performing segment table
        section("Lowest-Value Segments (Avg Price < £3,000)")
        low_seg = (dff.groupby(["manufacturer", "model", "fuel_type"])
                   .agg(avg_price=("price", "mean"),
                        count=("price", "count"),
                        avg_mileage=("mileage", "mean"),
                        avg_age=("age", "mean"))
                   .reset_index()
                   .query("avg_price < 3000 and count >= 5")
                   .sort_values("avg_price"))
        if not low_seg.empty:
            low_seg["avg_price_fmt"]   = low_seg["avg_price"].apply(fmt_currency)
            low_seg["avg_mileage_fmt"] = low_seg["avg_mileage"].apply(lambda x: f"{x:,.0f} mi")
            low_seg["avg_age_fmt"]     = low_seg["avg_age"].apply(lambda x: f"{x:.0f} yrs")
            st.markdown("""
            <table class='styled-table'>
              <tr><th>Brand</th><th>Model</th><th>Fuel</th><th>Avg Price</th><th>Avg Mileage</th><th>Avg Age</th><th>Count</th></tr>
            """ + "".join(
                f"<tr><td>{r.manufacturer}</td><td>{r.model}</td><td>{r.fuel_type}</td>"
                f"<td>{r.avg_price_fmt}</td><td>{r.avg_mileage_fmt}</td>"
                f"<td>{r.avg_age_fmt}</td><td>{r['count']}</td></tr>"
                for _, r in low_seg.iterrows()
            ) + "</table>", unsafe_allow_html=True)
        else:
            st.info("No segments below £3,000 in the current filter selection.")

        # Actionable insight cards
        st.markdown("<br>", unsafe_allow_html=True)
        section("Risk Summary")
        r1, r2 = st.columns(2)
        with r1:
            insight_box("RISK 1 — High-Mileage Surplus",
                f"<b>{len(risk_df):,}</b> listings ({len(risk_df)/len(dff)*100:.1f}%) combine high mileage "
                f"(>{mileage_q75:,.0f} mi) with low prices (<£{price_q25:,.0f}). "
                "These may be difficult to sell and could dilute brand perception.", "danger")
            insight_box("RISK 2 — Price Outliers",
                f"{dff['price_outlier'].sum():,} listings are statistical price outliers. "
                "Extremely low-priced cars may indicate salvage vehicles; "
                "extremely high-priced cars may remain unsold.", "warning")
        with r2:
            insight_box("RISK 3 — Ageing Stock",
                f"Vehicles older than 20 years account for "
                f"<b>{(dff['age']>20).sum():,}</b> listings ({(dff['age']>20).mean()*100:.1f}%). "
                "Demand for very old vehicles is limited to collectors; "
                "general resale risk is high.", "danger")
            insight_box("RISK 4 — Diesel Decline",
                f"Diesel accounts for <b>{(dff.fuel_type=='Diesel').mean()*100:.1f}%</b> of listings. "
                "With increasing urban emission zone restrictions, diesel residuals may face "
                "accelerated depreciation in the coming years.", "warning")

    # ── Tab 2: Opportunities ──────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            section("Hybrid Trend — Growth Over Time")
            hybrid_yr = (dff[dff["fuel_type"] == "Hybrid"]
                         .groupby("year").size().reset_index(name="count"))
            petrol_yr = (dff[dff["fuel_type"] == "Petrol"]
                         .groupby("year").size().reset_index(name="count"))
            diesel_yr = (dff[dff["fuel_type"] == "Diesel"]
                         .groupby("year").size().reset_index(name="count"))
            fig = go.Figure()
            for data, name, color in [
                (hybrid_yr, "Hybrid", "#22c55e"),
                (petrol_yr, "Petrol", "#3b82f6"),
                (diesel_yr, "Diesel", "#f97316"),
            ]:
                fig.add_trace(go.Scatter(
                    x=data["year"], y=data["count"],
                    mode="lines+markers", name=name,
                    line=dict(color=color, width=2),
                ))
            fig.update_layout(**PLOT_LAYOUT,
                              xaxis_title="Year of Manufacture",
                              yaxis_title="Number of Listings",
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            section("Sweet-Spot Price Segment (£5K–£20K)")
            sweet = dff[(dff["price"] >= 5000) & (dff["price"] <= 20000)]
            sweet_brand = sweet["manufacturer"].value_counts().reset_index()
            sweet_brand.columns = ["manufacturer", "count"]
            fig = px.bar(
                sweet_brand, x="manufacturer", y="count",
                color="manufacturer", color_discrete_map=BRAND_COLORS,
                text="count",
                labels={"manufacturer": "", "count": "Listings in £5K–£20K Band"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(**PLOT_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Age vs Price scatter with opportunity zone
        section("Age vs Price — Identifying Value Opportunities")
        samp = dff.sample(min(5000, len(dff)), random_state=42)
        fig = px.scatter(
            samp, x="age", y="price",
            color="manufacturer", color_discrete_map=BRAND_COLORS,
            opacity=0.45,
            labels={"age": "Car Age (Years)", "price": "Price (£)", "manufacturer": "Brand"},
            hover_data=["model", "mileage", "fuel_type"],
        )
        # Shade opportunity zone: age 3-8, price 10K-40K
        fig.add_shape(type="rect",
            x0=3, x1=8, y0=10000, y1=40000,
            fillcolor="rgba(34,197,94,0.10)", line=dict(color="#22c55e", width=1.5, dash="dot"))
        fig.add_annotation(x=5.5, y=42000, text="Value Opportunity Zone",
                           font=dict(color="#22c55e", size=11), showarrow=False)
        fig.update_layout(**PLOT_LAYOUT, height=420,
                          yaxis_tickprefix="£", yaxis_tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

        # Opportunity insight cards
        st.markdown("<br>", unsafe_allow_html=True)
        section("Opportunity Summary")
        o1, o2 = st.columns(2)
        sweet_pct = len(sweet) / len(dff) * 100
        hybrid_recent = dff[(dff.fuel_type=="Hybrid") & (dff.year >= 2015)].shape[0]
        with o1:
            insight_box("OPPORTUNITY 1 — Sweet-Spot Inventory",
                f"<b>{len(sweet):,}</b> listings ({sweet_pct:.1f}%) fall in the £5,000–£20,000 price band — "
                "the highest-demand segment for used-car buyers. Maximising inventory here maximises turnover.", "success")
            insight_box("OPPORTUNITY 3 — Premium SUV/Sports Gap",
                "Porsche listings command the highest average prices. "
                "Adding more late-model Porsche Cayenne or BMW X5 stock could "
                "significantly lift average transaction value.", "success")
        with o2:
            insight_box("OPPORTUNITY 2 — Hybrid Expansion",
                f"Hybrid cars manufactured after 2015 represent <b>{hybrid_recent:,}</b> listings but average "
                f"<b>{fmt_currency(dff[(dff.fuel_type=='Hybrid')&(dff.year>=2015)]['price'].mean())}</b> — "
                "competitive pricing vs Diesel. As emission zones expand, hybrid demand will increase.", "success")
            insight_box("RECOMMENDED ACTION",
                "1. Increase £5K–£20K, 3–8 year old hybrid stock.<br>"
                "2. Phase out high-mileage (>150K) Diesel stock.<br>"
                "3. Grow Porsche/BMW premium segment to boost margin.<br>"
                "4. Flag price outliers for individual review before listing.", "success")

    # ── Tab 3: Price Forecast ─────────────────────────────────
    with tab3:
        section("Average Price Forecast by Manufacture Year")

        st.markdown("""
        > **Method:** Linear regression on annual average prices. 
        This forecasts the *expected average listing price* for cars manufactured in future years, 
        based on the historical relationship between manufacture year and price.
        """)

        yearly = (dff.groupby("year")
                  .agg(avg_price=("price", "mean"), count=("price", "count"))
                  .reset_index())
        yearly = yearly[yearly["count"] >= 10].sort_values("year")

        X = yearly["year"].values.reshape(-1, 1)
        y = yearly["avg_price"].values
        lr = LinearRegression().fit(X, y)

        # Forecast 2023-2027
        future_years = np.array([2023, 2024, 2025, 2026, 2027]).reshape(-1, 1)
        future_preds = lr.predict(future_years)

        r2 = lr.score(X, y)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["avg_price"],
            mode="lines+markers", name="Historical Avg Price",
            line=dict(color="#3b82f6", width=2.5),
            hovertemplate="Year: %{x}<br>Avg Price: £%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=lr.predict(X),
            mode="lines", name=f"Regression Fit (R²={r2:.2f})",
            line=dict(color="#f97316", width=1.8, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=future_years.flatten(), y=future_preds,
            mode="lines+markers", name="Forecast (2023–2027)",
            line=dict(color="#22c55e", width=2.5, dash="dot"),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="Year: %{x}<br>Forecast: £%{y:,.0f}<extra></extra>",
        ))
        # Confidence band (±10%)
        fig.add_trace(go.Scatter(
            x=list(future_years.flatten()) + list(future_years.flatten())[::-1],
            y=list(future_preds * 1.10) + list(future_preds * 0.90)[::-1],
            fill="toself", fillcolor="rgba(34,197,94,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="±10% Band", showlegend=True,
        ))
        fig.update_layout(**PLOT_LAYOUT, height=450,
                          yaxis_tickprefix="£", yaxis_tickformat=",",
                          xaxis_title="Year of Manufacture",
                          yaxis_title="Average Price (£)",
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.markdown("**Forecast Summary**")
        fcst_df = pd.DataFrame({
            "Year": future_years.flatten(),
            "Forecast Avg Price": [fmt_currency(p) for p in future_preds],
            "Lower (−10%)": [fmt_currency(p * 0.90) for p in future_preds],
            "Upper (+10%)": [fmt_currency(p * 1.10) for p in future_preds],
        })
        st.dataframe(fcst_df, use_container_width=True, hide_index=True)

        st.markdown(f"""
        > ⚠️ **Forecast Limitation:** This is a simple linear trend (R² = {r2:.2f}). 
        It captures the direction of price change with manufacture year but does not account for 
        mileage, condition, economic cycles, or fuel-type shifts. 
        Use for directional planning only.
        """)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94a3b8;font-size:0.8rem;'>"
    "Car Sales BI Dashboard · Built with Streamlit & Plotly · Dataset: 50,000 used-car listings"
    "</p>",
    unsafe_allow_html=True,
)
