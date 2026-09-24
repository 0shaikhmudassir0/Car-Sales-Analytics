# 🚗 Car Sales Data Analytics & Business Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![Plotly](https://img.shields.io/badge/Plotly-5.19%2B-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Title
**Car Sales Data Analytics & Business Intelligence Dashboard**

---

## 🎯 Project Objective
Build a complete end-to-end Data Analytics and Business Intelligence project on a real-world used-car
sales dataset. The goal is to extract meaningful business insights, identify risks and opportunities,
and present findings through a professional interactive Streamlit dashboard — designed for
non-technical business users.

---

## ❓ Problem Statement
Used-car dealerships and automotive marketplaces generate large volumes of listing data but often
lack the analytical tools to answer key business questions:

- Which manufacturers and models command the highest prices?
- Where are the low-performing, high-risk listings?
- What is the relationship between mileage, age, fuel type and price?
- What are the emerging market opportunities?
- How can pricing strategy be improved?

This project answers all of these questions from a single dataset.

---

## 📂 Dataset Description

| Property             | Detail                                              |
|----------------------|-----------------------------------------------------|
| **File**             | `car_sales.csv`                                     |
| **Rows**             | 50,000                                              |
| **Columns**          | 7                                                   |
| **Missing Values**   | None                                                |
| **Duplicate Rows**   | None                                                |
| **Manufacturers**    | BMW, Ford, Porsche, Toyota, VW                      |
| **Fuel Types**       | Petrol, Diesel, Hybrid                              |
| **Year Range**       | 1984 – 2022                                         |
| **Price Range**      | £76 – £168,081 (avg ≈ £13,829)                      |
| **Mileage Range**    | 630 – 453,537 miles                                 |

### Columns

| Column               | Type    | Description                             |
|----------------------|---------|-----------------------------------------|
| Manufacturer         | string  | Car brand (BMW, Ford, Porsche, Toyota, VW) |
| Model                | string  | Specific car model name                 |
| Engine size          | float   | Engine displacement in litres           |
| Fuel type            | string  | Petrol / Diesel / Hybrid                |
| Year of manufacture  | integer | Year the car was built                  |
| Mileage              | integer | Odometer reading in miles               |
| Price                | integer | Asking/sale price in GBP (£)            |

---

## 🌐 Dataset Source
This dataset is based on used-car listing data, structured similarly to publicly available
automotive datasets on Kaggle and other open-data platforms.

**Dataset Source:** [INSERT KAGGLE DATASET LINK HERE]

> Note: Replace the placeholder above with the exact Kaggle URL once confirmed.

---

## 🛠️ Technologies Used

| Category       | Tool / Library                    |
|----------------|-----------------------------------|
| Language       | Python 3.9+                       |
| Dashboard      | Streamlit                         |
| Visualisation  | Plotly Express / Plotly Graph Objects |
| Data Wrangling | Pandas, NumPy                     |
| ML / Forecast  | Scikit-learn (LinearRegression)   |
| Report         | python-docx                       |

---

## 📦 Python Libraries Used

```
streamlit>=1.32.0
pandas>=2.1.0
numpy>=1.26.0
plotly>=5.19.0
scikit-learn>=1.4.0
```

---

## 🔄 Project Workflow

```
Raw CSV Data
    │
    ▼
Data Loading & Column Standardisation
    │
    ▼
Data Cleaning (type coercion, deduplication, invalid row removal)
    │
    ▼
Feature Engineering (age, price_per_mile, decade, outlier flag)
    │
    ▼
Exploratory Data Analysis (distributions, correlations, trends)
    │
    ▼
KPI Calculation
    │
    ▼
Business Intelligence Analysis (trends, drivers, risks, opportunities)
    │
    ▼
Linear Regression Forecast
    │
    ▼
Interactive Streamlit Dashboard (3 pages)
    │
    ▼
Actionable Business Insights
```

---

## 🧹 Data Cleaning Process

1. **Column Renaming** — Standardised to lowercase snake_case for code safety.
2. **Type Coercion** — `engine_size`, `price`, `mileage` → float; `year` → Int64.
3. **Whitespace Stripping** — Applied to all text columns (manufacturer, model, fuel_type).
4. **Missing Value Removal** — Dropped rows with null in any of: price, mileage, year, engine_size.
5. **Duplicate Removal** — Exact duplicate rows dropped.
6. **Implausible Value Removal:**
   - Price ≤ 0 removed
   - Mileage ≤ 0 removed
   - Year < 1980 or > 2025 removed
7. **Derived Columns Added:**
   - `age` = 2024 − year
   - `price_per_mile` = price ÷ mileage
   - `decade` = decade band (e.g. "2010s")
   - `price_outlier` = IQR-based flag (True/False)

---

## 📊 EDA Process

- **Univariate:** Price distribution by manufacturer (box plot, violin plot)
- **Bivariate:** Engine size vs price (scatter + OLS trendline), mileage vs price (LOWESS curve)
- **Time Series:** Listings count and median price by manufacture year
- **Categorical:** Fuel type share per manufacturer, model rankings by price
- **Heatmap:** Average price by decade × fuel type
- **Depreciation:** Mileage band vs average price curves per brand

---

## 📈 KPIs

| KPI                        | Why It Matters                                                    |
|----------------------------|-------------------------------------------------------------------|
| Total Listings             | Market size indicator                                             |
| Combined Inventory Value   | Total potential revenue from all listings                         |
| Average Selling Price      | Benchmark for pricing strategy                                    |
| Median Selling Price       | Robust central tendency (not skewed by outliers)                  |
| Number of Manufacturers    | Market breadth / competition level                                |
| Number of Models           | Product diversity                                                 |
| Average Mileage            | Indicator of overall stock quality                                |
| Average Car Age            | Freshness of inventory                                            |
| Price Outlier %            | Risk exposure from extreme listings                               |
| Top Brand by Volume        | Market leader in listings                                         |
| Top Brand by Revenue       | Market leader in value                                            |

---

## 🖥️ Dashboard Features

### Page 1 — Executive Overview
- KPI cards: total listings, combined value, avg/median price, brands, models, avg mileage
- Price distribution by manufacturer (box plot)
- Listings count by manufacturer (bar chart)
- Average price trend by manufacture year with regression line
- Fuel type mix (donut chart)
- Key business insight callout boxes

### Page 2 — Sales & Product Analysis
- **Brand & Model tab:** avg price by brand, violin plots, top-20 models by price
- **Fuel & Engine tab:** price by fuel type, fuel share stacked bar, engine size vs price scatter
- **Year Trends tab:** listings by year, median price by year, decade×fuel heatmap, mileage depreciation curve
- **Rankings tab:** top/bottom 10 models table, most-listed models bar chart
- Global sidebar filters: Manufacturer, Fuel Type, Year Range

### Page 3 — Market & Risk Analysis
- **Risk Indicators tab:** outlier pie, high-mileage/low-price risk bar, depreciation curves, low-value segment table
- **Opportunities tab:** hybrid growth trend, sweet-spot band, age vs price opportunity zone scatter
- **Price Forecast tab:** linear regression forecast 2023–2027 with confidence band and summary table

---

## 💡 Business Insights

### FACTS
- Ford has the highest listing volume; Porsche has the highest average price
- Post-2015 cars average significantly more than pre-2015 stock
- Hybrid listings have grown consistently since 2012

### INSIGHTS
- Premium brands (Porsche, BMW) command 3–5× higher prices but represent smaller inventory shares
- Mileage is the single strongest predictor of price depreciation
- The £5K–£20K band is the largest segment by volume, driving the bulk of turnover

### RISKS
- ~12% of listings are statistical price outliers requiring review
- High-mileage/low-price listings dilute brand positioning
- Very old cars (>20 years) face limited resale demand
- Diesel faces increasing regulatory headwinds

### OPPORTUNITIES
- Hybrid inventory expansion ahead of emission zone growth
- 3–8 year old cars in the £10K–£40K range represent a value opportunity zone
- Growing Porsche/BMW premium inventory significantly raises average transaction value

---

## ⚠️ Risks

1. High-mileage surplus cars create a low-value inventory drag
2. Price outliers (both high and low) represent underpriced/overpriced listing risk
3. Ageing diesel stock faces accelerating depreciation from policy changes
4. Very old vehicles (pre-2000) are collector-only and hard to sell quickly

---

## 🚀 Opportunities

1. Expand hybrid inventory — growing demand, competitive pricing
2. Focus stocking on the £5K–£20K "sweet-spot" band for fastest turnover
3. Increase premium brand (Porsche, BMW) inventory to lift margin per unit
4. Phase out or discount high-mileage diesel stock proactively

---

## ✅ Recommended Actions

1. **Prioritise 2015–2022 vehicles with under 80,000 miles** — best price/demand balance
2. **Increase hybrid stock** — particularly post-2015 models
3. **Review all price outliers** individually before listing
4. **Phase out Diesel >150K miles** — mark down aggressively or avoid acquisition
5. **Grow Porsche/BMW premium segment** to improve average transaction value

---

## 🖥️ How to Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

1. Clone or download this repository.
2. Place `car_sales.csv` in the **same folder** as `app.py`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
5. The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📤 Expected Output

- An interactive 3-page Streamlit dashboard in your browser
- Page 1: Executive KPIs and overview charts
- Page 2: Detailed product/sales analysis with filters
- Page 3: Risk indicators, market opportunities, and price forecast

---

## 🔮 Future Improvements

1. Add a machine learning price predictor (XGBoost/Random Forest) so users can input car specs and get a predicted market price
2. Integrate live market data feeds for real-time listings
3. Add geolocation/regional analysis if dealer location data becomes available
4. Build a time-series sales velocity tracker once transaction date data is captured
5. Add user authentication and saved filter profiles for dealership teams
6. Export reports to PDF directly from the dashboard

---

## 📁 Project Files

```
├── app.py                  # Complete Streamlit dashboard (single file)
├── car_sales.csv           # Original dataset (do not modify)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── Project_Report.docx     # Professional project report
```

---

*Car Sales BI Dashboard · Built with Streamlit & Plotly*
