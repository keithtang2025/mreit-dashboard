import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="mREIT Risk Dashboard", layout="wide")
st.title("mREIT Risk Dashboard — IVR / CIM / AGNC / NLY vs SOFR & HY OAS")

# ----------------------------
# Controls
# ----------------------------
col1, col2, col3 = st.columns([1.2, 1.2, 1.0])
with col1:
    start = st.date_input("Start date", value=pd.to_datetime("2018-01-01"))
with col2:
    end = st.date_input("End date", value=pd.Timestamp.today().date())
with col3:
    normalize = st.checkbox("Normalize series (start=100)", value=True)

tickers_all = ["IVR", "CIM", "AGNC", "NLY"]
tickers = st.multiselect("Tickers", tickers_all, default=tickers_all)
rolling = st.slider("Rolling correlation window (days)", 20, 180, 60)

if not tickers:
    st.warning("Please select at least one ticker.")
    st.stop()

# ----------------------------
# FRED loader (no pandas_datareader)
# ----------------------------
@st.cache_data(ttl=6 * 60 * 60)
def load_fred_series(series_id: str) -> pd.Series:
    # FRED provides a simple CSV endpoint
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    s = pd.read_csv(url)
    s.columns = ["DATE", series_id]
    s["DATE"] = pd.to_datetime(s["DATE"])
    s = s.set_index("DATE")[series_id]
    s = pd.to_numeric(s, errors="coerce")
    return s

@st.cache_data(ttl=6 * 60 * 60)
def load_data(tickers, start, end):
    px = yf.download(
        tickers,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False
    )["Close"]

    sofr = load_fred_series("SOFR")
    hy_oas = load_fred_series("BAMLH0A0HYM2")

    df = pd.concat([px, sofr, hy_oas], axis=1)
    df = df.rename(columns={"SOFR": "SOFR", "BAMLH0A0HYM2": "HY_OAS"})
    df = df.sort_index().ffill()

    # Filter to selected date range
    df = df.loc[pd.to_datetime(start):pd.to_datetime(end)]
    return df

def normalize_to_100(s: pd.Series) -> pd.Series:
    s2 = s.dropna()
    if len(s2) == 0:
        return s
    return 100 * s / s2.iloc[0]

def risk_score(latest_sofr, latest_hy):
    score = 0
    if latest_hy >= 7: score += 4
    elif latest_hy >= 5: score += 3
    elif latest_hy >= 4: score += 2
    elif latest_hy >= 3: score += 1

    if latest_sofr >= 5: score += 3
    elif latest_sofr >= 4: score += 2
    elif latest_sofr >= 3: score += 1

    score = min(score, 10)
    if score <= 2: level = "🟢 Normal"
    elif score <= 5: level = "🟡 Stress building"
    elif score <= 8: level = "🟠 Dangerous"
    else: level = "🔴 Crisis"
    return score, level

df = load_data(tickers, start, end)

if df.dropna().empty:
    st.error("No data returned. Try a different date range.")
    st.stop()

latest = df.dropna().iloc[-1]
score, level = risk_score(latest["SOFR"], latest["HY_OAS"])

m1, m2, m3 = st.columns(3)
m1.metric("SOFR (%)", f"{latest['SOFR']:.2f}")
m2.metric("HY OAS (%)", f"{latest['HY_OAS']:.2f}")
m3.metric("Risk score (0-10)", f"{score}  {level}")

st.divider()

# ----------------------------
# Plot 1
# ----------------------------
plot_df = df.copy()
if normalize:
    for c in tickers + ["SOFR", "HY_OAS"]:
        plot_df[c] = normalize_to_100(plot_df[c])

fig = plt.figure(figsize=(14, 6))
for t in tickers:
    plt.plot(plot_df.index, plot_df[t], label=t, linewidth=2)
plt.plot(plot_df.index, plot_df["SOFR"], label="SOFR", linestyle="--")
plt.plot(plot_df.index, plot_df["HY_OAS"], label="HY OAS", linestyle="--")
plt.title("Normalized comparison" if normalize else "Levels")
plt.grid(True)
plt.legend()
st.pyplot(fig, use_container_width=True)

# ----------------------------
# Plot 2: rolling correlation
# ----------------------------
st.subheader("Rolling correlation (stock returns vs changes in HY_OAS & SOFR)")

corr_df = pd.DataFrame(index=df.index)
for t in tickers:
    corr_df[f"{t} vs HY_OAS"] = df[t].pct_change().rolling(rolling).corr(df["HY_OAS"].diff())
    corr_df[f"{t} vs SOFR"] = df[t].pct_change().rolling(rolling).corr(df["SOFR"].diff())

fig2 = plt.figure(figsize=(14, 5))
for c in corr_df.columns:
    plt.plot(corr_df.index, corr_df[c], label=c)
plt.axhline(0, linewidth=1)
plt.title(f"{rolling}-day rolling correlation")
plt.grid(True)
plt.legend(ncols=2)
st.pyplot(fig2, use_container_width=True)

st.divider()

st.subheader("Data preview")
st.dataframe(df.tail(20), use_container_width=True)

csv = df.to_csv().encode("utf-8")
st.download_button(
    "Download combined CSV",
    data=csv,
    file_name="mreit_dashboard_data.csv",
    mime="text/csv"
)