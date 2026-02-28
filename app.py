import pandas as pd
import yfinance as yf
from pandas_datareader import data as web
import matplotlib.pyplot as plt

start = "2018-01-01"
end   = None

# ------------------------
# Download stock prices
# ------------------------

tickers = ["IVR", "CIM", "AGNC", "NLY"]

px = yf.download(
    tickers,
    start=start,
    end=end,
    auto_adjust=True
)["Close"]

# ------------------------
# Download funding + credit stress indicators
# ------------------------

sofr = web.DataReader("SOFR", "fred", start, end)
hy_oas = web.DataReader("BAMLH0A0HYM2", "fred", start, end)

# Combine

df = pd.concat([px, sofr, hy_oas], axis=1)

df = df.rename(columns={
    "SOFR": "SOFR",
    "BAMLH0A0HYM2": "HY_OAS"
})

df = df.sort_index().ffill()

# ------------------------
# Plot 1: normalized comparison
# ------------------------

norm = df.copy()

for col in norm.columns:
    norm[col] = 100 * norm[col] / norm[col].dropna().iloc[0]

plt.figure(figsize=(14,7))

plt.plot(norm.index, norm["IVR"], label="IVR", linewidth=2)
plt.plot(norm.index, norm["CIM"], label="CIM", linewidth=2)
plt.plot(norm.index, norm["AGNC"], label="AGNC", linewidth=2)
plt.plot(norm.index, norm["NLY"], label="NLY", linewidth=2)

plt.plot(norm.index, norm["HY_OAS"], label="HY Spread", linestyle="--")
plt.plot(norm.index, norm["SOFR"], label="SOFR", linestyle="--")

plt.title("Mortgage REIT vs Funding Stress vs Credit Spread")

plt.legend()

plt.grid(True)

plt.show()


# ------------------------
# Plot 2: dual axis
# ------------------------

fig, ax1 = plt.subplots(figsize=(14,7))

ax1.plot(df.index, df["IVR"], label="IVR")
ax1.plot(df.index, df["CIM"], label="CIM")
ax1.plot(df.index, df["AGNC"], label="AGNC")
ax1.plot(df.index, df["NLY"], label="NLY")

ax1.set_ylabel("Stock Price")

ax1.legend(loc="upper left")

ax2 = ax1.twinx()

ax2.plot(df.index, df["HY_OAS"], label="HY Spread", linestyle="--")
ax2.plot(df.index, df["SOFR"], label="SOFR", linestyle="--")

ax2.set_ylabel("Rate / Spread")

ax2.legend(loc="upper right")

plt.title("Mortgage REIT vs Repo Funding Stress")

plt.grid(True)

plt.show()