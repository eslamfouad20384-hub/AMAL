import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🚀 HYBRID EGX SMART ENGINE PRO (FIXED INDICATORS)")

# =========================
# 📌 STOCK LIST
# =========================
EGX = [
    "COMI.CA","MFPC.CA","PHDC.CA","ACRI.CA","ORAS.CA","HRHO.CA",
    "TMGH.CA","FWRY.CA","SWDY.CA","ETEL.CA","AMOC.CA","HELI.CA"
]

# =========================
# 📊 DATA
# =========================
@st.cache_data(ttl=86400)
def load_data(symbols, period, interval):
    return yf.download(symbols, period=period, interval=interval, group_by="ticker", threads=True)

# =========================
# 📈 INDICATORS
# =========================
def add_indicators(df):
    df = df.copy()

    # RSI / MACD
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd()

    # EMA
    df["ema20"] = df["Close"].ewm(span=20).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()
    df["ema200"] = df["Close"].ewm(span=200).mean()

    # Volume
    df["vol_ma"] = df["Volume"].rolling(20).mean()

    # Support / Resistance (basic)
    df["support"] = df["Low"].rolling(20).min()
    df["resistance"] = df["High"].rolling(20).max()

    # OBV
    df["obv"] = (np.where(
        df["Close"] > df["Close"].shift(1),
        df["Volume"],
        -df["Volume"]
    )).cumsum()

    return df

# =========================
# 📊 ATR
# =========================
def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()

# =========================
# 📊 ADX (TRUE WILDER)
# =========================
def adx(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)

    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)

    atr_val = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr_val + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr_val + 1e-9))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx_val = dx.rolling(period).mean()

    return adx_val

# =========================
# 📊 VWAP (DAILY RESET)
# =========================
def vwap(df):
    df = df.copy()
    df["date"] = df.index.date

    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    df["tpv"] = typical * df["Volume"]

    df["cum_tpv"] = df.groupby("date")["tpv"].cumsum()
    df["cum_vol"] = df.groupby("date")["Volume"].cumsum()

    df["vwap"] = df["cum_tpv"] / (df["cum_vol"] + 1e-9)

    return df

# =========================
# 🧠 ANALYZE
# =========================
def analyze(df):
    last = df.iloc[-1]

    score = 0

    # ================= TREND REGIME (EMA 200)
    bullish_regime = last["Close"] > last["ema200"]

    if bullish_regime:
        score += 20

    # ================= RSI
    if last["rsi"] < 55:
        score += 10

    # ================= MACD
    if last["macd"] > 0:
        score += 10

    # ================= VWAP
    if last["Close"] > last["vwap"]:
        score += 10

    # ================= Volume
    if last["Volume"] > last["vol_ma"]:
        score += 10

    # ================= ATR VOLATILITY FILTER
    atr_val = atr(df).iloc[-1]
    atr_avg = atr(df).mean()

    volatility_ok = atr_val > atr_avg * 0.7

    if volatility_ok:
        score += 10

    # ================= ADX TREND STRENGTH
    adx_val = adx(df).iloc[-1]

    if adx_val > 20:
        score += 20

    # ================= SUPPORT ZONE
    if last["Close"] <= last["support"] * 1.05:
        score += 10

    # ================= SIGNAL
    if score >= 80:
        signal = "🔥 قوية جداً"
    elif score >= 60:
        signal = "🟢 فرصة"
    elif score >= 45:
        signal = "⚠️ متابعة"
    else:
        signal = "🟡 ضعيف"

    return score, signal

# =========================
# ⚙️ PROCESS
# =========================
def process(symbol, daily, weekly, monthly):

    try:
        df = daily[symbol].dropna()

        df = add_indicators(df)
        df = vwap(df)

        score, signal = analyze(df)

        last = df.iloc[-1]

        entry = last["Close"]
        sl = last["Close"] - atr(df).iloc[-1] * 1.5
        tp = entry + (entry - sl) * 2

        return {
            "Symbol": symbol.replace(".CA",""),
            "Score": round(score,2),
            "Signal": signal,
            "Entry": round(entry,2),
            "SL": round(sl,2),
            "TP": round(tp,2)
        }

    except:
        return None

# =========================
# 🚀 RUN
# =========================
if st.button("🚀 RUN PRO SCAN"):

    daily = load_data(EGX, "6mo", "1d")
    weekly = load_data(EGX, "1y", "1wk")
    monthly = load_data(EGX, "5y", "1mo")

    results = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [
            ex.submit(process, s, daily, weekly, monthly)
            for s in EGX
        ]

        for f in futures:
            r = f.result()
            if r:
                results.append(r)

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("Score", ascending=False)

        st.success("🔥 PRO RESULTS READY")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download",
            df.to_csv(index=False),
            "egx_pro.csv"
        )
    else:
        st.warning("No signals found")
