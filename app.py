import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🔥 EGX AI PRO MAX v2 (SMART MULTI-TIMEFRAME ENGINE)")

# =========================
# 📌 EGX STOCKS
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

    df["ema20"] = df["Close"].ewm(span=20).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()
    df["ema200"] = df["Close"].ewm(span=200).mean()

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd()

    df["vol_ma"] = df["Volume"].rolling(20).mean()

    df["support"] = df["Low"].rolling(20).min()
    df["resistance"] = df["High"].rolling(20).max()

    # ================= OBV (TRUE)
    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    df["obv"] = obv.on_balance_volume()

    return df

# =========================
# 📊 ATR (Wilder)
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

    return tr.ewm(alpha=1/period, adjust=False).mean()

# =========================
# 📊 ADX (WILDER TRUE)
# =========================
def adx_wilder(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)

    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period).mean() / (atr_val + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period).mean() / (atr_val + 1e-9))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx

# =========================
# 📊 VWAP
# =========================
def vwap(df):
    df = df.copy()
    df["date"] = df.index.date

    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["tpv"] = tp * df["Volume"]

    df["cum_tpv"] = df.groupby("date")["tpv"].cumsum()
    df["cum_vol"] = df.groupby("date")["Volume"].cumsum()

    df["vwap"] = df["cum_tpv"] / (df["cum_vol"] + 1e-9)

    return df

# =========================
# 🧠 MARKET REGIME
# =========================
def market_regime(last):
    bullish = 0

    if last["Close"] > last["ema200"]:
        bullish += 1
    if last["ema20"] > last["ema50"]:
        bullish += 1
    if last["macd"] > 0:
        bullish += 1

    if bullish == 3:
        return "🚀 Strong Bull"
    elif bullish == 2:
        return "🟢 Bullish"
    elif bullish == 1:
        return "⚠️ Weak Trend"
    else:
        return "🔴 Bearish"

# =========================
# 🧠 ANALYSIS ENGINE
# =========================
def analyze(df_daily, df_weekly):

    df_daily = add_indicators(df_daily)
    df_daily = vwap(df_daily)

    last_d = df_daily.iloc[-1]

    # weekly trend confirmation
    df_weekly = add_indicators(df_weekly)
    last_w = df_weekly.iloc[-1]

    score = 0

    # ================= TREND
    if last_d["Close"] > last_d["ema200"]:
        score += 20

    if last_d["ema20"] > last_d["ema50"]:
        score += 10

    # ================= RSI
    if 40 < last_d["rsi"] < 65:
        score += 10

    # ================= MACD
    if last_d["macd"] > 0:
        score += 10

    # ================= VWAP
    if last_d["Close"] > last_d["vwap"]:
        score += 10

    # ================= VOLUME
    if last_d["Volume"] > last_d["vol_ma"]:
        score += 10

    # ================= ATR filter
    atr_val = atr(df_daily).iloc[-1]
    if atr_val > atr(df_daily).mean() * 0.8:
        score += 10

    # ================= ADX strength
    adx_val = adx_wilder(df_daily).iloc[-1]
    if adx_val > 20:
        score += 15

    # ================= SUPPORT ZONE
    if last_d["Close"] <= last_d["support"] * 1.03:
        score += 5

    # ================= WEEKLY CONFIRMATION
    if last_w["Close"] > last_w["ema200"]:
        score += 10

    # ================= REGIME
    regime = market_regime(last_d)

    # ================= FINAL SIGNAL
    if score >= 85:
        signal = "🔥 قوي جداً"
    elif score >= 70:
        signal = "🟢 فرصة قوية"
    elif score >= 55:
        signal = "⚠️ متابعة"
    else:
        signal = "🟡 ضعيف"

    return score, signal, regime

# =========================
# ⚙️ PROCESS
# =========================
def process(symbol, daily, weekly):

    try:
        df_d = daily[symbol].dropna()
        df_w = weekly[symbol].dropna()

        score, signal, regime = analyze(df_d, df_w)

        last = df_d.iloc[-1]

        entry = last["Close"]
        sl = entry - atr(df_d).iloc[-1] * 1.5
        tp = entry + (entry - sl) * 2

        return {
            "Symbol": symbol.replace(".CA",""),
            "Score": round(score,2),
            "Signal": signal,
            "Regime": regime,
            "Entry": round(entry,2),
            "SL": round(sl,2),
            "TP": round(tp,2)
        }

    except:
        return None

# =========================
# 🚀 RUN
# =========================
if st.button("🚀 RUN AI PRO MAX v2"):

    daily = load_data(EGX, "6mo", "1d")
    weekly = load_data(EGX, "2y", "1wk")

    results = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(process, s, daily, weekly) for s in EGX]

        for f in futures:
            r = f.result()
            if r:
                results.append(r)

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("Score", ascending=False)

        st.success("🔥 AI PRO MAX v2 READY")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            "egx_ai_pro_max_v2.csv"
        )
    else:
        st.warning("No signals found")
