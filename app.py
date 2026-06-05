import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide")
st.title("🚀 EGX AI PRO MAX v3.1 (MULTI-TIMEFRAME PRO)")

# =========================
# 📌 EGX UNIVERSE
# =========================
EGX = [
    "COMI.CA","MFPC.CA","PHDC.CA","ACRI.CA","ORAS.CA","HRHO.CA",
    "TMGH.CA","FWRY.CA","SWDY.CA","ETEL.CA","AMOC.CA","HELI.CA"
]

# =========================
# 📊 DATA
# =========================
@st.cache_data(ttl=3600)
def load_data(symbols, period, interval):
    return yf.download(symbols, period=period, interval=interval,
                       group_by="ticker", threads=True, auto_adjust=True)

# =========================
# 📈 INDICATORS
# =========================
def add_indicators(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    df["rsi"] = ta.momentum.RSIIndicator(close).rsi()
    df["macd"] = ta.trend.MACD(close).macd()

    df["vol_ma"] = vol.rolling(20).mean()

    df["support"] = low.rolling(20).min()
    df["resistance"] = high.rolling(20).max()

    obv = ta.volume.OnBalanceVolumeIndicator(close, vol)
    df["obv"] = obv.on_balance_volume()

    return df

# =========================
# ATR
# =========================
def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()

# =========================
# ADX (Wilder Improved)
# =========================
def adx(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period).mean() / (atr_val + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period).mean() / (atr_val + 1e-9))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()

# =========================
# REGIME
# =========================
def market_regime(last):
    score = 0

    if last["Close"] > last["ema200"]:
        score += 1
    if last["ema20"] > last["ema50"]:
        score += 1
    if last["macd"] > 0:
        score += 1
    if last["rsi"] > 50:
        score += 1

    if score >= 4:
        return "🚀 Strong Bull"
    elif score == 3:
        return "🟢 Bullish"
    elif score == 2:
        return "⚠️ Neutral"
    return "🔴 Bearish"

# =========================
# ANALYZE ENGINE v3.1
# =========================
def analyze(df_d, df_w, df_m):

    df_d = add_indicators(df_d)
    df_w = add_indicators(df_w)
    df_m = add_indicators(df_m)

    last_d = df_d.iloc[-1]
    last_w = df_w.iloc[-1]
    last_m = df_m.iloc[-1]

    atr_val = atr(df_d).iloc[-1]
    adx_val = adx(df_d).iloc[-1]

    score = 0

    # ================= TREND
    if last_d["Close"] > last_d["ema200"]:
        score += 15
    if last_d["ema20"] > last_d["ema50"]:
        score += 10

    # ================= MOMENTUM
    if 45 < last_d["rsi"] < 65:
        score += 10
    if last_d["macd"] > 0:
        score += 8

    # ================= VOLUME
    if last_d["Volume"] > last_d["vol_ma"]:
        score += 10

    obv_slope = df_d["obv"].diff().iloc[-1]
    if obv_slope > 0:
        score += 6

    # ================= VOLATILITY (fixed)
    if atr_val / last_d["Close"] > 0.008:
        score += 7

    # ================= TREND STRENGTH
    if adx_val > 20:
        score += 12

    # ================= SUPPORT
    if last_d["Close"] <= last_d["support"] * 1.02:
        score += 5

    # ================= WEEKLY CONFIRMATION
    if last_w["Close"] > last_w["ema200"]:
        score += 10

    # ================= MONTHLY CONFIRMATION (NEW)
    if last_m["Close"] > last_m["ema200"]:
        score += 12

    # ================= REGIME BONUS
    regime = market_regime(last_d)
    if "Strong" in regime:
        score += 6
    elif "Bullish" in regime:
        score += 3

    # ================= NO TRADE ZONE
    if adx_val < 15:
        return score, "⛔ No Trade", regime, atr_val

    # ================= SIGNAL
    if score >= 85:
        signal = "🔥 قوي جداً"
    elif score >= 70:
        signal = "🟢 فرصة قوية"
    elif score >= 55:
        signal = "⚠️ متابعة"
    else:
        signal = "🟡 ضعيف"

    return score, signal, regime, atr_val

# =========================
# PROCESS
# =========================
def process(symbol, daily, weekly, monthly):

    try:
        df_d = daily[symbol].dropna()
        df_w = weekly[symbol].dropna()
        df_m = monthly[symbol].dropna()

        if len(df_d) < 200:
            return None

        score, signal, regime, atr_val = analyze(df_d, df_w, df_m)

        entry = df_d.iloc[-1]["Close"]

        sl = entry - atr_val * 1.5
        tp = entry + atr_val * 3

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
# RUN
# =========================
if st.button("🚀 RUN PRO MAX v3.1"):

    daily = load_data(EGX, "6mo", "1d")
    weekly = load_data(EGX, "2y", "1wk")
    monthly = load_data(EGX, "5y", "1mo")

    results = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(process, s, daily, weekly, monthly) for s in EGX]

        for f in futures:
            r = f.result()
            if r:
                results.append(r)

    if results:
        df = pd.DataFrame(results).sort_values("Score", ascending=False)

        st.success("🔥 PRO MAX v3.1 READY")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download",
            df.to_csv(index=False),
            "egx_pro_max_v3_1.csv"
        )
    else:
        st.warning("No signals found")
