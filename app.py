import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide")
st.title("🚀 EGX AI PRO MAX v4 (INSTITUTIONAL TARGET SYSTEM)")

# =========================
# 📌 EGX UNIVERSE
# =========================
EGX = [
    "COMI.CA","MFPC.CA","PHDC.CA","ACRI.CA","ORAS.CA","HRHO.CA",
    "TMGH.CA","FWRY.CA","SWDY.CA","ETEL.CA","AMOC.CA","HELI.CA"
]

# =========================
# 📊 DATA LOADER
# =========================
@st.cache_data(ttl=3600)
def load_data(symbols, period, interval):
    return yf.download(
        symbols,
        period=period,
        interval=interval,
        group_by="ticker",
        threads=True,
        auto_adjust=True
    )

# =========================
# 📈 INDICATORS
# =========================
def add_indicators(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    df["ema20"] = close.ewm(span=20).mean()
    df["ema50"] = close.ewm(span=50).mean()
    df["ema200"] = close.ewm(span=200).mean()

    df["rsi"] = ta.momentum.RSIIndicator(close).rsi()
    df["macd"] = ta.trend.MACD(close).macd()

    df["vol_ma"] = vol.rolling(20).mean()

    df["support"] = low.rolling(20).min()
    df["resistance"] = high.rolling(20).max()

    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()

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
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()

# =========================
# 📊 ADX
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

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period).mean() / (atr_val + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period).mean() / (atr_val + 1e-9)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()

# =========================
# 🧠 REGIME
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
# 🧠 PRO ANALYSIS ENGINE
# =========================
def analyze(df_d, df_w, df_m):

    df_d = add_indicators(df_d)
    df_w = add_indicators(df_w)
    df_m = add_indicators(df_m)

    last_d = df_d.iloc[-1]
    last_w = df_w.iloc[-1]
    last_m = df_m.iloc[-1]

    entry = last_d["Close"]

    atr_val = atr(df_d).iloc[-1]
    adx_val = adx(df_d).iloc[-1]

    score = 0

    # ================= TREND
    if last_d["Close"] > last_d["ema200"]:
        score += 15
    if last_w["Close"] > last_w["ema200"]:
        score += 12
    if last_m["Close"] > last_m["ema200"]:
        score += 15

    # ================= MOMENTUM
    if 45 < last_d["rsi"] < 65:
        score += 8
    if last_d["macd"] > 0:
        score += 6

    # ================= VOLUME
    if last_d["Volume"] > last_d["vol_ma"]:
        score += 8

    # ================= ADX
    if adx_val > 20:
        score += 10

    regime = market_regime(last_d)
    if "Strong" in regime:
        score += 6

    # ================= RISK MODEL
    risk = atr_val / entry
    if risk < 0.05:
        score += 5
    else:
        score -= 5

    # ================= TARGETS (PRO LEVEL)
    support = last_d["support"]
    resistance = last_d["resistance"]

    sl = entry - atr_val * 1.5

    # SHORT TARGET
    tp1 = entry + atr_val * 2

    # MID TARGET
    tp2 = entry + atr_val * 4

    # LONG TARGET (INSTITUTIONAL)
    tp3 = max(
        resistance,
        entry + atr_val * 6,
        entry * 1.20
    )

    # ================= PROBABILITY MODEL (simple)
    def prob(x):
        base = score / 100
        dist_factor = max(0.3, 1 - abs(x - entry) / entry)
        return round(min(0.95, base * dist_factor), 2)

    return {
        "Score": round(score,2),
        "Signal": "🔥 قوي جداً" if score > 85 else "🟢 فرصة قوية" if score > 70 else "⚠️ متابعة",
        "Regime": regime,
        "Entry": round(entry,2),
        "SL": round(sl,2),

        "TP1 (Short)": round(tp1,2),
        "TP2 (Mid)": round(tp2,2),
        "TP3 (Long)": round(tp3,2),

        "Prob_TP1": prob(tp1),
        "Prob_TP2": prob(tp2),
        "Prob_TP3": prob(tp3),
    }

# =========================
# PROCESSOR
# =========================
def process(symbol, daily, weekly, monthly):
    try:
        df_d = daily[symbol].dropna()
        df_w = weekly[symbol].dropna()
        df_m = monthly[symbol].dropna()

        if df_d.empty or df_w.empty or df_m.empty:
            return None

        result = analyze(df_d, df_w, df_m)
        result["Symbol"] = symbol.replace(".CA","")
        return result

    except:
        return None

# =========================
# RUN
# =========================
if st.button("🚀 RUN PRO MAX v4"):

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
        df = pd.DataFrame(results)
        df = df.sort_values("Score", ascending=False)

        st.success("🔥 INSTITUTIONAL SYSTEM READY")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download",
            df.to_csv(index=False),
            "egx_pro_max_v4.csv"
        )
    else:
        st.warning("No signals found")
