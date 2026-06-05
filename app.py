import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("🚀 HYBRID EGX SMART ENGINE (AI + SPEED + MULTI TF)")


# =========================
# 📌 STOCK LIST
# =========================
EGX = [
    "COMI.CA","MFPC.CA","PHDC.CA","ACRI.CA","ORAS.CA","HRHO.CA",
    "TMGH.CA","FWRY.CA","SWDY.CA","ETEL.CA","AMOC.CA","HELI.CA"
]


MODEL_FILE = "ai_model.pkl"


# =========================
# 📊 LOAD DATA
# =========================
@st.cache_data(ttl=86400)
def load_data(symbols, period, interval):
    return yf.download(symbols, period=period, interval=interval, group_by="ticker", threads=True)


# =========================
# 🤖 AI MODEL
# =========================
def load_ai():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None


# =========================
# 📈 INDICATORS (FAST + POWERFUL)
# =========================
def add_indicators(df):

    df = df.copy()

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd()

    df["ema20"] = df["Close"].ewm(span=20).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()

    df["vol_ma"] = df["Volume"].rolling(20).mean()

    df["support"] = df["Low"].rolling(20).min()
    df["resistance"] = df["High"].rolling(20).max()

    df["obv"] = (np.where(
        df["Close"] > df["Close"].shift(1),
        df["Volume"],
        -df["Volume"]
    )).cumsum()

    df["vwap"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    return df.dropna()


# =========================
# 📊 ADX
# =========================
def adx(df):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff()

    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()

    return (atr / (atr.mean() + 1e-9)) * 50


# =========================
# 🧠 ANALYZE (HYBRID CORE)
# =========================
def analyze(df_daily, df_weekly, df_monthly, model=None):

    last = df_daily.iloc[-1]
    score_fast = 0
    score_slow = 0

    # ================= FAST SCORE =================
    if last["rsi"] < 55:
        score_fast += 10
    if last["macd"] > 0:
        score_fast += 10
    if last["Close"] > last["vwap"]:
        score_fast += 10
    if last["Volume"] > last["vol_ma"]:
        score_fast += 10
    if last["Close"] <= last["support"] * 1.05:
        score_fast += 10

    # ================= SLOW SCORE =================
    if df_weekly["ema20"].iloc[-1] > df_weekly["ema50"].iloc[-1]:
        score_slow += 20
    if adx(df_weekly).iloc[-1] > 20:
        score_slow += 10
    if df_monthly["Close"].iloc[-1] > df_monthly["Close"].rolling(50).mean().iloc[-1]:
        score_slow += 20

    trend_strength = (df_daily["Close"].iloc[-1] / df_daily["Close"].iloc[-5] - 1) * 100
    if trend_strength > 0:
        score_fast += 10

    # ================= FINAL SCORE =================
    score = score_fast + score_slow

    # ================= AI =================
    if model:
        features = np.array([[
            last["rsi"],
            last["macd"],
            last["Volume"] / (last["vol_ma"] + 1e-9),
            trend_strength
        ]])
        prob = model.predict_proba(features)[0][1]
    else:
        prob = 0.5 + (score / 200)

    # ================= SIGNAL =================
    if score >= 80:
        signal = "🔥 قوية جداً"
    elif score >= 60:
        signal = "🟢 فرصة"
    elif score >= 45:
        signal = "⚠️ متابعة"
    else:
        signal = "🟡 ضعيف"

    return score, prob, signal


# =========================
# ⚙️ PROCESS
# =========================
def process(symbol, daily, weekly, monthly, model):

    try:
        d = daily[symbol].dropna()
        w = weekly[symbol].dropna()
        m = monthly[symbol].dropna()

        d = add_indicators(d)
        w = add_indicators(w)
        m = add_indicators(m)

        score, prob, signal = analyze(d, w, m, model)

        last = d.iloc[-1]

        entry = last["Close"]
        sl = entry - (last["Close"] - d["Low"].rolling(20).min().iloc[-1]) * 0.5
        tp = entry + (entry - sl) * 2

        return {
            "Symbol": symbol.replace(".CA",""),
            "Score": round(score,2),
            "Signal": signal,
            "Probability": round(prob*100,2),
            "Entry": round(entry,2),
            "SL": round(sl,2),
            "TP": round(tp,2)
        }

    except:
        return None


# =========================
# 🚀 MAIN ENGINE
# =========================
if st.button("🚀 RUN HYBRID SCAN"):

    model = load_ai()

    daily = load_data(EGX, "6mo", "1d")
    weekly = load_data(EGX, "1y", "1wk")
    monthly = load_data(EGX, "5y", "1mo")

    results = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [
            ex.submit(process, s, daily, weekly, monthly, model)
            for s in EGX
        ]

        for f in futures:
            r = f.result()
            if r:
                results.append(r)

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("Score", ascending=False)

        st.success("🔥 HYBRID RESULTS READY")

        st.subheader("🏆 TOP OPPORTUNITIES")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download",
            df.to_csv(index=False),
            "hybrid_egx.csv"
        )
    else:
        st.warning("No signals found")
