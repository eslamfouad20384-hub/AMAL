import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("🚀 EGX AI PRO MAX v5 (SMART AI + ML ENGINE)")

# =========================
# 📌 EGX STOCKS
# =========================
EGX = [
    "COMI.CA","MFPC.CA","PHDC.CA","ACRI.CA","ORAS.CA","HRHO.CA",
    "TMGH.CA","FWRY.CA","SWDY.CA","ETEL.CA","AMOC.CA","HELI.CA"
]

# =========================
# 📊 LOAD DATA
# =========================
@st.cache_data(ttl=3600)
def load_data(symbols, period, interval):
    return yf.download(symbols, period=period, interval=interval,
                        group_by="ticker", threads=True, auto_adjust=True)

# =========================
# 📈 INDICATORS FIXED
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

    df["atr"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

    adx_indicator = ta.trend.ADXIndicator(high, low, close)
    df["adx"] = adx_indicator.adx()

    return df

# =========================
# 📊 SUPPORT / RESISTANCE FIXED
# =========================
def levels(df):
    support = df["Low"].rolling(20).min().iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]
    return support, resistance

# =========================
# 🧠 SIDEWAYS FILTER
# =========================
def is_sideways(df):
    atr_pct = df["atr"].iloc[-1] / df["Close"].iloc[-1]
    return atr_pct < 0.015  # سوق هادي جدًا

# =========================
# 🧠 ML MODEL (simple training)
# =========================
def train_ml(df):
    df = df.copy()

    df["future_ret"] = df["Close"].pct_change(3).shift(-3)
    df["target"] = (df["future_ret"] > 0).astype(int)

    features = ["ema20","ema50","ema200","rsi","macd","adx","atr","Volume"]
    df = df.dropna()

    X = df[features]
    y = df["target"]

    if len(df) < 100:
        return None

    model = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42)
    model.fit(X, y)

    return model, features

# =========================
# 🧠 PREDICT SIGNAL
# =========================
def ml_predict(model_pack, last_row):
    if model_pack is None:
        return 0.5

    model, features = model_pack
    X = pd.DataFrame([last_row[features]])
    return model.predict_proba(X)[0][1]

# =========================
# 🧠 ANALYZE ENGINE
# =========================
def analyze(df):
    df = add_indicators(df)

    last = df.iloc[-1]

    entry = last["Close"]
    atr_val = last["atr"]
    adx_val = last["adx"]

    support, resistance = levels(df)

    # ===== REGIME =====
    trend_score = 0
    if last["Close"] > last["ema200"]:
        trend_score += 1
    if last["ema20"] > last["ema50"]:
        trend_score += 1
    if last["macd"] > 0:
        trend_score += 1
    if last["rsi"] > 50:
        trend_score += 1

    regime = "🚀 قوي" if trend_score >= 3 else "⚠️ ضعيف" if trend_score == 2 else "🔴 هابط"

    # ===== SIDEWAYS FILTER =====
    if is_sideways(df):
        return None

    # ===== SCORE =====
    score = 0
    score += trend_score * 20
    score += 10 if adx_val > 20 else 0
    score += 10 if last["Volume"] > last["vol_ma"] else 0

    risk = atr_val / entry
    score += 10 if risk < 0.04 else -10

    # ===== TARGETS =====
    ema_trend = (last["ema20"] - last["ema200"]) / entry

    tp1 = entry + atr_val * 1.2
    tp2 = entry + atr_val * 2.2
    tp3 = max(resistance, entry + atr_val * (3 + abs(ema_trend) * 5))

    sl = entry - atr_val * 1.6

    # ===== ML MODEL =====
    ml_model = train_ml(df)
    ml_prob = ml_predict(ml_model, last)

    # ===== FINAL CONFIDENCE =====
    base_conf = score / 100
    final_conf = (base_conf * 0.6) + (ml_prob * 0.4)

    signal = "🔥 قوي جداً" if final_conf > 0.75 else "🟢 قوي" if final_conf > 0.6 else "⚠️ ضعيف"

    return {
        "Score": round(score,2),
        "ML_Prob": round(ml_prob,2),
        "Confidence": round(final_conf,2),
        "Signal": signal,
        "Regime": regime,

        "Entry": round(entry,2),
        "SL": round(sl,2),
        "TP1": round(tp1,2),
        "TP2": round(tp2,2),
        "TP3": round(tp3,2),

        "ADX": round(adx_val,2),
        "ATR": round(atr_val,2),
        "Support": round(support,2),
        "Resistance": round(resistance,2)
    }

# =========================
# 🧠 PROCESS
# =========================
def process(symbol, data):
    try:
        df = data[symbol].dropna()
        if df.empty:
            return None

        res = analyze(df)
        if res is None:
            return None

        res["Symbol"] = symbol.replace(".CA","")
        return res

    except:
        return None

# =========================
# 🚀 RUN
# =========================
if st.button("🚀 RUN AI PRO MAX v5"):

    daily = load_data(EGX, "1y", "1d")

    results = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(process, s, daily) for s in EGX]

        for f in futures:
            r = f.result()
            if r:
                results.append(r)

    if results:
        df = pd.DataFrame(results)

        df = df.sort_values("Confidence", ascending=False)

        st.success("🔥 v5 AI SYSTEM READY")

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download",
            df.to_csv(index=False),
            "egx_ai_pro_max_v5.csv"
        )
    else:
        st.warning("No strong signals found")
