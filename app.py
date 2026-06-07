import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("🚀 EGX AI PRO MAX v6 (HYBRID INSTITUTIONAL ENGINE)")

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

    df["ema20"] = close.ewm(span=20).mean()
    df["ema50"] = close.ewm(span=50).mean()
    df["ema200"] = close.ewm(span=200).mean()

    df["rsi"] = ta.momentum.RSIIndicator(close).rsi()
    df["macd"] = ta.trend.MACD(close).macd()
    df["adx"] = ta.trend.ADXIndicator(high, low, close).adx()
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

    df["vol_ma"] = vol.rolling(20).mean()

    return df

# =========================
# 📊 LEVELS
# =========================
def levels(df):
    return df["Low"].rolling(20).min().iloc[-1], df["High"].rolling(20).max().iloc[-1]

# =========================
# 🌍 MARKET REGIME (NEW)
# =========================
def market_regime(df):
    trend = df["Close"].iloc[-1] > df["ema200"].iloc[-1]
    strength = df["adx"].iloc[-1]

    if trend and strength > 20:
        return "BULL"
    elif not trend and strength > 20:
        return "BEAR"
    return "SIDEWAYS"

# =========================
# 🧠 GLOBAL ML MODEL (NEW)
# =========================
ML_MODEL = None
ML_FEATURES = ["ema20","ema50","ema200","rsi","macd","adx","atr","Volume"]

def train_global_ml(data):
    global ML_MODEL

    frames = []

    for symbol in EGX:
        try:
            df = data[symbol].copy()
            df = add_indicators(df)

            df["future"] = df["Close"].pct_change(3).shift(-3)
            df["target"] = (df["future"] > 0).astype(int)

            frames.append(df)
        except:
            continue

    full = pd.concat(frames).dropna()

    if len(full) < 500:
        return None

    X = full[ML_FEATURES]
    y = full["target"]

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=7,
        random_state=42
    )

    model.fit(X, y)
    ML_MODEL = model

    return model

def ml_predict(row):
    if ML_MODEL is None:
        return 0.5

    X = pd.DataFrame([row[ML_FEATURES]])
    return ML_MODEL.predict_proba(X)[0][1]

# =========================
# 🧠 ANALYZE
# =========================
def analyze(df):

    df = add_indicators(df)
    last = df.iloc[-1]

    entry = last["Close"]
    atr = last["atr"]

    support, resistance = levels(df)
    regime = market_regime(df)

    # ===== TREND SCORE =====
    score = 0

    if last["Close"] > last["ema200"]:
        score += 1
    if last["ema20"] > last["ema50"]:
        score += 1
    if last["macd"] > 0:
        score += 1
    if last["rsi"] > 50:
        score += 1
    if last["adx"] > 20:
        score += 1
    if last["Volume"] > last["vol_ma"]:
        score += 1

    # ===== REGIME ADJUSTMENT =====
    if regime == "BEAR":
        score -= 2
    elif regime == "SIDEWAYS":
        score -= 1

    # ===== FILTER =====
    if (atr / entry) < 0.012:
        return None

    # ===== ML =====
    ml_prob = ml_predict(last)

    final_conf = (score / 6) * 0.6 + ml_prob * 0.4

    # ===== SIGNAL =====
    if final_conf > 0.75:
        signal = "🔥 قوي جداً"
    elif final_conf > 0.6:
        signal = "🟢 قوي"
    else:
        signal = "⚠️ ضعيف"

    # ===== TARGETS =====
    tp1 = entry + atr * 1.2
    tp2 = entry + atr * 2
    tp3 = entry + atr * 3

    sl = entry - atr * 1.5

    return {
        "Symbol": df.iloc[-1].name if "Symbol" in df else "",
        "Score": round(score,2),
        "ML": round(ml_prob,2),
        "Confidence": round(final_conf,2),
        "Signal": signal,
        "Regime": regime,

        "Entry": round(entry,2),
        "SL": round(sl,2),
        "TP1": round(tp1,2),
        "TP2": round(tp2,2),
        "TP3": round(tp3,2),

        "ADX": round(last["adx"],2),
        "ATR": round(atr,2),
        "Support": round(support,2),
        "Resistance": round(resistance,2)
    }

# =========================
# 🧠 PROCESS
# =========================
def process(symbol, data):
    try:
        if symbol not in data:
            return None

        df = data[symbol].dropna()
        if len(df) < 50:
            return None

        res = analyze(df)
        if not res:
            return None

        res["Symbol"] = symbol.replace(".CA","")
        return res

    except:
        return None

# =========================
# 🚀 RUN
# =========================
if st.button("🚀 RUN v6 ENGINE"):

    data = load_data(EGX, "1y", "1d")

    train_global_ml(data)

    results = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(process, s, data) for s in EGX]

        for f in futures:
            r = f.result()
            if r:
                results.append(r)

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("Confidence", ascending=False)

        st.success("🔥 v6 READY - INSTITUTIONAL MODE")

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download",
            df.to_csv(index=False),
            "egx_ai_pro_max_v6.csv"
        )

    else:
        st.warning("No strong signals found")
