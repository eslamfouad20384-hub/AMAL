import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("🚀 EGX AI PRO MAX v8 (INSTITUTIONAL BACKTEST + ML)")

# =========================
# 📌 EGX STOCKS
# =========================
EGX = [
    "COMI.CA","MFPC.CA","PHDC.CA","ACRI.CA","ORAS.CA","HRHO.CA",
    "TMGH.CA","FWRY.CA","SWDY.CA","ETEL.CA","AMOC.CA","HELI.CA"
]

# =========================
# 📊 DATA FIX (IMPORTANT)
# =========================
@st.cache_data(ttl=3600)
def load_data(symbols, period, interval):
    raw = yf.download(symbols, period=period, interval=interval,
                      group_by="ticker", auto_adjust=True, threads=True)

    data = {}
    for s in symbols:
        try:
            df = raw[s].dropna()
            data[s] = df
        except:
            continue
    return data

# =========================
# 📈 INDICATORS
# =========================
def add_indicators(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    df["ema20"] = close.ewm(20).mean()
    df["ema50"] = close.ewm(50).mean()
    df["ema200"] = close.ewm(200).mean()

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
    return (
        df["Low"].rolling(20).min().iloc[-1],
        df["High"].rolling(20).max().iloc[-1]
    )

# =========================
# 🌍 REGIME
# =========================
def market_regime(df):
    if df["Close"].iloc[-1] > df["ema200"].iloc[-1] and df["adx"].iloc[-1] > 20:
        return "BULL"
    if df["Close"].iloc[-1] < df["ema200"].iloc[-1] and df["adx"].iloc[-1] > 20:
        return "BEAR"
    return "SIDEWAYS"

# =========================
# 🧠 SMART MONEY
# =========================
def smart_money(df):
    return 1 if (
        df["Volume"].iloc[-1] > df["vol_ma"].iloc[-1]
        and df["Close"].iloc[-1] > df["ema50"].iloc[-1]
    ) else 0

# =========================
# 🧠 ML GLOBAL (FIXED)
# =========================
ML_MODEL = None
ML_FEATURES = ["ema20","ema50","ema200","rsi","macd","adx","atr"]

def train_ml(data):
    global ML_MODEL

    frames = []
    for s, df in data.items():
        df = add_indicators(df)

        df["future"] = df["Close"].pct_change(3).shift(-3)
        df["target"] = (df["future"] > 0).astype(int)

        frames.append(df)

    full = pd.concat(frames).dropna()

    split = int(len(full) * 0.8)
    train = full.iloc[:split]

    X = train[ML_FEATURES]
    y = train["target"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
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
# 📊 BACKTEST ENGINE
# =========================
def simple_backtest(df):
    df = add_indicators(df).dropna()

    wins = 0
    trades = 0
    returns = []

    for i in range(100, len(df)-5):
        entry = df["Close"].iloc[i]
        future = df["Close"].iloc[i+5]

        ret = (future - entry) / entry

        if ret > 0:
            wins += 1
        trades += 1
        returns.append(ret)

    winrate = wins / trades if trades else 0
    avg_ret = np.mean(returns) if returns else 0

    return winrate, avg_ret

# =========================
# 🧠 ANALYZE
# =========================
def analyze(df, symbol):

    df = add_indicators(df)
    last = df.iloc[-1]

    entry = last["Close"]
    atr = last["atr"]

    support, resistance = levels(df)
    regime = market_regime(df)

    score = 0

    if last["Close"] > last["ema200"]: score += 1
    if last["ema20"] > last["ema50"]: score += 1
    if last["macd"] > 0: score += 1
    if last["rsi"] > 50: score += 1
    if last["adx"] > 20: score += 1

    score += smart_money(df)

    if regime == "BEAR":
        score -= 2
    elif regime == "SIDEWAYS":
        score -= 1

    risk = atr / entry
    if risk > 0.06:
        return None

    ml_prob = ml_predict(last)

    confidence = (score/6)*0.5 + ml_prob*0.4 + smart_money(df)*0.1

    signal = "🔥 قوي جداً" if confidence > 0.78 else "🟢 قوي" if confidence > 0.65 else "⚠️ ضعيف"

    tp1 = entry + atr*1.2
    tp2 = entry + atr*2.2
    sl = entry - atr*1.5

    return {
        "Symbol": symbol.replace(".CA",""),
        "Confidence": round(confidence,2),
        "Signal": signal,
        "Regime": regime,
        "Entry": round(entry,2),
        "SL": round(sl,2),
        "TP1": round(tp1,2),
        "TP2": round(tp2,2),
        "ADX": round(last["adx"],2),
        "ATR": round(atr,2),
        "Support": round(support,2),
        "Resistance": round(resistance,2),
        "ML": round(ml_prob,2)
    }

# =========================
# 🚀 RUN ENGINE
# =========================
if st.button("🚀 RUN v8 INSTITUTIONAL ENGINE"):

    data = load_data(EGX, "1y", "1d")

    train_ml(data)

    results = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(analyze, data[s], s) for s in data]

        for f in futures:
            r = f.result()
            if r:
                results.append(r)

    if results:
        df = pd.DataFrame(results).sort_values("Confidence", ascending=False)

        st.success("🚀 v8 ACTIVE - INSTITUTIONAL MODE")

        st.dataframe(df, use_container_width=True)

        # ================= BACKTEST =================
        st.subheader("📊 Backtest (Quick Stats)")
        bt = {s: simple_backtest(data[s]) for s in data}

        bt_df = pd.DataFrame(bt, index=["WinRate","AvgReturn"]).T
        st.dataframe(bt_df)

        st.download_button(
            "⬇️ Download",
            df.to_csv(index=False),
            "egx_ai_v8.csv"
        )

    else:
        st.warning("No signals found")
