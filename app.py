import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide")
st.title("🚀 EGX SMART SCANNER PRO MAX (ADAPTIVE AI MODE)")


# =========================
# 📦 LOAD STOCKS
# =========================
@st.cache_data(ttl=86400)
def get_all_stocks():
    return pd.read_csv("egx_symbols.csv")


# =========================
# 📊 DATA
# =========================
@st.cache_data(ttl=300)
def get_data(symbol):

    try:
        symbol = f"{symbol}.CA"

        df = yf.download(symbol, period="1y", interval="1d", progress=False)

        if df is None or df.empty:
            return None

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        return df

    except:
        return None


# =========================
# 🧠 MARKET REGIME DETECTOR
# =========================
def detect_market_regime(df):

    if df is None or len(df) < 50:
        return "weak"

    returns = df["close"].pct_change()

    volatility = returns.std()
    trend = df["close"].iloc[-1] / df["close"].iloc[-20] - 1

    volume_strength = df["volume"].iloc[-1] / (df["volume"].mean() + 1e-9)

    score = 0

    if trend > 0.03:
        score += 1
    if volatility > 0.02:
        score += 1
    if volume_strength > 1:
        score += 1

    if score == 3:
        return "strong"
    elif score == 2:
        return "normal"
    else:
        return "weak"


# =========================
# 📈 INDICATORS
# =========================
def add_indicators(df):

    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).ewm(alpha=1/14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()

    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9).mean()

    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    df["vol_ma"] = df["volume"].rolling(20).mean()

    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()

    df["obv"] = (np.where(
        df["close"] > df["close"].shift(1), df["volume"],
        np.where(df["close"] < df["close"].shift(1), -df["volume"], 0)
    )).cumsum()

    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()

    return df.dropna()


# =========================
# 🧠 SMART FILTER (ADAPTIVE)
# =========================
def smart_filter(df, regime):

    if df is None or df.empty:
        return False

    if len(df) < 60:
        return False

    liquidity = (df["close"] * df["volume"]).mean()

    if regime == "strong":
        return liquidity > 500_000
    elif regime == "normal":
        return liquidity > 300_000
    else:
        return liquidity > 150_000


# =========================
# 📈 ANALYZE (ADAPTIVE SCORING)
# =========================
def analyze(df, regime):

    latest = df.iloc[-1]
    score = 0
    reasons = []

    boost = 1.2 if regime == "strong" else 1.0 if regime == "normal" else 0.9

    if latest["rsi"] < 35:
        score += 10 * boost
        reasons.append("RSI Oversold")

    if latest["macd"] > latest["signal"]:
        score += 10 * boost
        reasons.append("MACD Bullish")

    if latest["ema50"] > latest["ema200"]:
        score += 10 * boost
        reasons.append("Uptrend EMA")

    if latest["volume"] > latest["vol_ma"] * 0.8:
        score += 10 * boost
        reasons.append("Volume Activity")

    if latest["obv"] > df["obv"].mean():
        score += 10 * boost
        reasons.append("OBV Strength")

    if latest["close"] > latest["vwap"]:
        score += 10 * boost
        reasons.append("Above VWAP")

    if latest["close"] <= latest["support"] * 1.05:
        score += 10 * boost
        reasons.append("Near Support")

    # 🔥 regime bonus
    if regime == "strong":
        score += 5

    # SIGNAL
    if score >= 75:
        signal = "🔥 قوي جدًا"
    elif score >= 60:
        signal = "🟢 فرصة"
    elif score >= 45:
        signal = "⚠️ مراقبة"
    else:
        signal = "🟡 ضعيف"

    return signal, score, reasons


# =========================
# 🎯 RISK
# =========================
def risk_management(df):

    latest = df.iloc[-1]

    entry = latest["close"]
    atr = latest["atr"]
    resistance = latest["resistance"]

    if pd.isna(atr):
        return None

    sl = entry - (1.5 * atr)

    risk = entry - sl

    tp1 = entry + risk
    tp2 = entry + (risk * 2)

    tp3 = entry + (risk * 3) if resistance <= entry else resistance

    return entry, sl, tp1, tp2, tp3


# =========================
# ⚙️ PROCESS STOCK
# =========================
def process_stock(row, regime):

    try:
        symbol = row["Symbol"]

        df = get_data(symbol)

        if df is None:
            return None

        df = add_indicators(df)

        if not smart_filter(df, regime):
            return None

        signal, score, reasons = analyze(df, regime)

        if score < (40 if regime != "weak" else 35):
            return None

        risk = risk_management(df)
        if risk is None:
            return None

        entry, sl, tp1, tp2, tp3 = risk

        return {
            "Symbol": symbol,
            "Sector": row["Sector"],
            "Name": row["Name"],
            "Regime": regime,
            "Signal": signal,
            "Score": round(score, 2),
            "Entry": round(entry, 3),
            "SL": round(sl, 3),
            "TP1": round(tp1, 3),
            "TP2": round(tp2, 3),
            "TP3": round(tp3, 3),
            "Reasons": ", ".join(reasons)
        }

    except:
        return None


# =========================
# 🚀 MAIN ENGINE
# =========================
results = []

if st.button("🚀 SCAN EGX ADAPTIVE MODE"):

    stocks = get_all_stocks()

    progress = st.progress(0)

    # أول سهم نستخدمه لتحديد السوق (تقريب بسيط)
    sample_df = get_data(stocks.iloc[0]["Symbol"])
    regime = detect_market_regime(sample_df)

    st.info(f"📊 Market Regime Detected: {regime.upper()}")

    with ThreadPoolExecutor(max_workers=8) as executor:

        futures = [
            executor.submit(process_stock, row, regime)
            for row in stocks.to_dict("records")
        ]

        for i, f in enumerate(futures):

            try:
                res = f.result(timeout=10)
                if res:
                    results.append(res)
            except:
                pass

            progress.progress((i + 1) / len(futures))

    if results:

        df_res = pd.DataFrame(results)

        top20 = df_res.sort_values("Score", ascending=False).head(20)
        best_sector = df_res.groupby("Sector").head(1)

        st.success("🔥 ADAPTIVE SIGNALS GENERATED")

        st.subheader("🏆 Top 20 Stocks")
        st.dataframe(top20, use_container_width=True)

        st.subheader("📊 Best Per Sector")
        st.dataframe(best_sector, use_container_width=True)

        csv = df_res.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "egx_adaptive.csv", "text/csv")

    else:
        st.warning("⚠️ No signals found even in adaptive mode")
