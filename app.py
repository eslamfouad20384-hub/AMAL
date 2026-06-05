import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide")
st.title("🚀 EGX SMART SCANNER PRO MAX")

# =========================
# 📦 LOAD STOCKS
# =========================
@st.cache_data(ttl=86400)
def get_all_stocks():
    return pd.read_csv("egx_symbols.csv")


# =========================
# 📊 DATA LOADER (FIXED STABILITY)
# =========================
@st.cache_data(ttl=300)
def get_data(symbol, interval="1d", period="1y"):
    try:
        symbol = f"{symbol}.CA"

        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False
        )

        if df is None or df.empty:
            return None

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        return df

    except:
        return None


# =========================
# 📈 INDICATORS (UNCHANGED LOGIC)
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

    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
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

    mf = tp * df["volume"]
    df["mfi"] = 100 - (100 / (1 + mf.rolling(14).mean() / (mf.rolling(14).mean() + 1e-9)))

    rsi_min = df["rsi"].rolling(14).min()
    rsi_max = df["rsi"].rolling(14).max()
    df["stoch_rsi"] = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

    return df.dropna()


# =========================
# 🧠 FILTER (UNCHANGED)
# =========================
def smart_filter(df):

    if df is None or df.empty:
        return False

    if len(df) < 120:
        return False

    liquidity = (df["close"] * df["volume"]).mean()
    if liquidity < 1_000_000:
        return False

    return True


# =========================
# 📊 ADX (UNCHANGED)
# =========================
def calculate_adx(df, period=14):

    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    return dx.rolling(period).mean()


# =========================
# 📈 ANALYZE (UNCHANGED LOGIC)
# =========================
def analyze(df):

    latest = df.iloc[-1]
    score = 0
    reasons = []

    adx = calculate_adx(df).iloc[-1]

    if latest["rsi"] < 35:
        score += 10
        reasons.append("RSI Oversold")

    if latest["macd"] > latest["signal"]:
        score += 10
        reasons.append("MACD Bullish")

    if latest["ema50"] > latest["ema200"]:
        score += 10
        reasons.append("Uptrend EMA")

    if latest["volume"] > latest["vol_ma"]:
        score += 10
        reasons.append("Volume Breakout")

    if latest["atr"] > df["atr"].mean():
        score += 5
        reasons.append("Volatility Expansion")

    if adx > 20:
        score += 10
        reasons.append("ADX Trend Strength")

    if df["obv"].iloc[-1] > df["obv"].mean():
        score += 10
        reasons.append("OBV Strength")

    if latest["close"] > latest["vwap"]:
        score += 10
        reasons.append("Above VWAP")

    if latest["mfi"] < 50:
        score += 10
        reasons.append("Money Flow Entry")

    if latest["stoch_rsi"] < 0.2:
        score += 10
        reasons.append("Stoch Oversold")

    if latest["close"] <= latest["support"] * 1.02:
        score += 5
        reasons.append("Near Support")

    # FIXED SAFE SIGNAL
    if score >= 80:
        signal = "🔥 قوي جدًا"
    elif score >= 65:
        signal = "🟢 فرصة"
    elif score >= 50:
        signal = "⚠️ مراقبة"
    else:
        signal = "❌ ضعيف"

    return signal, score, reasons


# =========================
# 🎯 RISK (UNCHANGED)
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
# ⚙️ PROCESS STOCK (FIXED STABILITY)
# =========================
def process_stock(row):

    try:
        symbol = row["Symbol"]
        sector = row["Sector"]
        name = row["Name"]

        df = get_data(symbol)

        if df is None or df.empty:
            return None

        df = add_indicators(df)

        if df is None or df.empty:
            return None

        if not smart_filter(df):
            return None

        signal, score, reasons = analyze(df)

        if score < 50:
            return None

        risk = risk_management(df)
        if risk is None:
            return None

        entry, sl, tp1, tp2, tp3 = risk

        return {
            "Symbol": symbol,
            "Name": name,
            "Sector": sector,
            "Signal": signal,
            "Score": score,
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
# 🚀 MAIN ENGINE (FIXED THREADING)
# =========================
results = []

if st.button("🚀 SCAN EGX FULL MARKET"):

    stocks = get_all_stocks()

    progress = st.progress(0)

    with ThreadPoolExecutor(max_workers=8) as executor:

        futures = [
            executor.submit(process_stock, row)
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
        best_sector = df_res.sort_values("Score", ascending=False).groupby("Sector").head(1)

        st.success("🔥 SIGNALS GENERATED")

        st.subheader("🏆 Top 20 Stocks")
        st.dataframe(top20, use_container_width=True)

        st.subheader("📊 Best Per Sector")
        st.dataframe(best_sector, use_container_width=True)

        csv = df_res.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "egx_signals.csv", "text/csv")

    else:
        st.warning("❌ No signals found")
