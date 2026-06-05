import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🚀 EGX SMART SCANNER PRO MAX (RANKING ALWAYS ON)")


# =========================
# 📦 LOAD STOCKS
# =========================
@st.cache_data(ttl=86400)
def get_all_stocks():
    df = pd.read_csv("egx_symbols.csv")

    df.columns = df.columns.str.strip().str.lower()

    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"
    if "name" not in df.columns:
        df["name"] = "UNKNOWN"
    if "sector" not in df.columns:
        df["sector"] = "UNKNOWN"

    return df


# =========================
# 🧠 MARKET STATUS
# =========================
def is_market_open():

    now = datetime.now()

    if now.weekday() >= 5:
        return False

    if 10 <= now.hour < 14:
        return True

    return False


# =========================
# 📊 DATA SOURCE (MULTI)
# =========================
def get_data(symbol):

    try:
        df = pd.read_csv(f"data/{symbol}.csv")

        if df is not None and len(df) > 20:
            df.columns = [c.lower() for c in df.columns]

            df = df.rename(columns={
                "adj close": "close",
                "date": "datetime",
                "time": "datetime"
            })

            return df
    except:
        pass

    try:
        df = yf.download(f"{symbol}.CA", period="1y", interval="1d", progress=False)

        if df is not None and not df.empty:
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            return df
    except:
        pass

    return None


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

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()

    return df


# =========================
# 🧠 FILTER
# =========================
def smart_filter(df):

    if df is None or df.empty:
        return False

    if len(df) < 30:
        return False

    return True


# =========================
# 📊 ADX
# =========================
def calculate_adx(df):

    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()

    plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100

    return dx.rolling(14).mean()


# =========================
# 📈 ANALYZE
# =========================
def analyze(df, market_open):

    latest = df.iloc[-1]
    score = 0
    reasons = []

    adx = calculate_adx(df).iloc[-1]

    price_trend = latest["close"] / df["close"].iloc[-5] - 1

    if price_trend > 0:
        score += 15
        reasons.append("Short Uptrend")

    if latest["close"] <= latest["support"] * 1.06:
        score += 15
        reasons.append("Near Support")

    if latest["rsi"] < 50:
        score += 10
        reasons.append("RSI OK")

    if latest["macd"] > latest["signal"]:
        score += 10
        reasons.append("MACD Positive")

    if latest["close"] > latest["vwap"]:
        score += 10
        reasons.append("Above VWAP")

    if latest["volume"] > df["volume"].mean() * 0.7:
        score += 10
        reasons.append("Volume Active")

    if adx > 15:
        score += 10
        reasons.append("Trend Exists")

    if df["obv"].iloc[-1] > df["obv"].mean():
        score += 10
        reasons.append("OBV Strength")

    score += 5 if market_open else 8

    if score >= 70:
        signal = "🔥 فرصة قوية"
    elif score >= 55:
        signal = "🟢 فرصة"
    elif score >= 40:
        signal = "⚠️ متابعة"
    else:
        signal = "🟡 ضعيف"

    return signal, score, reasons


# =========================
# 🎯 RISK MANAGEMENT
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
def process_stock(row, market_open):

    try:
        symbol = row["symbol"]
        name = row["name"]
        sector = row["sector"]

        df = get_data(symbol)

        if df is None:
            return None

        df = add_indicators(df)

        if not smart_filter(df):
            return None

        signal, score, reasons = analyze(df, market_open)

        if score < 25:
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

stocks = get_all_stocks()   # ✅ FIX: تعريف ثابت خارج الزر
market_open = is_market_open()

if st.button("🚀 SCAN EGX RANKING ENGINE"):

    st.info("🟢 Market Open Mode" if market_open else "🔵 Market Closed Mode")

    progress = st.progress(0)

    with ThreadPoolExecutor(max_workers=8) as executor:

        futures = [
            executor.submit(process_stock, row, market_open)
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


# =========================
# 📊 RANKING ALWAYS ON
# =========================

if results:
    df_res = pd.DataFrame(results)
else:
    st.warning("⚠️ No strong signals → switching to MARKET RANKING MODE")

    fallback = []

    for row in stocks.to_dict("records")[:80]:

        try:
            df = get_data(row["symbol"])
            if df is None:
                continue

            df = add_indicators(df)

            signal, score, reasons = analyze(df, False)

            fallback.append({
                "Symbol": row["symbol"],
                "Name": row["name"],
                "Sector": row["sector"],
                "Score": float(score),
                "Signal": signal,
                "Reasons": ", ".join(reasons) if isinstance(reasons, list) else ""
            })

        except:
            continue

    df_res = pd.DataFrame(fallback)


# =========================
# 🏆 OUTPUT ALWAYS
# =========================

if df_res is not None and not df_res.empty:

    top20 = df_res.sort_values("Score", ascending=False).head(20)
    best_sector = df_res.sort_values("Score", ascending=False).groupby("Sector").head(1)

    st.success("🔥 MARKET RANKING READY")

    st.subheader("🏆 Top 20 Stocks")
    st.dataframe(top20, use_container_width=True)

    st.subheader("📊 Best Per Sector")
    st.dataframe(best_sector, use_container_width=True)

    csv = df_res.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "egx_ranking.csv", "text/csv")

else:
    st.error("❌ No data available")
