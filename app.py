
# EGX AI PRO MAX v5
# Streamlit + GitHub Ready
# Generated template with:
# VWAP, Relative Strength, Momentum Ranking,
# Position Sizing, Breakout Detection,
# Institutional Volume Filter, AI Score

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide")
st.title("🚀 EGX AI PRO MAX v5")

EGX = [
    "COMI.CA","MFPC.CA","PHDC.CA","ACRI.CA","ORAS.CA","HRHO.CA",
    "TMGH.CA","FWRY.CA","SWDY.CA","ETEL.CA","AMOC.CA","HELI.CA"
]

@st.cache_data(ttl=3600)
def load_data(symbols, period, interval):
    return yf.download(
        symbols,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        threads=True
    )

def add_indicators(df):
    df=df.copy()

    c=df["Close"]
    h=df["High"]
    l=df["Low"]
    v=df["Volume"]

    df["ema20"]=c.ewm(span=20).mean()
    df["ema50"]=c.ewm(span=50).mean()
    df["ema200"]=c.ewm(span=200).mean()

    df["rsi"]=ta.momentum.RSIIndicator(c).rsi()
    df["macd"]=ta.trend.MACD(c).macd()

    df["vol_ma"]=v.rolling(20).mean()

    df["support"]=l.rolling(20).min()
    df["resistance"]=h.rolling(20).max()

    df["obv"]=ta.volume.OnBalanceVolumeIndicator(c,v).on_balance_volume()

    df["vwap"]=((c*v).cumsum()/v.cumsum())

    return df

def atr(df,period=14):
    return ta.volatility.AverageTrueRange(
        df["High"],df["Low"],df["Close"],window=period
    ).average_true_range()

def adx(df,period=14):
    return ta.trend.ADXIndicator(
        df["High"],df["Low"],df["Close"],window=period
    ).adx()

def relative_strength(df):
    if len(df) < 65:
        return 0
    r1=(df["Close"].iloc[-1]/df["Close"].iloc[-21])-1
    r3=(df["Close"].iloc[-1]/df["Close"].iloc[-63])-1
    return (r1*0.3+r3*0.7)*100

def momentum_rank(df):
    if len(df) < 126:
        return 0
    r1=(df["Close"].iloc[-1]/df["Close"].iloc[-21])-1
    r3=(df["Close"].iloc[-1]/df["Close"].iloc[-63])-1
    r6=(df["Close"].iloc[-1]/df["Close"].iloc[-126])-1
    return round((r1*20+r3*35+r6*45)*100,2)

def analyze(df):
    df=add_indicators(df)

    if len(df)<220:
        return None

    last=df.iloc[-1]

    entry=float(last["Close"])
    atr_val=float(atr(df).iloc[-1])
    adx_val=float(adx(df).iloc[-1])

    rs=relative_strength(df)
    momentum=momentum_rank(df)

    score=0

    if last["Close"]>last["ema200"]: score+=15
    if last["ema20"]>last["ema50"]: score+=10
    if last["ema50"]>last["ema200"]: score+=10
    if last["macd"]>0: score+=10
    if last["rsi"]>50: score+=10

    if last["Volume"]>last["vol_ma"]*1.5:
        score+=10

    if last["Close"]>last["vwap"]:
        score+=10

    if adx_val>25:
        score+=10

    if rs>0:
        score+=5

    if momentum>0:
        score+=10

    breakout = last["Close"] > last["resistance"]*0.99

    score=min(100,score)

    stop=entry-(atr_val*2)

    tp1=entry+atr_val*1.5
    tp2=entry+atr_val*3
    tp3=entry+atr_val*5

    risk=entry-stop
    reward=tp2-entry

    rr=reward/risk if risk>0 else 0

    capital=100000
    risk_pct=0.01
    risk_amount=capital*risk_pct

    shares=int(risk_amount/risk) if risk>0 else 0

    signal="🔥 Strong Buy" if score>=85 else \
           "🟢 Buy" if score>=70 else \
           "⚠️ Watch"

    return {
        "Score":round(score,2),
        "Signal":signal,
        "Entry":round(entry,2),
        "SL":round(stop,2),
        "TP1":round(tp1,2),
        "TP2":round(tp2,2),
        "TP3":round(tp3,2),
        "RS":round(rs,2),
        "Momentum":round(momentum,2),
        "ADX":round(adx_val,2),
        "RR":round(rr,2),
        "PositionSize":shares,
        "Breakout":"YES" if breakout else "NO"
    }

def process(symbol,data):
    try:
        df=data[symbol].dropna()
        r=analyze(df)
        if r:
            r["Symbol"]=symbol.replace(".CA","")
        return r
    except:
        return None

if st.button("RUN SCANNER"):
    data=load_data(EGX,"2y","1d")

    results=[]

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures=[ex.submit(process,s,data) for s in EGX]

        for f in futures:
            x=f.result()
            if x:
                results.append(x)

    if results:
        df=pd.DataFrame(results)
        df=df.sort_values(["Score","Momentum"],ascending=False)

        st.dataframe(df,use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "egx_ai_pro_max_v5.csv"
        )
    else:
        st.warning("No data found")
