# hammer.py
import io, os, math, time, datetime as dt
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------- Config ----------------
MIN_PRICE, MAX_PRICE = 10, 500
MIN_CAP, MAX_CAP = 2e9, 15e12
MIN_AVG_VOL = 500_000
BATCH_SIZE = 100
OUT_DIR = Path(".")

def _num(x, default=0):
    if x is None: return default
    try:
        f = float(x)
        return default if math.isnan(f) or math.isinf(f) else f
    except: return default

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def _fetch(url, **kw):
    r = requests.get(url, timeout=30, **kw)
    r.raise_for_status()
    return r.text

def get_us_tickers():
    txt = _fetch("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt")
    df = pd.read_csv(io.StringIO(txt), sep="|")
    df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
    syms = df["Symbol"].dropna().astype(str)
    return sorted({s for s in syms if s.isalpha() and s.isupper()})

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=3, max=20))
def _download(batch):
    return yf.download(batch, period="3mo", group_by="ticker", progress=False, threads=True, auto_adjust=True)

def main():
    print("Fetching symbol list...")
    universe = get_us_tickers()
    print(f"Universe: {len(universe)} tickers")

    keep = {}
    total_batches = (len(universe) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(universe), BATCH_SIZE):
        batch = universe[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"Processing batch {batch_num}/{total_batches} ({batch[0]} to {batch[-1]})...")
        
        try:
            data = _download(batch)
        except Exception:
            continue

        if data is None or data.empty: continue
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_product([[batch[0]], data.columns])

        for t in batch:
            if t not in data.columns.get_level_values(0): continue
            d = data[t].dropna()
            if len(d) < 40: continue
            close = float(d["Close"].iloc[-1])
            avgvol = float(d["Volume"].tail(20).mean())
            
            if not (MIN_PRICE <= close <= MAX_PRICE): continue
            if avgvol < MIN_AVG_VOL: continue
            
            keep[t] = d
        time.sleep(1.0)
        
    print(f"Passed price/volume: {len(keep)}")
    
    rows = []
    for t, d in keep.items():
        try:
            info = yf.Ticker(t).info or {}
            cap = _num(info.get("marketCap"))
            if not (MIN_CAP <= cap <= MAX_CAP): continue
            
            # --- THE HAMMER LOGIC ---
            close = d["Close"].iloc[-1]
            ma20 = d["Close"].tail(20).mean()
            ma50 = d["Close"].tail(50).mean()
            
            vol_today = d["Volume"].iloc[-1]
            vol_avg = d["Volume"].tail(20).mean()
            vol_ratio = vol_today / vol_avg
            
            price_1d = close / d["Close"].iloc[-2] - 1
            price_5d = close / d["Close"].iloc[-5] - 1
            
            # Breakout criteria: Above MAs, Massive Volume, Up Day
            if close > ma20 and close > ma50 and vol_ratio > 1.5 and price_1d > 0.02:
                rows.append({
                    "ticker": t,
                    "close": round(close, 2),
                    "1D %": round(price_1d * 100, 2),
                    "5D %": round(price_5d * 100, 2),
                    "Vol Ratio": round(vol_ratio, 2),
                    "Market Cap (M)": round(cap / 1e6),
                    "Sector": info.get("sector")
                })
        except Exception:
            pass

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    if not rows:
        print("No breakouts found today.")
        empty_df = pd.DataFrame([{"ticker": "NONE", "close": 0, "flags": "No momentum breakouts found."}])
        empty_df.to_csv(OUT_DIR / f"hammer_{stamp}_EMPTY.csv", index=False, sep=",")
        return

    out = pd.DataFrame(rows).sort_values("Vol Ratio", ascending=False)
    out.to_csv(OUT_DIR / f"hammer_{stamp}.csv", index=False, sep=",", encoding="utf-8")
    print("\nTop Breakouts:\n")
    print(out.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
