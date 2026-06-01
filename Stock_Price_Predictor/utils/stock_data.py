import math
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─── NSE STOCK UNIVERSE ──────────────────────────────────────────────────────

NSE_STOCKS = {
    # Nifty 50
    "RELIANCE.NS":  {"name": "Reliance Industries",    "sector": "Energy",          "index": "Nifty 50",       "base": 2850},
    "TCS.NS":       {"name": "Tata Consultancy Svcs",  "sector": "IT",              "index": "Nifty 50",       "base": 3900},
    "HDFCBANK.NS":  {"name": "HDFC Bank",              "sector": "Banking",         "index": "Nifty 50",       "base": 1680},
    "INFY.NS":      {"name": "Infosys",                "sector": "IT",              "index": "Nifty 50",       "base": 1420},
    "ICICIBANK.NS": {"name": "ICICI Bank",             "sector": "Banking",         "index": "Nifty 50",       "base": 1280},
    "SBIN.NS":      {"name": "State Bank of India",    "sector": "Banking",         "index": "Nifty 50",       "base": 780},
    "BHARTIARTL.NS":{"name": "Bharti Airtel",          "sector": "Telecom",         "index": "Nifty 50",       "base": 1750},
    "WIPRO.NS":     {"name": "Wipro",                  "sector": "IT",              "index": "Nifty 50",       "base": 480},
    "HCLTECH.NS":   {"name": "HCL Technologies",       "sector": "IT",              "index": "Nifty 50",       "base": 1600},
    "LT.NS":        {"name": "Larsen & Toubro",        "sector": "Engineering",     "index": "Nifty 50",       "base": 3500},
    "ASIANPAINT.NS":{"name": "Asian Paints",           "sector": "Consumer",        "index": "Nifty 50",       "base": 2300},
    "MARUTI.NS":    {"name": "Maruti Suzuki",          "sector": "Auto",            "index": "Nifty 50",       "base": 12400},
    "BAJFINANCE.NS":{"name": "Bajaj Finance",          "sector": "NBFC",            "index": "Nifty 50",       "base": 7200},
    "TITAN.NS":     {"name": "Titan Company",          "sector": "Consumer",        "index": "Nifty 50",       "base": 3300},
    "SUNPHARMA.NS": {"name": "Sun Pharmaceutical",     "sector": "Pharma",          "index": "Nifty 50",       "base": 1650},
    "ULTRACEMCO.NS":{"name": "UltraTech Cement",       "sector": "Cement",          "index": "Nifty 50",       "base": 10800},
    "NTPC.NS":      {"name": "NTPC",                   "sector": "Power",           "index": "Nifty 50",       "base": 360},
    "POWERGRID.NS": {"name": "Power Grid Corp",        "sector": "Power",           "index": "Nifty 50",       "base": 295},
    "ONGC.NS":      {"name": "ONGC",                   "sector": "Energy",          "index": "Nifty 50",       "base": 275},
    "COALINDIA.NS": {"name": "Coal India",             "sector": "Mining",          "index": "Nifty 50",       "base": 450},
    # Nifty Bank
    "AXISBANK.NS":  {"name": "Axis Bank",              "sector": "Banking",         "index": "Nifty Bank",     "base": 1180},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank",    "sector": "Banking",         "index": "Nifty Bank",     "base": 1980},
    "INDUSINDBK.NS":{"name": "IndusInd Bank",          "sector": "Banking",         "index": "Nifty Bank",     "base": 780},
    "BANKBARODA.NS":{"name": "Bank of Baroda",         "sector": "Banking",         "index": "Nifty Bank",     "base": 245},
    "PNB.NS":       {"name": "Punjab National Bank",   "sector": "Banking",         "index": "Nifty Bank",     "base": 105},
    "FEDERALBNK.NS":{"name": "Federal Bank",           "sector": "Banking",         "index": "Nifty Bank",     "base": 185},
    "IDFCFIRSTB.NS":{"name": "IDFC First Bank",        "sector": "Banking",         "index": "Nifty Bank",     "base": 65},
    # Nifty Next 50
    "ADANIPORTS.NS":{"name": "Adani Ports",            "sector": "Infrastructure",  "index": "Nifty Next 50",  "base": 1350},
    "ADANIGREEN.NS":{"name": "Adani Green Energy",     "sector": "Renewables",      "index": "Nifty Next 50",  "base": 1100},
    "DLF.NS":       {"name": "DLF",                    "sector": "Real Estate",     "index": "Nifty Next 50",  "base": 820},
    "TATAPOWER.NS": {"name": "Tata Power",             "sector": "Power",           "index": "Nifty Next 50",  "base": 395},
    "DMART.NS":     {"name": "Avenue Supermarts",      "sector": "Retail",          "index": "Nifty Next 50",  "base": 4200},
    "DRREDDY.NS":   {"name": "Dr. Reddy's Labs",       "sector": "Pharma",          "index": "Nifty Next 50",  "base": 1280},
    "CIPLA.NS":     {"name": "Cipla",                  "sector": "Pharma",          "index": "Nifty Next 50",  "base": 1520},
    "HINDALCO.NS":  {"name": "Hindalco Industries",    "sector": "Metals",          "index": "Nifty Next 50",  "base": 670},
    "TATASTEEL.NS": {"name": "Tata Steel",             "sector": "Metals",          "index": "Nifty Next 50",  "base": 145},
    "BRITANNIA.NS": {"name": "Britannia Industries",   "sector": "FMCG",            "index": "Nifty Next 50",  "base": 5300},
    "ITC.NS":       {"name": "ITC",                    "sector": "FMCG",            "index": "Nifty 50",       "base": 450},
    "HINDUNILVR.NS":{"name": "Hindustan Unilever",     "sector": "FMCG",            "index": "Nifty 50",       "base": 2250},
    "NESTLEIND.NS": {"name": "Nestle India",           "sector": "FMCG",            "index": "Nifty Next 50",  "base": 2250},
    "BAJAJFINSV.NS":{"name": "Bajaj Finserv",          "sector": "NBFC",            "index": "Nifty 50",       "base": 1680},
    "TECHM.NS":     {"name": "Tech Mahindra",          "sector": "IT",              "index": "Nifty 50",       "base": 1550},
    "EICHERMOT.NS": {"name": "Eicher Motors",          "sector": "Auto",            "index": "Nifty 50",       "base": 5100},
    "TATAMOTORS.NS":{"name": "Tata Motors",            "sector": "Auto",            "index": "Nifty Next 50",  "base": 690},
    "M&M.NS":       {"name": "Mahindra & Mahindra",    "sector": "Auto",            "index": "Nifty 50",       "base": 2950},
    "JSWSTEEL.NS":  {"name": "JSW Steel",              "sector": "Metals",          "index": "Nifty 50",       "base": 910},
    "GRASIM.NS":    {"name": "Grasim Industries",      "sector": "Cement",          "index": "Nifty 50",       "base": 2650},
}

SECTOR_MAP = {}
for sym, info in NSE_STOCKS.items():
    s = info["sector"]
    if s not in SECTOR_MAP:
        SECTOR_MAP[s] = []
    SECTOR_MAP[s].append(sym)

# ─── PRICE SIMULATION ENGINE ──────────────────────────────────────────────────

def _seed(symbol):
    """Deterministic seed per symbol for reproducible but 'live-like' data."""
    return abs(hash(symbol)) % (2**31)

def simulate_history(symbol, days=180):
    """Generate realistic OHLCV history for a symbol."""
    info = NSE_STOCKS.get(symbol, {"base": 1000})
    base = info["base"]
    rng = random.Random(_seed(symbol) + int(datetime.now().date().toordinal()))

    prices = [base]
    trend = rng.gauss(0.0003, 0.0005)   # slight upward drift

    for i in range(days - 1):
        shock = rng.gauss(0, 0.012)
        mean_rev = (base - prices[-1]) * 0.005
        new_p = prices[-1] * (1 + trend + shock + mean_rev / prices[-1])
        new_p = max(new_p, base * 0.4)
        prices.append(round(new_p, 2))

    dates = [datetime.now().date() - timedelta(days=days - 1 - i) for i in range(days)]
    # Remove weekends
    business_dates = [d for d in dates if d.weekday() < 5]
    prices = prices[:len(business_dates)]

    rows = []
    for i, (d, close) in enumerate(zip(business_dates, prices)):
        open_p  = prices[i-1] if i > 0 else close * rng.uniform(0.99, 1.01)
        high_p  = close * rng.uniform(1.001, 1.025)
        low_p   = close * rng.uniform(0.975, 0.999)
        high_p  = max(high_p, open_p, close)
        low_p   = min(low_p, open_p, close)
        volume  = int(rng.uniform(500_000, 8_000_000))
        rows.append({
            "Date": d, "Open": round(open_p,2), "High": round(high_p,2),
            "Low": round(low_p,2), "Close": round(close,2), "Volume": volume
        })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df

def get_stock_history(symbol, period="6mo"):
    """
    ── LIVE DATA SWITCH ──
    Uncomment below to use real yfinance data when internet is available:

        import yfinance as yf
        ticker = yf.Ticker(symbol)
        period_map = {"1mo":30, "3mo":90, "6mo":180, "1y":365}
        df = ticker.history(period=period)
        if not df.empty:
            return df[["Open","High","Low","Close","Volume"]]

    Currently using simulation for offline/development use.
    """
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
    days = days_map.get(period, 180)
    return simulate_history(symbol, days)

def get_live_quote(symbol):
    """
    Returns current price snapshot.
    ── LIVE DATA SWITCH ──
        import yfinance as yf
        t = yf.Ticker(symbol)
        info = t.fast_info
        return {"price": info.last_price, "change": info.regular_market_change, ...}
    """
    info = NSE_STOCKS.get(symbol, {"base": 1000, "name": symbol})
    df = simulate_history(symbol, 5)
    if df.empty:
        return {}
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    change = round(latest["Close"] - prev["Close"], 2)
    pct    = round((change / prev["Close"]) * 100, 2) if prev["Close"] else 0
    rng = random.Random(_seed(symbol) + int(datetime.now().timestamp() // 30))
    live_price = round(latest["Close"] * rng.uniform(0.998, 1.002), 2)
    return {
        "symbol":    symbol,
        "name":      info.get("name", symbol),
        "sector":    info.get("sector", "—"),
        "index":     info.get("index", "—"),
        "price":     live_price,
        "open":      round(float(latest["Open"]), 2),
        "high":      round(float(latest["High"]), 2),
        "low":       round(float(latest["Low"]), 2),
        "prev_close":round(float(prev["Close"]), 2),
        "change":    change,
        "pct_change":pct,
        "volume":    int(latest["Volume"]),
        "52w_high":  round(float(df["High"].max()), 2),
        "52w_low":   round(float(df["Low"].min()), 2),
    }

def get_all_quotes():
    return [get_live_quote(sym) for sym in NSE_STOCKS]

def get_market_indices():
    """Simulated index values."""
    rng = random.Random(int(datetime.now().timestamp() // 60))
    return {
        "NIFTY 50":      {"value": round(22680 + rng.uniform(-300, 300), 2), "change": round(rng.uniform(-1.5, 1.5), 2)},
        "NIFTY BANK":    {"value": round(48200 + rng.uniform(-600, 600), 2), "change": round(rng.uniform(-1.5, 1.5), 2)},
        "NIFTY IT":      {"value": round(33800 + rng.uniform(-500, 500), 2), "change": round(rng.uniform(-2, 2), 2)},
        "SENSEX":        {"value": round(74600 + rng.uniform(-800, 800), 2), "change": round(rng.uniform(-1.5, 1.5), 2)},
        "NIFTY MIDCAP":  {"value": round(44200 + rng.uniform(-400, 400), 2), "change": round(rng.uniform(-2, 2), 2)},
        "NIFTY PHARMA":  {"value": round(18900 + rng.uniform(-300, 300), 2), "change": round(rng.uniform(-1.5, 1.5), 2)},
    }

def search_stocks(query):
    q = query.lower()
    results = []
    for sym, info in NSE_STOCKS.items():
        if q in sym.lower() or q in info["name"].lower() or q in info["sector"].lower():
            results.append({"symbol": sym, "name": info["name"], "sector": info["sector"]})
    return results[:12]
