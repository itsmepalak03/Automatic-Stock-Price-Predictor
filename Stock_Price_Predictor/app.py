"""
Automatic Stock Price Predictor — Flask Application
====================================================
Run:  python app.py
Open: http://localhost:5000
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, jsonify, request, session
import json, random
from datetime import datetime, timedelta

from utils.stock_data import (
    get_stock_history, get_live_quote, get_all_quotes,
    get_market_indices, search_stocks, NSE_STOCKS, SECTOR_MAP
)
from models.predictor import predict_next_days, add_technical_indicators

app = Flask(__name__)
app.secret_key = "aspp-live-2026-secret"

# ── Jinja2 global helpers ─────────────────────────────────────────────────────
def _fmt_inr(n, d=2):
    if n is None: return "—"
    try: return "{:,.{p}f}".format(float(n), p=d)
    except: return str(n)

def _fmt_vol(n):
    if n is None: return "—"
    try:
        n = float(n)
        if n >= 1e7: return f"{n/1e7:.2f}Cr"
        if n >= 1e5: return f"{n/1e5:.1f}L"
        return f"{int(n):,}"
    except: return str(n)

app.jinja_env.globals.update(fmtINR=_fmt_inr, fmtVol=_fmt_vol)

# ── in-memory watchlist & portfolio (session-based) ───────────────────────────
DEFAULT_WATCHLIST  = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","SBIN.NS"]
DEFAULT_PORTFOLIO  = [
    {"symbol":"RELIANCE.NS","qty":10,"buy_price":2780.0},
    {"symbol":"TCS.NS",     "qty":5, "buy_price":3850.0},
    {"symbol":"INFY.NS",    "qty":20,"buy_price":1380.0},
    {"symbol":"HDFCBANK.NS","qty":15,"buy_price":1620.0},
]

def get_watchlist():
    return session.get("watchlist", list(DEFAULT_WATCHLIST))

def get_portfolio():
    return session.get("portfolio", list(DEFAULT_PORTFOLIO))

# ── HELPERS ───────────────────────────────────────────────────────────────────
def candlestick_data(df):
    rows = []
    for date, row in df.iterrows():
        rows.append({
            "x": str(date.date()),
            "open":  round(float(row["Open"]),  2),
            "high":  round(float(row["High"]),  2),
            "low":   round(float(row["Low"]),   2),
            "close": round(float(row["Close"]), 2),
        })
    return rows

def volume_data(df):
    rows = []
    for date, row in df.iterrows():
        rows.append({"x": str(date.date()), "y": int(row["Volume"]),
                     "color": "#22c55e" if row["Close"] >= row["Open"] else "#ef4444"})
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    indices  = get_market_indices()
    quotes   = get_all_quotes()
    gainers  = sorted([q for q in quotes if q["pct_change"] > 0], key=lambda x: -x["pct_change"])[:5]
    losers   = sorted([q for q in quotes if q["pct_change"] < 0], key=lambda x:  x["pct_change"])[:5]
    featured = quotes[:8]
    return render_template("home.html", indices=indices, gainers=gainers,
                           losers=losers, featured=featured,
                           total_stocks=len(NSE_STOCKS))

@app.route("/dashboard")
def dashboard():
    quotes  = get_all_quotes()
    indices = get_market_indices()
    sectors = {}
    for sym, info in NSE_STOCKS.items():
        s = info["sector"]
        if s not in sectors:
            sectors[s] = {"count": 0, "gainers": 0, "losers": 0}
        sectors[s]["count"] += 1
    for q in quotes:
        sec = NSE_STOCKS.get(q["symbol"], {}).get("sector", "—")
        if sec in sectors:
            if q["pct_change"] > 0: sectors[sec]["gainers"] += 1
            elif q["pct_change"] < 0: sectors[sec]["losers"] += 1
    return render_template("dashboard.html", quotes=quotes, indices=indices,
                           sectors=sectors, total=len(quotes))

@app.route("/stocks")
def stocks():
    quotes  = get_all_quotes()
    indices_list = sorted(set(info["index"] for info in NSE_STOCKS.values()))
    sectors_list = sorted(set(info["sector"] for info in NSE_STOCKS.values()))
    return render_template("stocks.html", quotes=quotes,
                           indices_list=indices_list, sectors_list=sectors_list)

@app.route("/stock/<symbol>")
def stock_detail(symbol):
    symbol = symbol.upper()
    if symbol not in NSE_STOCKS:
        return render_template("404.html"), 404
    quote = get_live_quote(symbol)
    df    = get_stock_history(symbol, "6mo")
    df    = add_technical_indicators(df)
    candles = candlestick_data(df.tail(90))
    vols    = volume_data(df.tail(90))
    line_dates  = [str(d.date()) for d in df.tail(90).index]
    line_close  = [round(float(v),2) for v in df["Close"].tail(90)]
    sma20 = [round(float(v),2) if not __import__("math").isnan(v) else None for v in df["SMA_20"].tail(90)]
    sma5  = [round(float(v),2) if not __import__("math").isnan(v) else None for v in df["SMA_5"].tail(90)]
    bb_up = [round(float(v),2) if not __import__("math").isnan(v) else None for v in df["BB_upper"].tail(90)]
    bb_dn = [round(float(v),2) if not __import__("math").isnan(v) else None for v in df["BB_lower"].tail(90)]
    rsi   = [round(float(v),2) if not __import__("math").isnan(v) else None for v in df["RSI"].tail(90)]
    macd  = [round(float(v),4) if not __import__("math").isnan(v) else None for v in df["MACD"].tail(90)]
    macd_s= [round(float(v),4) if not __import__("math").isnan(v) else None for v in df["MACD_signal"].tail(90)]
    in_wl = symbol in get_watchlist()
    related = [s for s, i in NSE_STOCKS.items()
               if i["sector"]==NSE_STOCKS[symbol]["sector"] and s!=symbol][:4]
    related_quotes = [get_live_quote(s) for s in related]
    return render_template("stock_detail.html", quote=quote, symbol=symbol,
        info=NSE_STOCKS[symbol], candles=json.dumps(candles), vols=json.dumps(vols),
        line_dates=json.dumps(line_dates), line_close=json.dumps(line_close),
        sma20=json.dumps(sma20), sma5=json.dumps(sma5),
        bb_upper=json.dumps(bb_up), bb_lower=json.dumps(bb_dn),
        rsi_data=json.dumps(rsi), macd_data=json.dumps(macd),
        macd_signal=json.dumps(macd_s),
        in_watchlist=in_wl, related=related_quotes)

@app.route("/predict", methods=["GET"])
def predict_page():
    symbol = request.args.get("symbol", "RELIANCE.NS").upper()
    if symbol not in NSE_STOCKS:
        symbol = "RELIANCE.NS"
    all_symbols = list(NSE_STOCKS.keys())
    return render_template("predict.html", symbol=symbol,
                           all_symbols=all_symbols, nse_stocks=NSE_STOCKS)

@app.route("/compare")
def compare():
    syms = request.args.getlist("s")
    if not syms:
        syms = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS"]
    syms = [s for s in syms if s in NSE_STOCKS][:4]
    all_symbols = list(NSE_STOCKS.keys())
    return render_template("compare.html", symbols=syms, all_symbols=all_symbols, nse_stocks=NSE_STOCKS)

@app.route("/market")
def market():
    quotes  = get_all_quotes()
    indices = get_market_indices()
    sectors = {}
    for sym, info in NSE_STOCKS.items():
        s = info["sector"]
        if s not in sectors:
            sectors[s] = []
        sectors[s].append(sym)
    return render_template("market.html", quotes=quotes, indices=indices, sectors=sectors)

@app.route("/portfolio")
def portfolio():
    port = get_portfolio()
    holdings = []
    total_invested = total_current = 0
    for h in port:
        q = get_live_quote(h["symbol"])
        if q:
            curr_val = q["price"] * h["qty"]
            inv_val  = h["buy_price"] * h["qty"]
            pnl      = curr_val - inv_val
            pnl_pct  = (pnl / inv_val * 100) if inv_val else 0
            holdings.append({**h, **q, "curr_val": round(curr_val,2),
                             "inv_val": round(inv_val,2), "pnl": round(pnl,2),
                             "pnl_pct": round(pnl_pct,2)})
            total_invested += inv_val
            total_current  += curr_val
    total_pnl     = round(total_current - total_invested, 2)
    total_pnl_pct = round((total_pnl / total_invested * 100) if total_invested else 0, 2)
    all_symbols   = list(NSE_STOCKS.keys())
    return render_template("portfolio.html", holdings=holdings,
        total_invested=round(total_invested,2),
        total_current=round(total_current,2),
        total_pnl=total_pnl, total_pnl_pct=total_pnl_pct,
        all_symbols=all_symbols, nse_stocks=NSE_STOCKS)

@app.route("/watchlist")
def watchlist():
    wl     = get_watchlist()
    quotes = [get_live_quote(s) for s in wl]
    quotes = [q for q in quotes if q]
    all_symbols = list(NSE_STOCKS.keys())
    return render_template("watchlist.html", quotes=quotes,
                           all_symbols=all_symbols, nse_stocks=NSE_STOCKS)

@app.route("/news")
def news():
    DUMMY_NEWS = [
        {"title":"Sensex gains 400 pts; Nifty crosses 22,700 mark","source":"Economic Times","time":"2 hours ago","tag":"Market","sentiment":"positive"},
        {"title":"RBI holds repo rate steady at 6.5% in MPC meeting","source":"Business Standard","time":"4 hours ago","tag":"Economy","sentiment":"neutral"},
        {"title":"Reliance Industries reports record Q3 revenue of ₹2.5 lakh crore","source":"CNBC TV18","time":"6 hours ago","tag":"Results","sentiment":"positive"},
        {"title":"IT sector under pressure; TCS, Infosys slip on weak guidance","source":"Mint","time":"8 hours ago","tag":"IT","sentiment":"negative"},
        {"title":"FII inflows: Foreign investors pour ₹12,000 crore in January","source":"Financial Express","time":"10 hours ago","tag":"FII","sentiment":"positive"},
        {"title":"Gold prices surge amid global uncertainty, cross ₹65,000/10g","source":"Moneycontrol","time":"12 hours ago","tag":"Commodities","sentiment":"neutral"},
        {"title":"HDFC Bank Q3 profit jumps 33%, NPA improves to 1.26%","source":"Economic Times","time":"1 day ago","tag":"Banking","sentiment":"positive"},
        {"title":"Auto sales data: Maruti leads, passenger vehicles up 8% YoY","source":"Business Today","time":"1 day ago","tag":"Auto","sentiment":"positive"},
        {"title":"Crude oil slips below $80; OMCs like BPCL, IOC rally","source":"CNBC TV18","time":"1 day ago","tag":"Energy","sentiment":"positive"},
        {"title":"SEBI tightens F&O regulations; new margin rules from April","source":"Mint","time":"2 days ago","tag":"Regulatory","sentiment":"neutral"},
        {"title":"Sun Pharma gets USFDA nod for key oncology drug","source":"PTI","time":"2 days ago","tag":"Pharma","sentiment":"positive"},
        {"title":"Power sector stocks surge on strong demand outlook","source":"Economic Times","time":"2 days ago","tag":"Power","sentiment":"positive"},
    ]
    return render_template("news.html", news=DUMMY_NEWS)

@app.route("/about")
def about():
    return render_template("about.html", total_stocks=len(NSE_STOCKS),
                           sectors=list(SECTOR_MAP.keys()))

@app.route("/contact")
def contact():
    return render_template("contact.html")


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES (JSON)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/quotes")
def api_quotes():
    return jsonify(get_all_quotes())

@app.route("/api/quote/<symbol>")
def api_quote(symbol):
    return jsonify(get_live_quote(symbol.upper()))

@app.route("/api/indices")
def api_indices():
    return jsonify(get_market_indices())

@app.route("/api/history/<symbol>")
def api_history(symbol):
    period = request.args.get("period", "6mo")
    df = get_stock_history(symbol.upper(), period)
    return jsonify({
        "candles": candlestick_data(df.tail(120)),
        "volume":  volume_data(df.tail(120)),
    })

@app.route("/api/predict/<symbol>")
def api_predict(symbol):
    symbol = symbol.upper()
    n_days = int(request.args.get("days", 10))
    df = get_stock_history(symbol, "6mo")
    result = predict_next_days(df, n_days)
    if not result:
        return jsonify({"error": "Not enough data"}), 400
    return jsonify(result)

@app.route("/api/compare")
def api_compare():
    syms = request.args.getlist("s")
    result = {}
    for s in syms:
        s = s.upper()
        if s in NSE_STOCKS:
            df = get_stock_history(s, "3mo")
            closes = [round(float(v),2) for v in df["Close"]]
            dates  = [str(d.date()) for d in df.index]
            pct    = [round((c - closes[0])/closes[0]*100, 2) for c in closes]
            result[s] = {"dates": dates, "close": closes, "pct": pct,
                         "name": NSE_STOCKS[s]["name"]}
    return jsonify(result)

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "")
    return jsonify(search_stocks(q))

@app.route("/api/watchlist/add", methods=["POST"])
def wl_add():
    sym = request.json.get("symbol","").upper()
    wl = get_watchlist()
    if sym in NSE_STOCKS and sym not in wl:
        wl.append(sym)
        session["watchlist"] = wl
    return jsonify({"watchlist": wl})

@app.route("/api/watchlist/remove", methods=["POST"])
def wl_remove():
    sym = request.json.get("symbol","").upper()
    wl = get_watchlist()
    wl = [s for s in wl if s != sym]
    session["watchlist"] = wl
    return jsonify({"watchlist": wl})

@app.route("/api/portfolio/add", methods=["POST"])
def port_add():
    data = request.json
    sym  = data.get("symbol","").upper()
    qty  = float(data.get("qty", 1))
    bp   = float(data.get("buy_price", 0))
    port = get_portfolio()
    existing = next((h for h in port if h["symbol"]==sym), None)
    if existing:
        existing["qty"] += qty
    else:
        port.append({"symbol":sym,"qty":qty,"buy_price":bp})
    session["portfolio"] = port
    return jsonify({"ok": True})

@app.route("/api/portfolio/remove", methods=["POST"])
def port_remove():
    sym  = request.json.get("symbol","").upper()
    port = [h for h in get_portfolio() if h["symbol"]!=sym]
    session["portfolio"] = port
    return jsonify({"ok": True})

@app.route("/api/sector_perf")
def api_sector_perf():
    quotes = get_all_quotes()
    perf = {}
    for q in quotes:
        sec = NSE_STOCKS.get(q["symbol"],{}).get("sector","—")
        if sec not in perf:
            perf[sec] = []
        perf[sec].append(q["pct_change"])
    return jsonify({s: round(sum(v)/len(v),2) for s,v in perf.items()})

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
