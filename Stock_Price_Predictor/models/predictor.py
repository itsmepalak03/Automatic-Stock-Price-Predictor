"""
ML Prediction Engine
====================
Uses scikit-learn Linear Regression with feature engineering.
Architecture is LSTM-ready — swap model in train_model() to use LSTM
when tensorflow/keras is available.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def add_technical_indicators(df):
    """Add SMA, EMA, RSI, Bollinger Bands, MACD to dataframe."""
    close = df["Close"].values.astype(float)
    n = len(close)

    # SMA
    for w in [5, 10, 20, 50]:
        df[f"SMA_{w}"] = pd.Series(close, index=df.index).rolling(w).mean()

    # EMA
    for w in [12, 26]:
        df[f"EMA_{w}"] = pd.Series(close, index=df.index).ewm(span=w, adjust=False).mean()

    # RSI (14)
    delta = pd.Series(close, index=df.index).diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float('nan'))
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = pd.Series(close, index=df.index).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close, index=df.index).ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2σ)
    sma20 = pd.Series(close, index=df.index).rolling(20).mean()
    std20 = pd.Series(close, index=df.index).rolling(20).std()
    df["BB_upper"] = (sma20 + 2 * std20)
    df["BB_lower"] = (sma20 - 2 * std20)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma20.replace(0, float('nan'))

    # Price momentum
    df["Return_1d"] = pd.Series(close, index=df.index).pct_change(1)
    df["Return_5d"] = pd.Series(close, index=df.index).pct_change(5)
    df["Volatility_10d"] = pd.Series(close, index=df.index).rolling(10).std()

    return df


def build_features(df):
    """Create feature matrix for regression."""
    df = df.copy()
    df = add_technical_indicators(df)
    df = df.dropna()

    feature_cols = [
        "Open", "High", "Low", "Volume",
        "SMA_5", "SMA_10", "SMA_20", "EMA_12", "EMA_26",
        "RSI", "MACD", "BB_width",
        "Return_1d", "Return_5d", "Volatility_10d"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].values
    y = df["Close"].values
    return X, y, df, feature_cols


def predict_next_days(df, n_days=10):
    """
    Train Linear Regression on historical data.
    Returns dict with predictions, confidence bands, signals, indicators.

    ── LSTM UPGRADE PATH ──
    Replace the LinearRegression block below with:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        model = Sequential([LSTM(64, return_sequences=True, input_shape=(60,1)),
                            Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        # reshape X to (samples, timesteps, features) for LSTM
    """
    if len(df) < 30:
        return None

    X, y, df_feat, feature_cols = build_features(df)
    if len(X) < 10:
        return None

    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    model = LinearRegression()
    model.fit(X_sc, y_sc)

    # In-sample predictions (last 30 days for chart overlay)
    in_sample_pred = scaler_y.inverse_transform(model.predict(X_sc).reshape(-1,1)).ravel()

    # Future predictions — walk-forward
    last_row = df_feat.iloc[-1].copy()
    close_history = list(df["Close"].values[-50:])
    future_prices = []
    future_dates  = []
    last_date = df_feat.index[-1]

    for i in range(n_days):
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        future_dates.append(next_date)
        last_date = next_date

        # Build synthetic next row
        prev_close = close_history[-1]
        # Use last known feature ratios
        open_p  = prev_close * np.random.uniform(0.995, 1.005)
        high_p  = prev_close * np.random.uniform(1.002, 1.020)
        low_p   = prev_close * np.random.uniform(0.980, 0.998)
        vol     = float(last_row.get("Volume", 1_000_000))

        close_series = pd.Series(close_history)
        sma5  = close_series.rolling(5).mean().iloc[-1]
        sma10 = close_series.rolling(10).mean().iloc[-1]
        sma20 = close_series.rolling(20).mean().iloc[-1]
        ema12 = close_series.ewm(span=12,adjust=False).mean().iloc[-1]
        ema26 = close_series.ewm(span=26,adjust=False).mean().iloc[-1]

        delta = close_series.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rsi  = 100 - (100 / (1 + gain / (loss + 1e-9)))

        macd = ema12 - ema26
        sma20_std = close_series.rolling(20).std().iloc[-1]
        bb_width = (4 * sma20_std) / (sma20 + 1e-9)
        ret1  = (close_series.pct_change(1).iloc[-1])
        ret5  = (close_series.pct_change(5).iloc[-1])
        vol10 = close_series.rolling(10).std().iloc[-1]

        feat_row = np.array([[
            open_p, high_p, low_p, vol,
            sma5, sma10, sma20, ema12, ema26,
            rsi, macd, bb_width, ret1, ret5, vol10
        ]])
        feat_row_sc = scaler_X.transform(feat_row)
        pred_sc = model.predict(feat_row_sc)
        pred_price = float(scaler_y.inverse_transform(pred_sc.reshape(-1,1))[0,0])
        pred_price = max(pred_price, 1.0)
        future_prices.append(round(pred_price, 2))
        close_history.append(pred_price)

    # Signal logic
    current_price = float(df["Close"].iloc[-1])
    signals = []
    for p in future_prices:
        pct = (p - current_price) / current_price * 100
        if pct > 2:
            signals.append("STRONG BUY")
        elif pct > 0.5:
            signals.append("BUY")
        elif pct < -2:
            signals.append("STRONG SELL")
        elif pct < -0.5:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    # Confidence band ±1.5%
    upper = [round(p * 1.015, 2) for p in future_prices]
    lower = [round(p * 0.985, 2) for p in future_prices]

    # Historical indicators for chart
    hist_dates  = [str(d.date()) for d in df_feat.index[-60:]]
    hist_close  = [round(float(v), 2) for v in df_feat["Close"].values[-60:]]
    hist_sma20  = [round(float(v), 2) if not np.isnan(v) else None for v in df_feat["SMA_20"].values[-60:]]
    hist_sma5   = [round(float(v), 2) if not np.isnan(v) else None for v in df_feat["SMA_5"].values[-60:]]
    hist_bb_up  = [round(float(v), 2) if not np.isnan(v) else None for v in df_feat["BB_upper"].values[-60:]]
    hist_bb_dn  = [round(float(v), 2) if not np.isnan(v) else None for v in df_feat["BB_lower"].values[-60:]]
    hist_rsi    = [round(float(v), 2) if not np.isnan(v) else None for v in df_feat["RSI"].values[-60:]]
    hist_macd   = [round(float(v), 4) if not np.isnan(v) else None for v in df_feat["MACD"].values[-60:]]
    hist_macd_s = [round(float(v), 4) if not np.isnan(v) else None for v in df_feat["MACD_signal"].values[-60:]]
    hist_vol    = [int(v) for v in df_feat["Volume"].values[-60:]]
    in_sample_dates  = [str(d.date()) for d in df_feat.index[-60:]]
    in_sample_prices = [round(float(v), 2) for v in in_sample_pred[-60:]]

    # Overall signal (majority vote)
    buy_count  = sum(1 for s in signals if "BUY"  in s)
    sell_count = sum(1 for s in signals if "SELL" in s)
    if buy_count > sell_count:
        overall_signal = "BUY"
        signal_color   = "#22c55e"
    elif sell_count > buy_count:
        overall_signal = "SELL"
        signal_color   = "#ef4444"
    else:
        overall_signal = "HOLD"
        signal_color   = "#f59e0b"

    target_price = round(future_prices[-1], 2)
    upside_pct   = round((target_price - current_price) / current_price * 100, 2)

    return {
        "future_dates":  [str(d.date()) for d in future_dates],
        "future_prices": future_prices,
        "upper_band":    upper,
        "lower_band":    lower,
        "signals":       signals,
        "overall_signal":overall_signal,
        "signal_color":  signal_color,
        "target_price":  target_price,
        "upside_pct":    upside_pct,
        "current_price": round(current_price, 2),
        "hist_dates":    hist_dates,
        "hist_close":    hist_close,
        "hist_sma20":    hist_sma20,
        "hist_sma5":     hist_sma5,
        "hist_bb_upper": hist_bb_up,
        "hist_bb_lower": hist_bb_dn,
        "hist_rsi":      hist_rsi,
        "hist_macd":     hist_macd,
        "hist_macd_signal": hist_macd_s,
        "hist_volume":   hist_vol,
        "in_sample_dates":  in_sample_dates,
        "in_sample_prices": in_sample_prices,
    }
