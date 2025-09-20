# forex_complete.py — Complete Multi-User Forex & Crypto System with Professional Dashboard
# Run:   streamlit run forex_complete.py

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time, inspect, dataclasses, math, random, hashlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
import yfinance as yf
from datetime import datetime, timedelta
import sqlite3
import bcrypt
import json

# ──────────────────────────────────────────────────────────────────────────────
# DATABASE SETUP
# ──────────────────────────────────────────────────────────────────────────────
def setup_database():
    """Create database tables for multi-user system"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'trader',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Strategies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            parameters TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            performance_score REAL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Backtests table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            strategy_id INTEGER,
            symbol TEXT,
            timeframe TEXT,
            total_return REAL,
            final_equity REAL,
            initial_equity REAL DEFAULT 10000,
            total_trades INTEGER,
            win_rate REAL,
            profit_factor REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Trades table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            backtest_id INTEGER,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            symbol TEXT,
            side INTEGER,
            entry_price REAL,
            exit_price REAL,
            pnl REAL,
            r_mult REAL,
            size REAL,
            FOREIGN KEY (backtest_id) REFERENCES backtests (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# User Management
class UserManager:
    @staticmethod
    def hash_password(password):
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    @staticmethod
    def verify_password(password, hashed):
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    
    @staticmethod
    def create_user(username, email, password):
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        hashed_pw = UserManager.hash_password(password)
        try:
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, hashed_pw))
            conn.commit()
            conn.close()
            return True, "Account created successfully!"
        except sqlite3.IntegrityError as e:
            conn.close()
            if 'username' in str(e):
                return False, "Username already exists!"
            elif 'email' in str(e):
                return False, "Email already exists!"
            else:
                return False, "Registration failed!"
    
    @staticmethod
    def authenticate(username, password):
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, password_hash, role, email 
            FROM users 
            WHERE username = ? AND is_active = 1
        ''', (username,))
        user = cursor.fetchone()
        
        if user and UserManager.verify_password(password, user[1]):
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (user[0],))
            conn.commit()
            conn.close()
            
            return {
                'id': user[0],
                'username': username,
                'role': user[2],
                'email': user[3]
            }
        
        conn.close()
        return None

def save_backtest_results(user_id, symbol, stats, trades, strategy_config=None):
    """Save backtest results to database"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    # Convert all values to proper Python types and handle inf/nan
    def safe_float(value):
        if value is None or np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    
    def safe_int(value):
        if value is None:
            return 0
        return int(value)
    
    # Save backtest summary with safe conversions
    cursor.execute('''
        INSERT INTO backtests (user_id, symbol, total_return, final_equity, 
                             initial_equity, total_trades, win_rate, profit_factor, max_drawdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        int(user_id), 
        str(symbol), 
        safe_float(stats.ret_pct), 
        safe_float(stats.final_eq), 
        safe_float(stats.init_eq),
        safe_int(stats.n_trades), 
        safe_float(stats.win_rate), 
        safe_float(stats.pf), 
        safe_float(stats.max_dd_pct)
    ))
    
    backtest_id = cursor.lastrowid
    
    # Save individual trades with safe conversions
    for trade in trades:
        cursor.execute('''
            INSERT INTO trades (backtest_id, entry_time, exit_time, symbol, side,
                              entry_price, exit_price, pnl, r_mult, size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(backtest_id), 
            str(trade.entry_time), 
            str(trade.exit_time) if trade.exit_time else None, 
            str(symbol), 
            int(trade.side),
            safe_float(trade.entry), 
            safe_float(trade.exit_price), 
            safe_float(trade.pnl), 
            safe_float(trade.r_mult), 
            safe_float(trade.size)
        ))
    
    conn.commit()
    conn.close()
    return backtest_id

# ──────────────────────────────────────────────────────────────────────────────
# Try to import your backend (must sit next to this file)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import complete_forex_system as fx
    _import_error = None
except Exception as e:
    fx, _import_error = None, e

_global_seed = 42

def _backend_required():
    st.error(
        "Could not import **complete_forex_system.py**.\n\n"
        f"Error: `{_import_error}`\n\n"
        "• Make sure **forex_complete.py** and **complete_forex_system.py** are in the same folder.\n"
        "• Ensure **complete_forex_system.py** runs without errors (python3 -m py_compile complete_forex_system.py)."
    )
    st.stop()

def _select_ohlcv(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return df_in

    cols = df_in.columns

    def _find(colkey: str):
        for c in cols:
            if isinstance(c, str) and c.lower() == colkey:
                return c
        for c in cols:
            if isinstance(c, tuple):
                for p in c[::-1]:
                    if isinstance(p, str) and p.lower() == colkey:
                        return c
        for c in cols:
            s = str(c).lower()
            if s.endswith(f".{colkey}") or s.endswith(f"_{colkey}") or s == colkey:
                return c
        return None

    open_col  = _find("open")
    high_col  = _find("high")
    low_col   = _find("low")
    close_col = _find("close")
    vol_col   = _find("volume")

    if not all([open_col, high_col, low_col, close_col]):
        raise ValueError("Could not locate OHLC columns in the input DataFrame.")

    out = pd.DataFrame(index=df_in.index)
    out["open"]  = pd.to_numeric(df_in[open_col], errors="coerce")
    out["high"]  = pd.to_numeric(df_in[high_col], errors="coerce")
    out["low"]   = pd.to_numeric(df_in[low_col], errors="coerce")
    out["close"] = pd.to_numeric(df_in[close_col], errors="coerce")
    if vol_col is not None:
        out["volume"] = pd.to_numeric(df_in[vol_col], errors="coerce").fillna(0)
    else:
        out["volume"] = 0
    return out

def maybe_resample(df_in: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return df_in

    df = _select_ohlcv(df_in)

    if not isinstance(df.index, pd.DatetimeIndex) or not minutes or minutes <= 1:
        return df

    try:
        infer = pd.infer_freq(df.index)
        if infer and infer.endswith("T"):
            cur_min = int(infer[:-1])
            if cur_min >= int(minutes):
                return df
    except Exception:
        pass

    rule = f"{int(minutes)}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule).agg(agg).dropna()
    if out.empty:
        return df
    return out

# ──────────────────────────────────────────────────────────────────────────────
# CRYPTO & FX DATA HELPERS
# ──────────────────────────────────────────────────────────────────────────────
CRYPTO_MAP = {
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD", "BNBUSD": "BNB-USD",
    "XRPUSD": "XRP-USD", "ADAUSD": "ADA-USD", "DOGEUSD": "DOGE-USD", "AVAXUSD": "AVAX-USD",
    "TRXUSD": "TRX-USD", "DOTUSD": "DOT-USD",
}

def _is_crypto(symbol: str) -> bool:
    return (symbol in CRYPTO_MAP) or symbol.endswith("-USD")

def _fetch_crypto_ohlcv(symbol: str, days: int, interval_min: int) -> pd.DataFrame:
    yf_sym = CRYPTO_MAP.get(symbol, symbol)

    if interval_min <= 1 and days <= 7:
        interval = "1m"; max_days = min(days, 7)
    elif days <= 60:
        interval = "5m"; max_days = min(days, 60)
    else:
        interval = "15m"; max_days = min(days, 60)
    period = f"{max_days}d"

    df = yf.download(yf_sym, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} ({yf_sym}) with {interval}/{period}")

    # FIX: Handle MultiIndex columns by flattening them immediately
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]
    
    # Now safely convert to lowercase
    df = df.rename(columns=str.lower)
    
    # Ensure required columns exist
    for c in ["open","high","low","close"]:
        if c not in df.columns:
            raise ValueError(f"Downloaded data missing '{c}' for {symbol}")
    if "volume" not in df.columns:
        df["volume"] = 0

    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        try:
            df.index = df.index.tz_localize("UTC")
        except Exception:
            pass
    df = df.sort_index()

    return df[["open","high","low","close","volume"]]

# Live data functions
@st.cache_data(ttl=5)
def get_live_price(symbol):
    """Get most recent price data"""
    try:
        if symbol.endswith("-USD"):
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            return data.tail(60)
        else:
            fx_symbol = symbol[:3] + "=X"
            ticker = yf.Ticker(fx_symbol)
            data = ticker.history(period="1d", interval="5m") 
            return data.tail(60)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=10)
def get_crypto_info(symbol):
    """Get crypto market info"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'market_cap': info.get('marketCap', 0),
            '24h_volume': info.get('volume24Hr', 0),
            'supply': info.get('circulatingSupply', 0)
        }
    except:
        return {'market_cap': 0, '24h_volume': 0, 'supply': 0}

# ──────────────────────────────────────────────────────────────────────────────
# BACKEND INTEGRATION
# ──────────────────────────────────────────────────────────────────────────────
def _resolve(names: list[str]):
    if fx is None:
        return None
    for n in names:
        f = getattr(fx, n, None)
        if callable(f):
            return f
    return None

def _set_seed(seed: int):
    global _global_seed
    if seed == -1:
        s = int(time.time()) & 0xFFFFFFFF
    else:
        s = int(seed)
    
    _global_seed = s
    
    if fx is not None and hasattr(fx, "set_run_seed") and callable(getattr(fx, "set_run_seed")):
        fx.set_run_seed(s); return
    if fx is not None and hasattr(fx, "set_seed") and callable(getattr(fx, "set_seed")):
        fx.set_seed(s); return
    random.seed(s)
    np.random.seed(s)

def _generate_data(symbol: str, days: int, interval_min: int = 1) -> pd.DataFrame:
    if _is_crypto(symbol):
        return _fetch_crypto_ohlcv(symbol, days, interval_min)

    f = _resolve([
        "generate_synthetic_pair", "generate_synthetic_forex", "generate_synthetic_forex_data",
        "generate_forex_data", "generate_synthetic_data", "generate_data",
    ])
    if f is not None:
        sig = inspect.signature(f)
        kwargs = {}
        if "pair" in sig.parameters:    kwargs["pair"] = symbol
        if "symbol" in sig.parameters:  kwargs["symbol"] = symbol
        if "ticker" in sig.parameters:  kwargs["ticker"] = symbol
        if "days" in sig.parameters:    kwargs["days"] = days
        if "interval_min" in sig.parameters: kwargs["interval_min"] = interval_min
        if kwargs:
            return f(**kwargs)
        try:
            return f(symbol, days)
        except TypeError:
            return f(days=days)

    return _fallback_synth_fx(symbol, days, interval_min, seed=_global_seed)

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    f = _resolve(["compute_indicators", "calculate_indicators"])
    if f is None:
        return df
    return f(df)

def _build_signal_config(**cfg_kwargs):
    if fx is not None and hasattr(fx, "SignalConfig"):
        try:
            sc = fx.SignalConfig()
            for k, v in cfg_kwargs.items():
                if hasattr(sc, k):
                    setattr(sc, k, v)
            return sc
        except Exception:
            pass
    class _Cfg: ...
    sc = _Cfg()
    for k, v in cfg_kwargs.items():
        setattr(sc, k, v)
    return sc

def _generate_signals(df: pd.DataFrame, cfg) -> pd.DataFrame:
    f = _resolve(["generate_signals", "generate_trading_signals"])
    if f is None:
        return df
    sig = inspect.signature(f)
    return f(df) if len(sig.parameters) == 1 else f(df, cfg)

def _call_backtest(df: pd.DataFrame, **kwargs):
    if fx is None or not hasattr(fx, "backtest") or not callable(fx.backtest):
        raise AttributeError("No backtest(df, ...) function found in complete_forex_system.py")
    sig = inspect.signature(fx.backtest)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fx.backtest(df, **filtered)

def _result_to_dict(res) -> dict:
    if isinstance(res, dict):
        return res
    if dataclasses.is_dataclass(res):
        return dataclasses.asdict(res)
    if hasattr(res, "__dict__"):
        return dict(res.__dict__)
    fields = ["initial_balance", "final_equity", "total_return_pct", "total_trades", "winning_trades", "avg_win", "avg_loss"]
    out = {}
    for f in fields:
        if hasattr(res, f):
            out[f] = getattr(res, f)
    return out

def _fallback_synth_fx(pair="EURUSD", days=30, interval_min=1, seed=None) -> pd.DataFrame:
    if seed is None:
        seed = int(time.time())
    
    rng = np.random.default_rng(seed)
    n = int(days*24*60/interval_min)
    vol = 0.00008 + (seed % 100) * 0.000001
    drift = 0.00001 + (seed % 50) * 0.000002
    base = {"EURUSD":1.085,"GBPUSD":1.27,"USDJPY":156.0,"AUDUSD":0.66,"USDCAD":1.36,"USDCHF":0.90,"NZDUSD":0.61}.get(pair,1.10)
    
    base = base * (1 + (seed % 1000) * 0.00001)
    
    rets = drift + vol*rng.standard_normal(n)
    close = base * np.exp(np.cumsum(rets))
    ts = pd.date_range(end=pd.Timestamp.utcnow().floor("min"), periods=n, freq=f"{interval_min}min")
    df = pd.DataFrame(index=ts)
    df["Close"] = close
    df["Open"]  = df["Close"].shift(1).fillna(df["Close"])
    wick = np.abs(close)*vol*3
    df["High"] = np.maximum(df["Open"], df["Close"]) + wick
    df["Low"]  = np.minimum(df["Open"], df["Close"]) - wick
    df["Volume"] = 1000 + (seed % 500)
    
    out = pd.DataFrame({
        "open": df["Open"].values, "high": df["High"].values,
        "low":  df["Low"].values, "close":df["Close"].values, "volume":df["Volume"].values
    }, index=df.index)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────────────────────
def _atr(H, L, C, n=14):
    hl = (H - L).abs()
    hc = (H - C.shift(1)).abs()
    lc = (L - C.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _donch_prev(H, L, n):
    upper = H.rolling(n).max().shift(1)
    lower = L.rolling(n).min().shift(1)
    return upper, lower

def _rsi(C, n=14):
    delta = C.diff()
    gain = delta.where(delta > 0, 0).rolling(window=n).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=n).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _wma(C, n):
    return C.rolling(n).apply(lambda x: np.average(x, weights=np.arange(1, n+1)), raw=True)

def _macd(C, fast=12, slow=26, signal=9):
    ema_fast = C.ewm(span=fast).mean()
    ema_slow = C.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY V4
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class StratV4Config:
    lookback: int = 20
    exit_lookback: int = 10
    wma_len: int = 50
    atr_len: int = 14
    atr_mult: float = 3.0
    rr: float = 4.0
    risk_per_trade: float = 0.002
    cooldown_bars: int = 0
    slippage: float = 0.00002
    comm_bp: float = 0.2
    long_only: bool = True
    max_bars_in_trade: int = 600
    breakeven_at_R: float = 0.75
    partial_at_R: float = 2.0
    partial_size: float = 0.5
    use_trend_filter: bool = True
    use_vol_filter: bool = False
    use_session_filter: bool = False
    atr_pct_min: float = 0.00003
    atr_pct_max: float = 0.0080
    use_macd_filter: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    use_rsi_filter: bool = False
    rsi_len: int = 14
    rsi_long_min: float = 50.0
    breakout_buffer_atr: float = 0.0
    use_adx_filter: bool = False
    use_range_expansion: bool = False

def add_indicators_v4(df, cfg: StratV4Config):
    # Use the existing _select_ohlcv function that handles MultiIndex properly
    df_clean = _select_ohlcv(df)
    
    O, H, L, C = df_clean["open"], df_clean["high"], df_clean["low"], df_clean["close"]
    out = pd.DataFrame(index=df_clean.index)
    out["open"], out["high"], out["low"], out["close"] = O, H, L, C

    out["_ATR"] = _atr(H, L, C, cfg.atr_len)
    out["_ATRpct"] = (out["_ATR"] / C).clip(lower=1e-10)
    out["_WMA"] = _wma(C, cfg.wma_len)
    out["_WMA_slope"] = (out["_WMA"] - out["_WMA"].shift(5)) / out["_WMA"].shift(5)
    out["_U"], out["_L"] = _donch_prev(H, L, cfg.lookback)

    if cfg.use_macd_filter:
        macd, sig, hist = _macd(C, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        out["_MACD"] = macd
        out["_MACD_signal"] = sig
        out["_MACD_hist"] = hist

    if cfg.use_rsi_filter:
        out["_RSI"] = _rsi(C, cfg.rsi_len)

    out["_session"] = 1
    return out.dropna()

@dataclass
class Trade:
    entry_time: pd.Timestamp
    side: int
    entry: float
    stop: float
    tp: float
    size: float
    init_stop: float
    partial_size: float = 0.0
    bars_in: int = 0
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = 0.0
    r_mult: float = 0.0

@dataclass
class BTStats:
    init_eq: float
    final_eq: float
    ret_pct: float
    n_trades: int
    win_rate: float
    pf: float
    exp_per_trade: float
    max_dd_pct: float

def backtest_v4(df_in, cfg: StratV4Config, init_equity=10_000.0):
    df = add_indicators_v4(df_in, cfg)
    O, H, L, C = df["open"], df["high"], df["low"], df["close"]

    equity, peak, max_dd = init_equity, init_equity, 0.0
    open_trades: list[Trade] = []
    closed: list[Trade] = []
    cooldown = 0

    def _commission(notional):
        return abs(notional) * (cfg.comm_bp * 1e-4)

    for t in range(max(cfg.wma_len, cfg.macd_slow), len(df)):
        ts = df.index[t]
        o, h, l, c = O.iat[t], H.iat[t], L.iat[t], C.iat[t]
        atr = float(df["_ATR"].iat[t])
        atr_pct = float(df["_ATRpct"].iat[t])
        wma = float(df["_WMA"].iat[t])
        slope = float(df["_WMA_slope"].iat[t])
        u_prev = float(df["_U"].iat[t])
        l_prev = float(df["_L"].iat[t])

        macd_ok = True
        if cfg.use_macd_filter and not np.isnan(df["_MACD_hist"].iat[t]):
            macd_ok = df["_MACD"].iat[t] > df["_MACD_signal"].iat[t]
        rsi_ok = True
        if cfg.use_rsi_filter and not np.isnan(df["_RSI"].iat[t]):
            rsi_ok = df["_RSI"].iat[t] >= cfg.rsi_long_min

        if open_trades:
            tr = open_trades[0]
            tr.bars_in += 1
            remaining_size = tr.size - tr.partial_size
            r = abs(tr.entry - tr.init_stop)

            if cfg.breakeven_at_R > 0 and r > 0:
                curr_r = (h - tr.entry) / r if tr.side == 1 else (tr.entry - l) / r
                if curr_r >= cfg.breakeven_at_R:
                    tr.stop = max(tr.stop, tr.entry) if tr.side == 1 else min(tr.stop, tr.entry)

            if remaining_size > 0 and cfg.partial_at_R > 0 and curr_r >= cfg.partial_at_R:
                partial_px = h if tr.side == 1 else l
                partial_pnl = cfg.partial_size * tr.size * (partial_px - tr.entry if tr.side == 1 else tr.entry - partial_px) - _commission(cfg.partial_size * tr.size * partial_px)
                equity += partial_pnl
                tr.partial_size += cfg.partial_size * tr.size
                tr.pnl += partial_pnl

            trail_r = 1.5
            if curr_r >= trail_r:
                tr.stop = max(tr.stop, h - cfg.atr_mult * atr) if tr.side == 1 else min(tr.stop, l + cfg.atr_mult * atr)

            exit_px = None
            if tr.side == 1:
                if l <= tr.stop: exit_px = tr.stop - cfg.slippage
                elif h >= tr.tp: exit_px = tr.tp - cfg.slippage
            else:
                if h >= tr.stop: exit_px = tr.stop + cfg.slippage
                elif l <= tr.tp: exit_px = tr.tp + cfg.slippage

            if exit_px is None and tr.bars_in >= cfg.max_bars_in_trade:
                exit_px = c - cfg.slippage if tr.side == 1 else c + cfg.slippage

            if exit_px is not None and remaining_size > 0:
                pnl_remaining = remaining_size * (exit_px - tr.entry if tr.side == 1 else tr.entry - exit_px) - _commission(remaining_size * exit_px)
                tr.pnl += pnl_remaining
                equity += pnl_remaining
                tr.exit_time, tr.exit_price = ts, exit_px
                tr.r_mult = tr.pnl / (init_equity * cfg.risk_per_trade)
                closed.append(tr)
                open_trades.clear()
                peak = max(peak, equity)
                max_dd = min(max_dd, (equity / peak - 1.0) * 100)

        if not open_trades and cooldown == 0 and not np.isnan(atr) and atr > 0:
            in_vol = not cfg.use_vol_filter or (cfg.atr_pct_min <= atr_pct <= cfg.atr_pct_max)
            uptrend = not cfg.use_trend_filter or (c > wma and slope > 0)
            long_signal = in_vol and uptrend and macd_ok and rsi_ok and (c >= u_prev + cfg.breakout_buffer_atr * atr)

            if long_signal:
                side = 1
                entry = c + cfg.slippage
                stop = entry - cfg.atr_mult * atr
                tp = entry + cfg.rr * (entry - stop)

                risk_cash = equity * cfg.risk_per_trade
                per_unit_risk = entry - stop
                size = risk_cash / per_unit_risk if per_unit_risk > 0 else 0

                if size > 0:
                    equity -= _commission(size * entry)
                    open_trades.append(Trade(
                        entry_time=ts, side=side, entry=entry, stop=stop, tp=tp,
                        size=size, init_stop=stop, bars_in=0
                    ))

        if cooldown > 0: cooldown -= 1

    if open_trades:
        tr = open_trades[0]
        last_c = C.iat[-1]
        exit_px = last_c - cfg.slippage if tr.side == 1 else last_c + cfg.slippage
        remaining_size = tr.size - tr.partial_size
        if remaining_size > 0:
            pnl_remaining = remaining_size * (exit_px - tr.entry if tr.side == 1 else tr.entry - exit_px) - _commission(remaining_size * exit_px)
            tr.pnl += pnl_remaining
            equity += pnl_remaining
        tr.exit_time, tr.exit_price = df.index[-1], exit_px
        tr.r_mult = tr.pnl / max(1e-9, init_equity * cfg.risk_per_trade)
        closed.append(tr)
        peak = max(peak, equity)
        max_dd = min(max_dd, (equity / peak - 1.0) * 100)

    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf') if gross_win > 0 else 0.0
    wr = len(wins) / len(closed) * 100 if closed else 0.0
    exp = (gross_win - gross_loss) / len(closed) if closed else 0.0

    stats = BTStats(
        init_eq=init_equity, final_eq=equity, ret_pct=(equity / init_equity - 1.0) * 100,
        n_trades=len(closed), win_rate=wr, pf=pf, exp_per_trade=exp, max_dd_pct=max_dd
    )
    return stats, closed

def _atrpct_quantiles(df_in, cfg: StratV4Config):
    df = add_indicators_v4(df_in, cfg).dropna()
    ap = df["_ATRpct"].clip(lower=1e-10)
    if ap.empty:
        return 0.00005, 0.0020
    return float(ap.quantile(0.30)), float(ap.quantile(0.85))

def quick_grid_search(df):
    base_cfg = StratV4Config()
    q30, q85 = _atrpct_quantiles(df, base_cfg)
    bands = [
        (max(1e-6, q30*0.7), q85*1.0),
        (max(1e-6, q30*0.9), q85*1.1),
        (max(1e-6, q30*0.5), q85*1.3),
    ]
    best = None
    for lookback in [15, 20, 25]:
        for atr_mult in [2.5, 3.0, 3.5]:
            for rr in [3.5, 4.0, 4.5]:
                for atr_min, atr_max in bands:
                    cfg = StratV4Config(
                        lookback=lookback, atr_mult=atr_mult, rr=rr,
                        atr_pct_min=atr_min, atr_pct_max=atr_max
                    )
                    stats, trades = backtest_v4(df, cfg)
                    score = (len(trades)>0, np.isfinite(stats.pf), round(stats.pf,3), round(stats.ret_pct,2))
                    if (best is None) or (score > best[0]):
                        best = (score, cfg, stats, len(trades))
    return best

def _price_fig_with_trades(df: pd.DataFrame, trades: list[Trade], symbol: str, show_ma=True) -> go.Figure:
    # FIX: Handle MultiIndex/tuple columns properly
    cols = {}
    for c in df.columns:
        if isinstance(c, tuple):
            key = str(c[0]).lower()
        else:
            key = str(c).lower()
        cols[key] = c

    close_col = cols.get("close")
    open_col  = cols.get("open")
    high_col  = cols.get("high")
    low_col   = cols.get("low")

    fig = go.Figure()
    if all([open_col, high_col, low_col, close_col]):
        fig.add_trace(go.Candlestick(
            x=df.index, open=df[open_col], high=df[high_col], low=df[low_col], close=df[close_col],
            name="Price"
        ))
        if show_ma and len(df) >= 50:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[close_col].rolling(50).mean(), name="MA50",
                line=dict(width=1.5)
            ))
        if show_ma and len(df) >= 200:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[close_col].rolling(200).mean(), name="MA200", line=dict(dash="dot")
            ))
    elif close_col:
        fig.add_trace(go.Scatter(x=df.index, y=df[close_col], name=f"{symbol} Close", mode="lines"))

    if trades:
        long_entries  = [t.entry_time for t in trades if t.side==1]
        long_prices   = [t.entry      for t in trades if t.side==1]
        short_entries = [t.entry_time for t in trades if t.side==-1]
        short_prices  = [t.entry      for t in trades if t.side==-1]
        exits_ts      = [t.exit_time  for t in trades if t.exit_time is not None]
        exits_px      = [t.exit_price for t in trades if t.exit_time is not None]

        if long_entries:
            fig.add_trace(go.Scatter(
                x=long_entries, y=long_prices, mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="green"), name="Long Entry"
            ))
        if short_entries:
            fig.add_trace(go.Scatter(
                x=short_entries, y=short_prices, mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="red"), name="Short Entry"
            ))
        if exits_ts:
            fig.add_trace(go.Scatter(
                x=exits_ts, y=exits_px, mode="markers",
                marker=dict(symbol="x", size=9, color="black"), name="Exit"
            ))

    fig.update_layout(
        height=560,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Time", yaxis_title="Price",
        hovermode="x unified", template="plotly_white",
        title=f"{symbol} — Price & Trades"
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL DASHBOARD FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def calculate_overall_win_rate(user_id):
    """Calculate overall win rate across all user trades"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT COUNT(*) as winning_trades
        FROM trades t
        JOIN backtests b ON t.backtest_id = b.id
        WHERE b.user_id = ? AND t.pnl > 0
    ''', (user_id,))
    winning_trades = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT COUNT(*) as total_trades
        FROM trades t
        JOIN backtests b ON t.backtest_id = b.id
        WHERE b.user_id = ?
    ''', (user_id,))
    total_trades = cursor.fetchone()[0]
    
    conn.close()
    
    if total_trades > 0:
        return (winning_trades / total_trades) * 100
    return 0.0

def show_performance_chart(user_id):
    """Show performance chart over time"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT created_at, total_return, symbol
        FROM backtests 
        WHERE user_id = ? 
        ORDER BY created_at
    ''', (user_id,))
    
    data = cursor.fetchall()
    conn.close()
    
    if data:
        df = pd.DataFrame(data, columns=['Date', 'Return', 'Symbol'])
        df['Date'] = pd.to_datetime(df['Date'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Return'],
            mode='lines+markers',
            name='Returns %',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Return %",
            height=400,
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Run some backtests to see performance chart")



def show_recent_activity_summary(user_id):
    """Show recent activity in a clean format"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT symbol, total_return, total_trades, created_at
        FROM backtests 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 8
    ''', (user_id,))
    
    recent = cursor.fetchall()
    conn.close()
    
    if recent:
        for r in recent:
            symbol, ret, trades, date = r
            date_str = pd.to_datetime(date).strftime('%m/%d %H:%M')
            
            # Color coding
            if ret > 5:
                icon = "🚀"
            elif ret > 0:
                icon = "🟢"
            elif ret > -5:
                icon = "🟡"
            else:
                icon = "🔴"
            
            st.write(f"{icon} **{symbol}:** {ret:.1f}% ({trades} trades) - *{date_str}*")
    else:
        st.info("No recent activity. Start backtesting!")

def show_top_strategies(user_id):
    """Show top performing strategies"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT symbol, MAX(total_return) as best_return, COUNT(*) as times_tested
        FROM backtests 
        WHERE user_id = ? 
        GROUP BY symbol
        ORDER BY best_return DESC
        LIMIT 5
    ''', (user_id,))
    
    top_strategies = cursor.fetchall()
    conn.close()
    
    if top_strategies:
        for strategy in top_strategies:
            symbol, best_return, times_tested = strategy
            
            if best_return > 10:
                badge = "🏆"
            elif best_return > 5:
                badge = "🥈"
            elif best_return > 0:
                badge = "🥉"
            else:
                badge = "📊"
            
            st.write(f"{badge} **{symbol}:** {best_return:.1f}% (tested {times_tested}x)")
    else:
        st.info("No strategies tested yet.")

def get_user_statistics(user_id):
    """Get comprehensive user statistics"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    # Basic stats
    cursor.execute('SELECT COUNT(*) FROM backtests WHERE user_id = ?', (user_id,))
    total_backtests = cursor.fetchone()[0]
    
    cursor.execute('SELECT SUM(total_trades) FROM backtests WHERE user_id = ?', (user_id,))
    total_trades = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT AVG(total_return), MAX(total_return) FROM backtests WHERE user_id = ?', (user_id,))
    avg_return, best_return = cursor.fetchone()
    
    # User info
    cursor.execute('SELECT created_at FROM users WHERE id = ?', (user_id,))
    created_at = cursor.fetchone()
    
    conn.close()
    
    return {
        'total_backtests': total_backtests,
        'total_trades': total_trades,
        'avg_return': avg_return or 0.0,
        'best_return': best_return or 0.0,
        'created_at': created_at[0] if created_at else None
    }

def get_platform_statistics():
    """Get platform-wide statistics"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
    total_users = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM backtests')
    total_backtests = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM trades')
    total_trades = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_users': total_users,
        'total_backtests': total_backtests,
        'total_trades': total_trades
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP WITH PROFESSIONAL AUTHENTICATION
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="🚀 Professional Trading Platform", layout="wide")
    
    # Initialize database
    setup_database()
    
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
    if 'run_count' not in st.session_state:
        st.session_state.run_count = 0
    if 'trades' not in st.session_state:
        st.session_state.trades = []
        st.session_state.equity = 10000
        st.session_state.start_equity = 10000

    # Show login page or main app
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_enhanced_main_app()

def show_login_page():
    """Professional authentication interface"""
    st.title("🚀 Professional Trading Platform")
    st.markdown("### Multi-User Collaborative Trading System")
    
    tab1, tab2, tab3 = st.tabs(["🔐 Login", "📝 Register", "ℹ️ About"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                username = st.text_input("👤 Username", placeholder="Enter your username")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            with col2:
                st.write("")
                st.write("")
                submitted = st.form_submit_button("🚀 Login", type="primary", use_container_width=True)
            
            if submitted:
                if username and password:
                    user = UserManager.authenticate(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.success(f"Welcome back, {user['username']}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password!")
                else:
                    st.error("❌ Please fill in both username and password!")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("register_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_username = st.text_input("👤 Choose Username", placeholder="Enter desired username")
                email = st.text_input("📧 Email Address", placeholder="your@email.com")
                new_password = st.text_input("🔒 Password", type="password", placeholder="Enter secure password")
                confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
            
            with col2:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                submitted = st.form_submit_button("✨ Create Account", type="primary", use_container_width=True)
            
            if submitted:
                if not all([new_username, email, new_password, confirm_password]):
                    st.error("❌ Please fill in all fields!")
                elif new_password != confirm_password:
                    st.error("❌ Passwords don't match!")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters long!")
                elif "@" not in email:
                    st.error("❌ Please enter a valid email address!")
                else:
                    success, message = UserManager.create_user(new_username, email, new_password)
                    if success:
                        st.success(f"✅ {message} Please login with your new account.")
                    else:
                        st.error(f"❌ {message}")
    
    with tab3:
        st.subheader("About This Platform")
        st.markdown("""
        🎯 **Professional Multi-User Trading Platform Features:**
        
        ✅ **Individual User Accounts** - Each user has private access
        ✅ **Strategy Backtesting** - Test your trading strategies  
        ✅ **Live Market Data** - Real-time price feeds
        ✅ **Results Storage** - All backtests saved permanently
        ✅ **Trade History** - Detailed trade-by-trade analysis
        ✅ **Multi-Asset Support** - Forex pairs and cryptocurrencies
        ✅ **Professional Dashboard** - Advanced analytics and metrics
        
        🔧 **Technical Features:**
        - Advanced Strategy v4 with Donchian + MACD + WMA
        - Automated parameter optimization
        - Risk management and position sizing
        - Real-time performance metrics
        - Professional performance charts
        
        🚀 **Get Started:**
        1. Create your account above
        2. Login to access the trading system  
        3. Run backtests and save results
        4. Invite friends to create their own accounts
        """)

def show_enhanced_main_app():
    """Professional dashboard layout without sidebar clutter"""
    
    # Professional header with user info and logout
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    
    with col_header1:
        st.title("🚀 Professional Trading Platform")
        st.caption("Multi-User Collaborative Trading System")
    
    with col_header2:
        st.write(f"**Welcome, {st.session_state.user['username']}**")
        st.caption(f"📧 {st.session_state.user['email']}")
    
    with col_header3:
        col_logout1, col_logout2 = st.columns([1, 1])
        with col_logout1:
            if st.button("👤 Profile", use_container_width=True):
                st.session_state.show_profile = True
        with col_logout2:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.user = None
                st.rerun()
    
    st.markdown("---")
    
    # Main navigation tabs - Professional style
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 **BACKTESTING**", 
        "🔴 **LIVE DEMO**", 
        "📈 **MY RESULTS**", 
        "⚙️ **DASHBOARD**"
    ])
    
    # TAB 1: Enhanced Backtesting
    with tab1:
        show_enhanced_backtesting()
    
    # TAB 2: Live Demo
    with tab2:
        show_live_demo()
    
    # TAB 3: Results History
    with tab3:
        show_user_results_history()
    
    # TAB 4: Professional Dashboard with all user info
    with tab4:
        show_professional_dashboard()

def show_professional_dashboard():
    """Professional dashboard with user stats and system info"""
    st.header("📊 Professional Dashboard")
    
    # User Statistics Overview
    st.subheader("👤 Your Account Overview")
    
    user_stats = get_user_statistics(st.session_state.user['id'])
    
    # Professional metrics layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📈 Total Backtests", 
            value=user_stats['total_backtests'],
            help="Total number of backtests you've run"
        )
    
    with col2:
        st.metric(
            label="🎯 Total Trades", 
            value=user_stats['total_trades'],
            help="Total trades across all backtests"
        )
    
    with col3:
        st.metric(
            label="🏆 Best Return", 
            value=f"{user_stats['best_return']:.2f}%",
            help="Your best performing backtest return"
        )
    
    with col4:
        st.metric(
            label="💰 Avg Return", 
            value=f"{user_stats['avg_return']:.2f}%",
            help="Average return across all backtests"
        )
    
    with col5:
        win_rate = calculate_overall_win_rate(st.session_state.user['id'])
        st.metric(
            label="📊 Win Rate", 
            value=f"{win_rate:.1f}%",
            help="Overall win rate across all trades"
        )
    
    # Performance Chart
    st.subheader("📈 Performance Over Time")
    show_performance_chart(st.session_state.user['id'])
    
    # Recent Activity Summary
    col_recent1, col_recent2 = st.columns(2)
    
    with col_recent1:
        st.subheader("🕒 Recent Activity")
        show_recent_activity_summary(st.session_state.user['id'])
    
    with col_recent2:
        st.subheader("🏆 Top Performing Strategies")
        show_top_strategies(st.session_state.user['id'])
    
    # System Information Section
    st.markdown("---")
    st.subheader("🔧 System Information")
    
    col_sys1, col_sys2, col_sys3 = st.columns(3)
    
    with col_sys1:
        st.markdown("**🎯 Available Features:**")
        st.markdown("- ✅ Strategy Backtesting")
        st.markdown("- ✅ Live Market Data")
        st.markdown("- ✅ Results Persistence")
        st.markdown("- ✅ Multi-Asset Support")
        st.markdown("- ✅ Professional Analytics")
    
    with col_sys2:
        st.markdown("**📊 Supported Markets:**")
        st.markdown("- **Forex:** 7 Major Pairs")
        st.markdown("- **Crypto:** 10 Top Cryptocurrencies")
        st.markdown("- **Timeframes:** 1m, 5m, 15m")
        st.markdown("- **Strategies:** Advanced v4 System")
        st.markdown("- **Data:** Real-time & Historical")
    
    with col_sys3:
        platform_stats = get_platform_statistics()
        st.markdown("**🌐 Platform Stats:**")
        st.markdown(f"- **Total Users:** {platform_stats['total_users']}")
        st.markdown(f"- **Total Backtests:** {platform_stats['total_backtests']}")
        st.markdown(f"- **Total Trades:** {platform_stats['total_trades']}")
        st.markdown(f"- **System Status:** ✅ Online")
        st.markdown(f"- **Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Action Buttons
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    col_action1, col_action2, col_action3, col_action4 = st.columns(4)
    
    with col_action1:
        if st.button("🧪 Run New Backtest", use_container_width=True, type="primary"):
            st.info("💡 Switch to the Backtesting tab to run new tests")
    
    with col_action2:
        if st.button("📈 View Live Prices", use_container_width=True):
            st.info("💡 Switch to the Live Demo tab to see real-time prices")
    
    with col_action3:
        if st.button("📊 Analysis Reports", use_container_width=True):
            st.info("💡 Switch to My Results tab for detailed analysis")
    
    with col_action4:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Cache cleared!")
            st.rerun()

def show_enhanced_backtesting():
    """Enhanced backtesting with database integration"""
    st.header("📊 Strategy Backtesting with Results Storage")
    
    col_left, col_right = st.columns([1, 3])
    
    with col_left:
        st.subheader("Market & Engine")
        market = st.radio("Select Market", ["FX", "Crypto"], index=0)
        
        if market == "FX":
            symbol = st.selectbox("FX Pair", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"], index=0)
        else:
            symbol = st.selectbox("Crypto", list(CRYPTO_MAP.keys()), index=0)
        
        engine = st.radio(
            "Backtest Engine",
            ["Backend (complete_forex_system)", "Strategy v4 — Donchian+WMA+MACD"],
            index=(0 if market == "FX" else 1)
        )
        
        st.subheader("Run Settings")
        days = st.slider("Days of data", 5, 120, 30, step=5)
        seed = st.number_input("Seed (-1 = random)", value=-1, step=1)
        show_ma = st.checkbox("Show Moving Averages", value=True)
        
        if engine.startswith("Strategy v4"):
            st.subheader("Strategy v4 Settings")
            lookback = st.number_input("Donchian Lookback", 5, 100, 20, step=1)
            atr_mult = st.number_input("Stop × ATR", 0.5, 5.0, 3.0, step=0.1)
            rr = st.number_input("Risk:Reward", 1.0, 5.0, 4.0, step=0.1)
            use_macd = st.checkbox("Use MACD Filter", value=True)
            auto_opt = st.checkbox("Auto-Optimize", value=False)
        
        run_backtest = st.button("▶️ **RUN BACKTEST**", type="primary", use_container_width=True)
    
    with col_right:
        if run_backtest:
            try:
                st.cache_data.clear()
                st.session_state.run_count += 1
                
                # Generate unique seed for variation
                run_hash = hashlib.md5(f"{time.time()}{st.session_state.run_count}{symbol}".encode()).hexdigest()[:8]
                dynamic_seed = int(run_hash, 16) % 100000 if seed == -1 else seed
                
                t0 = time.time()
                _set_seed(dynamic_seed)
                
                # Vary data period for crypto
                actual_days = days
                if market == "Crypto" and seed == -1:
                    days_offset = (st.session_state.run_count * 3) % 30
                    actual_days = max(5, days - days_offset)
                
                st.info(f"🏃 Run #{st.session_state.run_count} | {symbol} | {actual_days} days | Seed: {dynamic_seed}")
                
                df = _generate_data(symbol, actual_days, interval_min=1)
                
                # Add variation to data
                if not df.empty and 'close' in df.columns and seed == -1:
                    variation = (dynamic_seed % 1000) * 0.0001
                    for col in ['close', 'open', 'high', 'low']:
                        df[col] *= (1 + variation)
                
                st.success(f"✅ Generated {len(df)} rows | First: {float(df['close'].iloc[0]):.6f} | Last: {float(df['close'].iloc[-1]):.6f}")
                
                if engine.startswith("Strategy v4"):
                    # Strategy v4 backtesting
                    cfg = StratV4Config(
                        lookback=lookback, atr_mult=atr_mult, rr=rr, use_macd_filter=use_macd
                    )
                    
                    if auto_opt:
                        score, cfg_opt, stats_opt, ntr = quick_grid_search(df)
                        cfg = cfg_opt
                        st.info(f"🎯 Auto-optimized: PF={stats_opt.pf:.2f}, Return={stats_opt.ret_pct:.2f}%")
                    
                    stats, trades = backtest_v4(df, cfg)
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Final Equity", f"${stats.final_eq:,.2f}")
                    col2.metric("Total Return", f"{stats.ret_pct:.2f}%")
                    col3.metric("Trades", f"{stats.n_trades}")
                    col4.metric("Win Rate", f"{stats.win_rate:.2f}%")
                    
                    wins = [t for t in trades if t.pnl > 0]
                    losses = [t for t in trades if t.pnl <= 0]
                    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
                    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
                    
                    col5, col6, col7, col8 = st.columns(4)
                    col5.metric("Avg Win", f"${avg_win:.2f}")
                    col6.metric("Avg Loss", f"${avg_loss:.2f}")
                    col7.metric("Profit Factor", f"{stats.pf:.2f}" if np.isfinite(stats.pf) else "∞")
                    col8.metric("Max Drawdown", f"{stats.max_dd_pct:.2f}%")
                    
                    verdict = "🟢 PROFITABLE STRATEGY" if stats.ret_pct >= 0 else "🔴 LOSS-MAKING STRATEGY"
                    st.subheader(verdict)
                    
                    # Save results to database
                    backtest_id = save_backtest_results(
                        st.session_state.user['id'], symbol, stats, trades
                    )
                    st.success(f"💾 **Results saved!** Backtest ID: {backtest_id}")
                    
                    # Chart
                    fig = _price_fig_with_trades(df, trades, symbol, show_ma=show_ma)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display trades table
                    if trades:
                        st.subheader("📋 Trade Details")
                        trades_df = pd.DataFrame([dataclasses.asdict(t) for t in trades])
        
                        # Format the dataframe for better display
                        trades_display = trades_df.copy()
                        trades_display['entry_time'] = pd.to_datetime(trades_display['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
                        trades_display['exit_time'] = pd.to_datetime(trades_display['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
                        trades_display['entry'] = trades_display['entry'].round(5)
                        trades_display['exit_price'] = trades_display['exit_price'].round(5)
                        trades_display['pnl'] = trades_display['pnl'].round(2)
                        trades_display['r_mult'] = trades_display['r_mult'].round(2)
                        
                        # Select and rename columns for display
                        display_cols = ['entry_time', 'exit_time', 'side', 'entry', 'exit_price', 'pnl', 'r_mult']
                        trades_display = trades_display[display_cols]
                        trades_display.columns = ['Entry Time', 'Exit Time', 'Side', 'Entry Price', 'Exit Price', 'P&L', 'R Multiple']
                        
                        # Color-code profitable vs losing trades
                        def color_pnl(val):
                            if isinstance(val, (int, float)):
                                return 'color: green' if val > 0 else 'color: red' if val < 0 else 'color: gray'
                            return ''
                        
                        styled_df = trades_display.style.applymap(color_pnl, subset=['P&L'])
                        st.dataframe(styled_df, use_container_width=True, height=400)
                        
                        st.write(f"**Total Trades:** {len(trades)} | **Winners:** {len([t for t in trades if t.pnl > 0])} | **Losers:** {len([t for t in trades if t.pnl <= 0])}")
                    else:
                        st.info("No trades generated with current parameters.")
                
                dt = time.time() - t0
                st.success(f"⚡ Completed in {dt:.2f} seconds")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.info("👆 Configure settings and click 'RUN BACKTEST' to start")
            
            # Show recent results preview
            st.subheader("📈 Recent Results Preview")
            show_recent_results_preview()

def show_recent_results_preview():
    """Show preview of recent results"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT symbol, total_return, total_trades, created_at
        FROM backtests 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 5
    ''', (st.session_state.user['id'],))
    
    recent = cursor.fetchall()
    conn.close()
    
    if recent:
        st.write("**Your Recent Results:**")
        for r in recent:
            symbol, ret, trades, date = r
            date_str = pd.to_datetime(date).strftime('%m/%d %H:%M')
            color = "🟢" if ret >= 0 else "🔴"
            st.write(f"{color} {symbol}: {ret:.1f}% ({trades} trades) - {date_str}")
    else:
        st.write("No recent results. Run your first backtest!")

def show_live_demo():
    """Live demo functionality"""
    st.header("🔴 Live Trading Demo")
    
    col_live_left, col_live_right = st.columns([1, 3])
    
    with col_live_left:
        st.subheader("📊 Live Settings")
        
        asset_type = st.radio("Asset Type", ["Crypto", "Forex"], key="live_asset")
        if asset_type == "Crypto":
            live_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD"]
        else:
            live_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "GBPJPY"]
        
        selected_symbol = st.selectbox("Select Symbol", live_symbols, key="live_symbol")
        
        st.subheader("🎯 Strategy Settings")
        ma_short = st.slider("MA Short", 5, 20, 5, key="live_ma_short")
        ma_long = st.slider("MA Long", 10, 50, 20, key="live_ma_long")
        rsi_period = st.slider("RSI Period", 10, 30, 14, key="live_rsi")
        
        auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=True, key="live_auto_refresh")
        refresh_button = st.button("🔄 Manual Refresh", key="live_refresh")
        
        st.write(f"⏰ **Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    with col_live_right:
        st.subheader(f"📈 {selected_symbol} - Live Price Movement")
        
        try:
            live_data = get_live_price(selected_symbol)
            
            if not live_data.empty:
                current_price = live_data['Close'].iloc[-1]
                previous_price = live_data['Close'].iloc[-2] if len(live_data) > 1 else current_price
                price_change = current_price - previous_price
                price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
                
                # Live price metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Current Price", f"{current_price:.5f}")
                col_b.metric("Change", f"{price_change:.5f}", f"{price_change_pct:.2f}%")
                col_c.metric("24h High", f"{live_data['High'].max():.5f}")
                col_d.metric("24h Low", f"{live_data['Low'].min():.5f}")
                
                # Additional crypto info
                if asset_type == "Crypto":
                    crypto_info = get_crypto_info(selected_symbol)
                    if crypto_info['market_cap'] > 0:
                        st.write(f"💰 **Market Cap:** ${crypto_info['market_cap']:,.0f} | **Volume:** {crypto_info['24h_volume']:,.0f}")
                
                # Calculate indicators
                live_data['MA_Short'] = live_data['Close'].rolling(ma_short).mean()
                live_data['MA_Long'] = live_data['Close'].rolling(ma_long).mean()
                
                # RSI calculation
                delta = live_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                live_data['RSI'] = 100 - (100 / (1 + rs))
                
                # Live chart
                fig = go.Figure()
                
                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=live_data.index,
                    open=live_data['Open'],
                    high=live_data['High'], 
                    low=live_data['Low'],
                    close=live_data['Close'],
                    name=selected_symbol
                ))
                
                # Moving averages
                if not live_data['MA_Short'].isna().all():
                    fig.add_trace(go.Scatter(
                        x=live_data.index, y=live_data['MA_Short'],
                        name=f"MA{ma_short}", line=dict(color="orange", width=2)
                    ))
                
                if not live_data['MA_Long'].isna().all():
                    fig.add_trace(go.Scatter(
                        x=live_data.index, y=live_data['MA_Long'],
                        name=f"MA{ma_long}", line=dict(color="blue", width=2)
                    ))
                
                fig.update_layout(
                    title=f"{selected_symbol} - Live Trading with Signals",
                    xaxis_title="Time",
                    yaxis_title="Price", 
                    height=500,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Current market analysis
                st.subheader("🎯 Current Market Analysis")
                
                col_signal1, col_signal2, col_signal3 = st.columns(3)
                
                # MA Signal
                if not live_data['MA_Short'].isna().all() and not live_data['MA_Long'].isna().all():
                    current_ma_short = live_data['MA_Short'].iloc[-1]
                    current_ma_long = live_data['MA_Long'].iloc[-1]
                    
                    with col_signal1:
                        if current_ma_short > current_ma_long:
                            st.success(f"🟢 **BULLISH** MA\nMA{ma_short} > MA{ma_long}")
                        else:
                            st.error(f"🔴 **BEARISH** MA\nMA{ma_short} < MA{ma_long}")
                
                # RSI Signal
                if not live_data['RSI'].isna().all():
                    current_rsi = live_data['RSI'].iloc[-1]
                    with col_signal2:
                        if current_rsi > 70:
                            st.warning(f"🟡 **OVERBOUGHT**\nRSI: {current_rsi:.1f}")
                        elif current_rsi < 30:
                            st.info(f"🔵 **OVERSOLD**\nRSI: {current_rsi:.1f}")
                        else:
                            st.write(f"⚪ **NEUTRAL**\nRSI: {current_rsi:.1f}")
                
                # Volume Signal
                with col_signal3:
                    if len(live_data) >= 10:
                        avg_volume = live_data['Volume'].rolling(10).mean().iloc[-1]
                        current_volume = live_data['Volume'].iloc[-1]
                        if current_volume > avg_volume * 1.5:
                            st.success("🟢 **HIGH VOLUME**\nAbove Average")
                        else:
                            st.write("⚪ **NORMAL VOLUME**")
                
            else:
                st.error("❌ No live data available. Try a different symbol.")
                
        except Exception as e:
            st.error(f"❌ Error fetching live  {str(e)}")

def show_user_results_history():
    """Show user's complete backtesting history"""
    st.header("📈 My Trading Results History")
    
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    # Get user's backtests
    cursor.execute('''
        SELECT id, symbol, total_return, final_equity, initial_equity, total_trades, 
               win_rate, profit_factor, max_drawdown, created_at
        FROM backtests 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (st.session_state.user['id'],))
    
    backtests = cursor.fetchall()
    conn.close()
    
    if backtests:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_backtests = len(backtests)
        avg_return = sum(b[2] for b in backtests) / total_backtests
        best_return = max(b[2] for b in backtests)
        total_trades = sum(b[5] for b in backtests)
        
        col1.metric("Total Backtests", total_backtests)
        col2.metric("Average Return", f"{avg_return:.2f}%")
        col3.metric("Best Return", f"{best_return:.2f}%")
        col4.metric("Total Trades", total_trades)
        
        # Display results table
        st.subheader("📊 Backtest Results")
        df = pd.DataFrame(backtests, columns=[
            'ID', 'Symbol', 'Return %', 'Final Equity', 'Initial Equity', 'Trades', 
            'Win Rate %', 'Profit Factor', 'Max Drawdown %', 'Date'
        ])
        
        # Format the dataframe
        df['Return %'] = df['Return %'].round(2)
        df['Win Rate %'] = df['Win Rate %'].round(2)
        df['Profit Factor'] = df['Profit Factor'].round(2)
        df['Max Drawdown %'] = df['Max Drawdown %'].round(2)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(df, use_container_width=True, height=400)
        
        # Show details for selected backtest
        st.subheader("📋 Backtest Details")
        if backtests:
            selected_index = st.selectbox(
                "Select backtest to view details", 
                range(len(backtests)), 
                format_func=lambda x: f"ID {backtests[x][0]} - {backtests[x][1]} ({backtests[x][2]:.2f}%)"
            )
            
            if selected_index is not None:
                show_backtest_trade_details(backtests[selected_index][0])
    else:
        st.info("📊 No backtests saved yet. Run some backtests to see results here!")
        st.markdown("### 🚀 Get Started:")
        st.markdown("1. Go to the **Backtesting** tab")
        st.markdown("2. Select a market and symbol")
        st.markdown("3. Configure strategy parameters")
        st.markdown("4. Run backtests and view results here")

def show_backtest_trade_details(backtest_id):
    """Show detailed trades for a specific backtest"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    # Get trades for this backtest
    cursor.execute('''
        SELECT entry_time, exit_time, side, entry_price, exit_price, pnl, r_mult, size
        FROM trades 
        WHERE backtest_id = ?
        ORDER BY entry_time
    ''', (backtest_id,))
    
    trades = cursor.fetchall()
    conn.close()
    
    if trades:
        trades_df = pd.DataFrame(trades, columns=[
            'Entry Time', 'Exit Time', 'Side', 'Entry Price', 
            'Exit Price', 'P&L', 'R Multiple', 'Size'
        ])
        
        # Format the trades
        trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time']).dt.strftime('%Y-%m-%d %H:%M')
        trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time']).dt.strftime('%Y-%m-%d %H:%M')
        trades_df['Entry Price'] = trades_df['Entry Price'].round(5)
        trades_df['Exit Price'] = trades_df['Exit Price'].round(5)
        trades_df['P&L'] = trades_df['P&L'].round(2)
        trades_df['R Multiple'] = trades_df['R Multiple'].round(2)
        trades_df['Size'] = trades_df['Size'].round(2)
        
        # Color-code by profitability
        def highlight_pnl(row):
            if row['P&L'] > 0:
                return ['background-color: #90EE90'] * len(row)  # Light green
            elif row['P&L'] < 0:
                return ['background-color: #FFB6C1'] * len(row)  # Light red
            else:
                return [''] * len(row)
        
        styled_df = trades_df.style.apply(highlight_pnl, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Trade summary
        profitable_trades = len([t for t in trades if t[5] > 0])  # P&L > 0
        losing_trades = len([t for t in trades if t[5] <= 0])
        total_pnl = sum(t[5] for t in trades)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", len(trades))
        col2.metric("Profitable", profitable_trades)
        col3.metric("Losing", losing_trades)
        col4.metric("Total P&L", f"${total_pnl:.2f}")
    else:
        st.info("No trade details found for this backtest.")

if __name__ == "__main__":
    main()
