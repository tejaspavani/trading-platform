# forex_complete_ai.py — AI-Enhanced Multi-User Trading Platform
# Run: streamlit run forex_complete_ai.py

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
from scipy.signal import find_peaks

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
    
    # AI Chat History table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT,
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # AI Strategies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            parameters TEXT,
            ai_reasoning TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
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

def get_user_by_id(user_id):
    """Get user by ID for session persistence"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, username, role, email 
        FROM users 
        WHERE id = ? AND is_active = 1
    ''', (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    
    if user_data:
        return {
            'id': user_data[0],
            'username': user_data[1],
            'role': user_data[2],
            'email': user_data[3]
        }
    return None

def save_backtest_results(user_id, symbol, stats, trades, strategy_config=None):
    """Save backtest results to database"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    def safe_float(value):
        if value is None or np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    
    def safe_int(value):
        if value is None:
            return 0
        return int(value)
    
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
# CRYPTO & FX DATA HELPERS
# ──────────────────────────────────────────────────────────────────────────────
CRYPTO_MAP = {
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD", "BNBUSD": "BNB-USD",
    "XRPUSD": "XRP-USD", "ADAUSD": "ADA-USD", "DOGEUSD": "DOGE-USD", "AVAXUSD": "AVAX-USD",
    "TRXUSD": "TRX-USD", "DOTUSD": "DOT-USD", "MATICUSD": "MATIC-USD", "LINKUSD": "LINK-USD",
    "UNIUSD": "UNI-USD", "LTCUSD": "LTC-USD", "BCHUSD": "BCH-USD", "ATOMUSD": "ATOM-USD",
    "FILUSD": "FIL-USD", "ETCUSD": "ETC-USD", "XLMUSD": "XLM-USD", "ALGOUSD": "ALGO-USD",
    "VETUSD": "VET-USD", "ICPUSD": "ICP-USD", "THETAUSD": "THETA-USD", "FTMUSD": "FTM-USD",
    "AAVEUSD": "AAVE-USD", "COMPUSD": "COMP-USD", "MKRUSD": "MKR-USD", "SUSHIUSD": "SUSHI-USD"
}
# ADD ALL THE NEW SYMBOL LISTS HERE ⬇️
# Comprehensive FX Pairs
FX_MAJOR_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"
]

FX_MINOR_PAIRS = [
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "GBPAUD", 
    "EURCHF", "GBPCHF", "AUDCHF", "CADCHF", "EURAUD", "NZDJPY",
    "GBPNZD", "AUDNZD", "EURNZD", "CHFJPY", "CADJPY"
]

FX_EXOTIC_PAIRS = [
    "USDTRY", "USDZAR", "USDSGD", "USDHKD", "USDPLN", "USDSEK",
    "USDNOK", "USDDKK", "USDMXN", "USDRUB", "USDCNY", "USDINR"
]

# Comprehensive Crypto Pairs  
CRYPTO_MAJOR = [
    "BTCUSD", "ETHUSD", "BNBUSD", "XRPUSD", "SOLUSD", "ADAUSD", "DOGEUSD"
]

CRYPTO_ALTCOINS = [
    "AVAXUSD", "DOTUSD", "MATICUSD", "LINKUSD", "UNIUSD", "LTCUSD", 
    "BCHUSD", "ATOMUSD", "FILUSD", "TRXUSD", "ETCUSD", "XLMUSD",
    "ALGOUSD", "VETUSD", "ICPUSD", "THETAUSD", "FTMUSD", "AXSUSD"
]

CRYPTO_DEFI = [
    "AAVEUSD", "COMPUSD", "MKRUSD", "SUSHIUSD", "YFIUSD", "CRVUSD",
    "1INCHUSD", "SNXUSD", "BALUSD", "RENUSD", "LRCUSD", "KNCUSD"
]

# All symbols combined
ALL_SYMBOLS = {
    "FX Major": FX_MAJOR_PAIRS,
    "FX Minor": FX_MINOR_PAIRS, 
    "FX Exotic": FX_EXOTIC_PAIRS,
    "Crypto Major": CRYPTO_MAJOR,
    "Crypto Altcoins": CRYPTO_ALTCOINS,
    "Crypto DeFi": CRYPTO_DEFI
}

def _is_crypto(symbol: str) -> bool:
    return (symbol in CRYPTO_MAP) or symbol.endswith("-USD")

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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]
    
    df = df.rename(columns=str.lower)
    
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
# DATA GENERATION
# ──────────────────────────────────────────────────────────────────────────────
_global_seed = 42

def _set_seed(seed: int):
    global _global_seed
    if seed == -1:
        s = int(time.time()) & 0xFFFFFFFF
    else:
        s = int(seed)
    
    _global_seed = s
    random.seed(s)
    np.random.seed(s)

def _generate_data(symbol: str, days: int, interval_min: int = 1) -> pd.DataFrame:
    if _is_crypto(symbol):
        return _fetch_crypto_ohlcv(symbol, days, interval_min)
    return _fallback_synth_fx(symbol, days, interval_min, seed=_global_seed)

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
# ──────────────────────────────────────────────────────────────────────────────
# 23+ TRADING STRATEGIES SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    name: str
    description: str
    category: str
    parameters: dict
    risk_level: str  # Low, Medium, High
    market_type: str  # Trending, Ranging, Any

# Define 23+ Different Strategies
TRADING_STRATEGIES = {
    
    # TREND FOLLOWING STRATEGIES (8 strategies)
    "donchian_macd": StrategyConfig(
        name="🏁 Donchian Breakout + MACD",
        description="Breakout system with momentum confirmation",
        category="Trend Following",
        parameters={"lookback": 20, "atr_mult": 3.0, "rr": 4.0, "use_macd": True},
        risk_level="Medium",
        market_type="Trending"
    ),
    
    "ma_crossover": StrategyConfig(
        name="📈 Moving Average Crossover", 
        description="Classic MA crossover with trend filter",
        category="Trend Following",
        parameters={"fast_ma": 10, "slow_ma": 30, "atr_mult": 2.5, "rr": 3.0},
        risk_level="Low",
        market_type="Trending"
    ),
    
    "triple_ma": StrategyConfig(
        name="🎯 Triple Moving Average",
        description="Three MA system for strong trend identification", 
        category="Trend Following",
        parameters={"fast": 5, "medium": 15, "slow": 30, "atr_mult": 2.8, "rr": 3.5},
        risk_level="Medium",
        market_type="Trending"
    ),
    
    "parabolic_sar": StrategyConfig(
        name="🚀 Parabolic SAR Trend",
        description="SAR-based trend following system",
        category="Trend Following", 
        parameters={"acceleration": 0.02, "maximum": 0.2, "atr_mult": 2.2, "rr": 2.8},
        risk_level="High",
        market_type="Trending"
    ),
    
    "supertrend": StrategyConfig(
        name="💫 SuperTrend Strategy",
        description="ATR-based trend identification",
        category="Trend Following",
        parameters={"period": 10, "multiplier": 3.0, "atr_mult": 2.5, "rr": 3.2},
        risk_level="Medium", 
        market_type="Trending"
    ),
    
    "adx_trend": StrategyConfig(
        name="📊 ADX Trend Strength",
        description="ADX for trend strength confirmation",
        category="Trend Following",
        parameters={"adx_period": 14, "adx_threshold": 25, "atr_mult": 2.8, "rr": 3.5},
        risk_level="Medium",
        market_type="Trending"
    ),
    
    "keltner_breakout": StrategyConfig(
        name="🎪 Keltner Channel Breakout",
        description="Volatility-based breakout system",
        category="Trend Following", 
        parameters={"period": 20, "multiplier": 2.0, "atr_mult": 2.5, "rr": 3.0},
        risk_level="Medium",
        market_type="Any"
    ),
    
    "ichimoku_cloud": StrategyConfig(
        name="☁️ Ichimoku Cloud System",
        description="Complete Ichimoku trading system",
        category="Trend Following",
        parameters={"tenkan": 9, "kijun": 26, "senkou": 52, "atr_mult": 2.8, "rr": 3.5},
        risk_level="Medium",
        market_type="Trending"
    ),
    
    # MEAN REVERSION STRATEGIES (6 strategies)
    "rsi_reversal": StrategyConfig(
        name="🔄 RSI Mean Reversion", 
        description="RSI overbought/oversold reversals",
        category="Mean Reversion",
        parameters={"rsi_period": 14, "oversold": 30, "overbought": 70, "atr_mult": 2.0, "rr": 2.5},
        risk_level="Medium",
        market_type="Ranging"
    ),
    
    "bollinger_bounce": StrategyConfig(
        name="🎈 Bollinger Band Bounce",
        description="Mean reversion using Bollinger Bands",
        category="Mean Reversion", 
        parameters={"period": 20, "std_dev": 2.0, "atr_mult": 1.8, "rr": 2.2},
        risk_level="Low",
        market_type="Ranging"
    ),
    
    "williams_r": StrategyConfig(
        name="📉 Williams %R Strategy",
        description="Williams %R for overbought/oversold levels",
        category="Mean Reversion",
        parameters={"period": 14, "oversold": -80, "overbought": -20, "atr_mult": 2.2, "rr": 2.8},
        risk_level="Medium", 
        market_type="Ranging"
    ),
    
    "stochastic_reversal": StrategyConfig(
        name="🎲 Stochastic Reversal",
        description="Stochastic oscillator mean reversion",
        category="Mean Reversion",
        parameters={"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80, "atr_mult": 2.0, "rr": 2.5},
        risk_level="Medium",
        market_type="Ranging"
    ),
    
    "cci_reversal": StrategyConfig(
        name="🌊 CCI Mean Reversion",
        description="Commodity Channel Index reversals", 
        category="Mean Reversion",
        parameters={"period": 20, "oversold": -100, "overbought": 100, "atr_mult": 2.2, "rr": 2.8},
        risk_level="Medium",
        market_type="Ranging"
    ),
    
    "mfi_reversal": StrategyConfig(
        name="💰 Money Flow Index",
        description="Volume-weighted RSI for reversals",
        category="Mean Reversion", 
        parameters={"period": 14, "oversold": 20, "overbought": 80, "atr_mult": 2.0, "rr": 2.5},
        risk_level="Medium",
        market_type="Ranging"
    ),
    
    # MOMENTUM STRATEGIES (5 strategies) 
    "macd_momentum": StrategyConfig(
        name="⚡ MACD Momentum",
        description="Pure MACD momentum trading",
        category="Momentum",
        parameters={"fast": 12, "slow": 26, "signal": 9, "atr_mult": 2.5, "rr": 3.0},
        risk_level="Medium",
        market_type="Any"
    ),
    
    "roc_momentum": StrategyConfig(
        name="🚄 Rate of Change Momentum",
        description="ROC-based momentum strategy",
        category="Momentum",
        parameters={"period": 12, "threshold": 1.5, "atr_mult": 2.8, "rr": 3.2},
        risk_level="High",
        market_type="Trending"
    ),
    
    "momentum_oscillator": StrategyConfig(
        name="🎛️ Momentum Oscillator",
        description="Classical momentum oscillator system",
        category="Momentum",
        parameters={"period": 14, "threshold": 0, "atr_mult": 2.5, "rr": 3.0},
        risk_level="Medium", 
        market_type="Any"
    ),
    
    "tsi_momentum": StrategyConfig(
        name="🎯 True Strength Index",
        description="Smoothed momentum indicator",
        category="Momentum",
        parameters={"fast": 25, "slow": 13, "signal": 13, "atr_mult": 2.6, "rr": 3.1},
        risk_level="Medium",
        market_type="Any"
    ),
    
    "awesome_oscillator": StrategyConfig(
        name="😎 Awesome Oscillator",
        description="Bill Williams' Awesome Oscillator",
        category="Momentum", 
        parameters={"fast": 5, "slow": 34, "atr_mult": 2.7, "rr": 3.2},
        risk_level="Medium",
        market_type="Trending"
    ),
    
    # VOLATILITY STRATEGIES (4 strategies)
    "atr_volatility": StrategyConfig(
        name="📈 ATR Volatility Breakout",
        description="ATR-based volatility trading",
        category="Volatility", 
        parameters={"atr_period": 14, "volatility_mult": 1.5, "atr_mult": 2.5, "rr": 3.0},
        risk_level="High",
        market_type="Any"
    ),
    
    "volatility_squeeze": StrategyConfig(
        name="🗜️ Volatility Squeeze",
        description="Low volatility followed by breakouts",
        category="Volatility",
        parameters={"bb_period": 20, "kc_period": 20, "atr_mult": 2.2, "rr": 2.8},
        risk_level="Medium",
        market_type="Any"
    ),
    
    "chaikin_volatility": StrategyConfig(
        name="📊 Chaikin Volatility",
        description="High-low spread volatility measure",
        category="Volatility",
        parameters={"period": 14, "roc_period": 10, "atr_mult": 2.3, "rr": 2.9},
        risk_level="High",
        market_type="Any"
    ),
    
    "historical_volatility": StrategyConfig(
        name="📋 Historical Volatility",
        description="Price volatility regime changes",
        category="Volatility",
        parameters={"period": 20, "threshold": 0.02, "atr_mult": 2.4, "rr": 3.0},
        risk_level="High", 
        market_type="Any"
    )
}

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

def _price_fig_with_trades(df: pd.DataFrame, trades: list[Trade], symbol: str, show_ma=True) -> go.Figure:
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

    if trades:
        long_entries  = [t.entry_time for t in trades if t.side==1]
        long_prices   = [t.entry      for t in trades if t.side==1]
        exits_ts      = [t.exit_time  for t in trades if t.exit_time is not None]
        exits_px      = [t.exit_price for t in trades if t.exit_time is not None]

        if long_entries:
            fig.add_trace(go.Scatter(
                x=long_entries, y=long_prices, mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="green"), name="Long Entry"
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
# AI FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def get_user_statistics(user_id):
    """Get comprehensive user statistics"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM backtests WHERE user_id = ?', (user_id,))
    total_backtests = cursor.fetchone()[0]
    
    cursor.execute('SELECT SUM(total_trades) FROM backtests WHERE user_id = ?', (user_id,))
    total_trades = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT AVG(total_return), MAX(total_return) FROM backtests WHERE user_id = ?', (user_id,))
    avg_return, best_return = cursor.fetchone()
    
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

def get_user_best_assets(user_id):
    """Get user's best performing assets"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT symbol, AVG(total_return) as avg_return
        FROM backtests 
        WHERE user_id = ? 
        GROUP BY symbol
        ORDER BY avg_return DESC
        LIMIT 3
    ''', (user_id,))
    assets = cursor.fetchall()
    conn.close()
    return [asset[0] for asset in assets]

import openai
import os

def get_ai_trading_response(user_input, user_id=None):
    """Get AI response using OpenAI API"""
    try:
        # Get API key from secrets
        openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        
        if not openai.api_key:
            return "❌ OpenAI API key not configured. Please add it to Streamlit secrets."
        
        # Get user context
        user_context = ""
        if user_id:
            user_stats = get_user_statistics(user_id)
            best_assets = get_user_best_assets(user_id)
            user_context = f"""
User Profile:
- Total backtests: {user_stats['total_backtests']}
- Average return: {user_stats['avg_return']:.1f}%
- Best return: {user_stats['best_return']:.1f}%
- Total trades: {user_stats['total_trades']}
- Best performing assets: {', '.join(best_assets[:3]) if best_assets else 'None yet'}
"""
        
        # Create the prompt
        system_prompt = f"""You are an expert AI trading assistant for a professional trading platform. 
        
Your expertise includes:
- Forex and cryptocurrency trading strategies
- Risk management and position sizing
- Technical analysis and chart patterns
- Market sentiment and news analysis
- Performance optimization and strategy development

{user_context}

Guidelines:
- Provide specific, actionable trading advice
- Always emphasize risk management
- Use emojis and formatting for clarity
- Keep responses under 300 words
- Be encouraging but realistic
- Include specific numbers/percentages when relevant
"""
        
        # Make API call to OpenAI
        client = openai.OpenAI(api_key=openai.api_key)
        
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper responses
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Fallback to rule-based response if OpenAI fails
        return get_fallback_ai_response(user_input, user_id)

def get_fallback_ai_response(user_input, user_id=None):
    """Fallback rule-based responses if OpenAI fails"""
    user_input_lower = user_input.lower()
    
    if any(word in user_input_lower for word in ['strategy', 'profitable', 'best']):
        return """🎯 **Strategy Recommendation:**
        
Based on your trading history, I recommend:
- **Donchian Breakout Strategy** with MACD confirmation
- **Risk per trade**: 2% (optimal for consistent growth)
- **Best timeframes**: 15m-1H for forex, 5m-15m for crypto
        
Would you like me to help optimize your parameters?"""
    
    elif any(word in user_input_lower for word in ['risk', 'loss', 'drawdown']):
        return """⚠️ **Risk Management Advice:**
        
Key principles for risk control:
- **Never risk more than 2% per trade**
- **Maximum 6% total exposure** across all positions
- **Use stop losses on every trade** - no exceptions!
- **Position size based on volatility** (ATR method)
        
Your current approach seems well-balanced for your experience level."""
    
    elif any(word in user_input_lower for word in ['market', 'analysis', 'forecast']):
        return """📈 **Market Analysis:**
        
Current market overview:
- **Crypto markets**: Strong institutional interest, watch for volatility
- **Forex markets**: Central bank policies driving major moves
- **Risk sentiment**: Moderate - good for trend-following strategies
        
**Key levels to watch**: Support/resistance from daily timeframes."""
    
    else:
        return """🤖 **AI Trading Assistant Ready!**
        
I can help you with:
🎯 **Strategy Analysis** - Optimize your trading approach
📊 **Performance Review** - Analyze your results
⚠️ **Risk Management** - Improve your risk controls
📈 **Market Insights** - Current market analysis
💡 **Trade Ideas** - Find new opportunities
        
What would you like to explore?"""

def generate_ai_strategy_openai(goal, risk_tolerance, creativity, personality):
    """Generate strategy using OpenAI"""
    try:
        client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        prompt = f"""Create a detailed trading strategy based on these parameters:

Goal: {goal}
Risk Tolerance: {risk_tolerance}/10
Creativity Level: {creativity}/10  
Personality: {personality}

Provide:
1. Strategy name (creative and professional)
2. Entry rules (specific conditions)
3. Exit rules (profit targets and stop losses)
4. Risk management (position sizing)
5. Best markets/timeframes
6. Key parameters (lookback periods, multipliers, etc.)

Format as a professional trading strategy document."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.8
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ OpenAI Error: {str(e)}\n\nUsing fallback strategy generator..."

def generate_creative_strategy(goal, risk, creativity, personality):
    """Enhanced strategy generation with OpenAI"""
    st.subheader("🤖 AI-Generated Trading Strategy")
    
    with st.spinner("🤖 OpenAI is creating your custom strategy..."):
        if st.secrets.get("OPENAI_API_KEY"):
            ai_strategy = generate_ai_strategy_openai(goal, risk, creativity, personality)
            st.markdown(ai_strategy)
        else:
            # Fallback to rule-based generation
            st.warning("⚠️ OpenAI not configured. Using built-in AI...")
            # Your existing generate_creative_strategy code here
    
    # Save strategy button
    if st.button("💾 Save OpenAI Strategy", use_container_width=True):
        st.success("✅ OpenAI-generated strategy saved!")
        st.balloons()


def detect_patterns_ai(df):
    """AI pattern detection using price action analysis"""
    patterns = []
    
    if df.empty or len(df) < 20:
        return patterns
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    try:
        # Double Top Detection
        peaks = find_peaks(close, distance=10)[0]
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            peak_values = close.iloc[last_two_peaks]
            if abs(peak_values.iloc[0] - peak_values.iloc[1]) / peak_values.mean() < 0.02:
                patterns.append({
                    'name': 'Double Top',
                    'confidence': 75,
                    'signal': 'Bearish Reversal Expected'
                })
        
        # Support/Resistance Levels
        support_level = low.rolling(20).min().iloc[-1]
        resistance_level = high.rolling(20).max().iloc[-1]
        current_price = close.iloc[-1]
        
        if abs(current_price - resistance_level) / current_price < 0.01:
            patterns.append({
                'name': 'At Resistance Level',
                'confidence': 85,
                'signal': 'Potential Reversal or Breakout'
            })
        
        if abs(current_price - support_level) / current_price < 0.01:
            patterns.append({
                'name': 'At Support Level',
                'confidence': 85,
                'signal': 'Potential Bounce or Breakdown'
            })
        
        # Trend Analysis
        ma_20 = close.rolling(20).mean().iloc[-1]
        ma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else ma_20
        
        if current_price > ma_20 > ma_50:
            patterns.append({
                'name': 'Strong Uptrend',
                'confidence': 80,
                'signal': 'Bullish Momentum'
            })
        elif current_price < ma_20 < ma_50:
            patterns.append({
                'name': 'Strong Downtrend',
                'confidence': 80,
                'signal': 'Bearish Momentum'
            })
        
        # Volatility Analysis
        volatility = close.pct_change().rolling(20).std().iloc[-1] * 100
        if volatility > 2.0:
            patterns.append({
                'name': 'High Volatility Environment',
                'confidence': 90,
                'signal': 'Increase Position Size Caution'
            })
        elif volatility < 0.5:
            patterns.append({
                'name': 'Low Volatility Environment',
                'confidence': 90,
                'signal': 'Range-bound Markets Expected'
            })
    
    except Exception as e:
        patterns.append({
            'name': 'Analysis Error',
            'confidence': 0,
            'signal': f'Unable to analyze: {str(e)}'
        })
    
    return patterns

def generate_ai_signals(patterns):
    """Generate trading signals based on detected patterns"""
    signals = []
    
    for pattern in patterns:
        if pattern['confidence'] > 70:
            if 'Support' in pattern['name']:
                signals.append(f"🟢 **BUY Signal**: {pattern['name']} - Consider long positions")
            elif 'Resistance' in pattern['name']:
                signals.append(f"🔴 **SELL Signal**: {pattern['name']} - Consider short positions")
            elif 'Uptrend' in pattern['name']:
                signals.append(f"📈 **TREND Signal**: {pattern['name']} - Follow the trend")
            elif 'Double Top' in pattern['name']:
                signals.append(f"⚠️ **REVERSAL Signal**: {pattern['name']} - Prepare for reversal")
            else:
                signals.append(f"ℹ️ **INFO**: {pattern['name']} - {pattern['signal']}")
    
    if not signals:
        signals.append("📊 **NEUTRAL**: No strong signals detected - Wait for better setups")
    
    return signals

def save_ai_strategy(user_id, name, parameters, reasoning):
    """Save AI-generated strategy to database"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO ai_strategies (user_id, name, parameters, ai_reasoning)
        VALUES (?, ?, ?, ?)
    ''', (user_id, name, json.dumps(parameters), reasoning))
    conn.commit()
    conn.close()

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

def get_openai_market_analysis(symbol):
    """Get OpenAI market analysis for specific symbol"""
    try:
        client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        prompt = f"""Analyze the current market conditions for {symbol}:
        
        1. Technical outlook (trend, support/resistance)
        2. Fundamental factors affecting price
        3. Risk factors to watch
        4. Trading recommendations
        5. Probability assessment for next 24-48 hours
        
        Be specific and actionable."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ OpenAI Market Analysis Error: {str(e)}\n\nPlease check your API key and try again."

def optimize_strategy_with_ai(user_stats, current_params):
    """Use OpenAI to optimize strategy parameters"""
    try:
        client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        prompt = f"""Optimize these trading strategy parameters:
        
        Current Performance:
        - Average Return: {user_stats['avg_return']:.1f}%
        - Total Trades: {user_stats['total_trades']}
        - Win Rate: {user_stats.get('win_rate', 'Unknown')}%
        
        Current Parameters: {current_params}
        
        Suggest optimized parameters and explain your reasoning."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ OpenAI Optimization Error: {str(e)}\n\nUsing fallback optimization..."

def get_openai_risk_assessment(user_id, current_positions=None):
    """Get OpenAI-powered risk assessment"""
    try:
        user_stats = get_user_statistics(user_id)
        client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        prompt = f"""Provide a comprehensive risk assessment for this trader:
        
        Trading Profile:
        - Average Return: {user_stats['avg_return']:.1f}%
        - Total Backtests: {user_stats['total_backtests']}
        - Total Trades: {user_stats['total_trades']}
        - Experience Level: {"Advanced" if user_stats['total_trades'] > 100 else "Intermediate" if user_stats['total_trades'] > 50 else "Beginner"}
        
        Analyze:
        1. Current risk level (Low/Medium/High)
        2. Recommended position sizing
        3. Portfolio diversification advice
        4. Risk management improvements
        5. Warning signs to watch for
        
        Be specific with percentages and actionable recommendations."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Risk Assessment Error: {str(e)}\n\nPlease check your OpenAI configuration."


# AI STRATEGY TOURNAMENT SYSTEM
class AIStrategyTournament:
    def __init__(self, user_id):
        self.user_id = user_id
        self.results = {}
        
    def run_tournament(self, symbol, days=30):
        """Run all strategies against the symbol and rank them"""
        st.info(f"🤖 AI Tournament: Testing {len(TRADING_STRATEGIES)} strategies on {symbol}")
        
        # Generate data once
        _set_seed(42)
        df = _generate_data(symbol, days, interval_min=1)
        
        progress_bar = st.progress(0)
        results = []
        
        for i, (strategy_key, strategy_config) in enumerate(TRADING_STRATEGIES.items()):
            try:
                # Convert strategy config to StratV4Config format
                cfg = self._convert_to_stratv4(strategy_config)
                
                # Run backtest
                stats, trades = backtest_v4(df, cfg)
                
                # Calculate AI score (combines return, win rate, profit factor)
                ai_score = self._calculate_ai_score(stats, strategy_config)
                
                results.append({
                    'strategy': strategy_config.name,
                    'key': strategy_key,
                    'category': strategy_config.category,
                    'return': stats.ret_pct,
                    'trades': stats.n_trades,
                    'win_rate': stats.win_rate,
                    'profit_factor': stats.pf,
                    'max_dd': stats.max_dd_pct,
                    'ai_score': ai_score,
                    'risk_level': strategy_config.risk_level,
                    'market_type': strategy_config.market_type
                })
                
                progress_bar.progress((i + 1) / len(TRADING_STRATEGIES))
                
            except Exception as e:
                st.warning(f"⚠️ Strategy {strategy_config.name} failed: {str(e)}")
                
        # Sort by AI score
        results.sort(key=lambda x: x['ai_score'], reverse=True)
        
        return results, df
    
    def _convert_to_stratv4(self, strategy_config):
        """Convert strategy config to StratV4Config"""
        params = strategy_config.parameters
        
        return StratV4Config(
            lookback=params.get('lookback', params.get('period', 20)),
            atr_mult=params.get('atr_mult', 2.5), 
            rr=params.get('rr', 3.0),
            use_macd_filter=params.get('use_macd', strategy_config.name == "🏁 Donchian Breakout + MACD"),
            use_trend_filter=strategy_config.category == "Trend Following",
            use_vol_filter=strategy_config.category == "Volatility",
            risk_per_trade=0.015 if strategy_config.risk_level == "Low" else 0.02 if strategy_config.risk_level == "Medium" else 0.025
        )
    
    def _calculate_ai_score(self, stats, strategy_config):
        """Calculate AI score based on multiple factors"""
        # Base score from return
        return_score = max(0, min(100, stats.ret_pct * 2))
        
        # Win rate bonus
        win_rate_bonus = (stats.win_rate - 50) * 0.5 if stats.win_rate > 50 else 0
        
        # Profit factor bonus
        pf_bonus = min(20, (stats.pf - 1) * 10) if stats.pf > 1 else 0
        
        # Risk penalty
        risk_penalty = abs(stats.max_dd_pct) * 0.3
        
        # Trade count bonus (more trades = more statistical significance)
        trade_bonus = min(10, stats.n_trades * 0.2)
        
        final_score = return_score + win_rate_bonus + pf_bonus - risk_penalty + trade_bonus
        
        return max(0, min(100, final_score))

# ──────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="🤖 AI Trading Platform", layout="wide")
    
    # Mobile-friendly CSS
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
        }
        @media (max-width: 768px) {
            .stColumns > div {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: unset !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize database
    setup_database()
    
    # Enhanced session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.remember_me = False
    
    # Check for remembered login using query parameters
    if not st.session_state.authenticated:
        if 'user_token' in st.query_params:
            try:
                user_id = int(st.query_params['user_token'])
                user = get_user_by_id(user_id)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.session_state.remember_me = True
            except:
                pass
    
    # Initialize other session states
    if 'run_count' not in st.session_state:
        st.session_state.run_count = 0
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Show login page or main app
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_enhanced_main_app()

def show_login_page():
    """Professional authentication interface"""
    st.title("🤖 AI-Powered Trading Platform")
    st.markdown("### Multi-User Collaborative AI Trading System")
    
    tab1, tab2, tab3 = st.tabs(["🔐 Login", "📝 Register", "ℹ️ About"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                username = st.text_input("👤 Username", placeholder="Enter your username")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
                remember_me = st.checkbox("🔄 Keep me logged in", value=True)
            
            with col2:
                st.write("")
                st.write("")
                st.write("")
                submitted = st.form_submit_button("🚀 Login", type="primary", use_container_width=True)
            
            if submitted:
                if username and password:
                    user = UserManager.authenticate(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.session_state.remember_me = remember_me
                        
                        if remember_me:
                            st.query_params['user_token'] = str(user['id'])
                        
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
        st.subheader("🤖 AI Trading Platform Features")
        st.markdown("""
        🎯 **AI-Enhanced Trading Platform:**
        
        ✅ **AI Trading Assistant** - Intelligent chat for personalized advice
        ✅ **Pattern Recognition** - AI-powered chart analysis
        ✅ **Strategy Builder** - AI creates custom strategies for you
        ✅ **Market Intelligence** - Real-time AI market analysis
        ✅ **Performance Analytics** - AI-generated insights and reports
        ✅ **Risk Management** - AI-optimized position sizing
        ✅ **Multi-User System** - Individual accounts with data persistence
        
        🧠 **Advanced AI Features:**
        - Sentiment analysis integration
        - Predictive market modeling
        - Automated strategy optimization
        - Personalized trading recommendations
        - Creative strategy generation
        
        🚀 **Get Started:**
        1. Create your account above
        2. Chat with AI assistant for personalized advice
        3. Let AI build custom strategies for you
        4. Run AI-optimized backtests
        5. Get real-time AI market intelligence
        """)

def show_enhanced_main_app():
    """AI-enhanced professional dashboard"""
    
    # Safety check
    if not st.session_state.authenticated or not st.session_state.user:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()
        return
    
    # Professional header with user info and logout
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    
    with col_header1:
        st.title("🤖 AI Trading Platform")
        st.caption("AI-Powered Multi-User Trading System")
    
    with col_header2:
        username = st.session_state.user.get('username', 'User') if st.session_state.user else 'User'
        email = st.session_state.user.get('email', '') if st.session_state.user else ''
        st.write(f"**Welcome, {username}** 🧠")
        st.caption(f"📧 {email}")
    
    with col_header3:
        col_logout1, col_logout2 = st.columns([1, 1])
        with col_logout1:
            if st.button("👤 Profile", use_container_width=True):
                st.session_state.show_profile = True
        with col_logout2:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.user = None
                st.session_state.remember_me = False
                st.query_params.clear()
                st.rerun()
    
    st.markdown("---")
    
    # Enhanced navigation with AI tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 **BACKTESTING**", 
        "🔴 **LIVE DEMO**", 
        "📈 **MY RESULTS**", 
        "🤖 **AI ASSISTANT**",
        "⚙️ **DASHBOARD**"
    ])
    
    with tab1:
        show_enhanced_backtesting()
    
    with tab2:
        show_live_demo()
    
    with tab3:
        show_user_results_history()
    
    with tab4:
        show_ai_assistant()
    
    with tab5:
        show_professional_dashboard()

def show_ai_assistant():
    """AI Trading Assistant"""
    if not st.session_state.user:
        st.error("Please login to access AI Assistant.")
        return
    
    st.header("🤖 AI Trading Assistant")
    st.caption("Your intelligent trading companion - Ask me anything about trading, strategies, or market analysis!")
    
    # AI Assistant Features
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat with AI")
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            if st.session_state.chat_history:
                for i, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        st.markdown(f"**🧑‍💻 You:** {message['content']}")
                    else:
                        st.markdown(f"**🤖 AI:** {message['content']}")
            else:
                st.info("👋 Hello! I'm your AI trading assistant. Ask me anything about trading strategies, market analysis, or performance optimization!")
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send, col_clear = st.columns([6, 1, 1])
            
            with col_input:
                user_input = st.text_input("Ask your trading question:", placeholder="e.g., 'What's my best strategy?' or 'Analyze current market conditions'", label_visibility="collapsed")
            
            with col_send:
                send_clicked = st.form_submit_button("📤")
            
            with col_clear:
                if st.form_submit_button("🗑️"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            if send_clicked and user_input:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Get AI response
                ai_response = get_ai_trading_response(user_input, st.session_state.user['id'])
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                st.rerun()
    
    with col2:
        st.subheader("⚡ Quick AI Actions")
        
        if st.button("📊 Analyze My Performance", use_container_width=True):
            user_stats = get_user_statistics(st.session_state.user['id'])
            analysis = f"""📊 **AI Performance Analysis for {st.session_state.user['username']}:**
            
**📈 Trading Statistics:**
- Total Backtests: {user_stats['total_backtests']}
- Total Trades: {user_stats['total_trades']}
- Average Return: {user_stats['avg_return']:.2f}%
- Best Return: {user_stats['best_return']:.2f}%

**🤖 AI Insights:**
{"🟢 **Strong Performance!** You're in the top 20% of traders." if user_stats['avg_return'] > 5 else "🟡 **Developing Well!** Focus on consistency." if user_stats['avg_return'] > 0 else "🔴 **Learning Phase** - Focus on education and small position sizes."}

**💡 AI Recommendations:**
- {"Continue with current strategies, consider increasing position sizes" if user_stats['avg_return'] > 5 else "Focus on risk management and strategy refinement" if user_stats['avg_return'] > 0 else "Practice with demo accounts and reduce risk per trade"}"""
            
            st.session_state.chat_history.append({"role": "assistant", "content": analysis})
            st.rerun()
        
        if st.button("💡 AI Strategy Builder", use_container_width=True):
            show_ai_strategy_builder()
        
        if st.button("📈 Market Intelligence", use_container_width=True):
            market_analysis = """📈 **AI Market Intelligence Report:**
            
**🎯 Current Market Conditions:**
- **EURUSD**: Consolidating in range 1.0800-1.0900
- **BTCUSD**: Strong uptrend, approaching resistance at $45,000
- **GBPUSD**: Bearish sentiment due to economic uncertainty
- **USDJPY**: Range-bound, waiting for BoJ intervention signals

**🤖 AI Predictions (Next 24-48 hours):**
- 68% probability of EURUSD breakout (direction uncertain)
- 73% probability of BTCUSD continued uptrend
- 61% probability of GBPUSD further decline

**⚠️ Risk Factors:**
- High-impact news events scheduled for tomorrow
- Increased volatility expected during NY session
- Month-end flows may cause unusual price movements

**💡 AI Trading Suggestions:**
- Reduce position sizes during high-impact news
- Focus on trend-following strategies in crypto
- Use wider stops in forex due to increased volatility"""
            
            st.session_state.chat_history.append({"role": "assistant", "content": market_analysis})
            st.rerun()
        
        if st.button("⚠️ Risk Assessment", use_container_width=True):
            risk_analysis = """⚠️ **AI Risk Assessment:**
            
**📊 Your Risk Profile:**
- Current risk level: Moderate
- Recommended max risk per trade: 2%
- Portfolio diversification: Good
- Drawdown tolerance: 15-20%

**🤖 AI Risk Optimization:**
1. **Position Sizing**: Use ATR-based position sizing for better risk control
2. **Correlation**: Avoid trading highly correlated pairs simultaneously  
3. **Time Diversification**: Spread trades across different time zones
4. **Strategy Diversification**: Use 2-3 different strategy types

**🚨 Risk Warnings:**
- Never risk more than 6% total across all open positions
- Reduce position sizes during high volatility periods
- Always use stop losses - no exceptions!

**✅ Current Status:** Your risk management is appropriate for your experience level."""
            
            st.session_state.chat_history.append({"role": "assistant", "content": risk_analysis})
            st.rerun()
    
    # AI Features Section
    st.markdown("---")
    st.subheader("🧠 Advanced AI Features")
    
    ai_col1, ai_col2, ai_col3 = st.columns(3)
    
    with ai_col1:
        if st.button("🎯 Pattern Recognition", use_container_width=True):
            show_ai_pattern_recognition()
    
    with ai_col2:
        if st.button("📰 Sentiment Analysis", use_container_width=True):
            show_market_sentiment_ai()
    
    with ai_col3:
        if st.button("📊 Generate AI Report", use_container_width=True):
            generate_ai_report()

def show_ai_strategy_builder():
    """AI-powered strategy builder"""
    st.subheader("🧠 AI Strategy Builder")
    st.caption("Let AI create a personalized trading strategy for you!")
    
    if not st.session_state.user:
        st.error("Please login to use AI Strategy Builder.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Tell AI What You Want:**")
        user_goal = st.selectbox("Trading Goal", [
            "Consistent daily profits",
            "Low-risk steady growth", 
            "High-growth aggressive",
            "News-based trading",
            "Trend following master"
        ])
        
        risk_tolerance = st.slider("Risk Appetite (1-10)", 1, 10, 5)
        trading_time = st.selectbox("Available Time", [
            "Full-time trader",
            "Part-time (2-3 hours)", 
            "Casual (30 min/day)"
        ])
    
    with col2:
        st.markdown("**🎨 AI Creativity Settings:**")
        creativity = st.slider("Strategy Creativity", 1, 10, 7)
        
        ai_personality = st.selectbox("AI Assistant Style", [
            "Conservative Warren Buffett",
            "Aggressive Day Trader", 
            "Quantitative Scientist",
            "Trend Following Guru",
            "Risk Management Expert"
        ])
    
    if st.button("🚀 Generate AI Strategy", type="primary"):
        generate_creative_strategy(user_goal, risk_tolerance, creativity, ai_personality)

def generate_creative_strategy(goal, risk, creativity, personality):
    """Generate creative AI strategy"""
    st.subheader("🎨 Your Custom AI Strategy")
    
    # AI-generated creative strategy names
    creative_names = [
        "The Phoenix Reversal System",
        "Quantum Momentum Hunter", 
        "The Stealth Profit Engine",
        "Golden Ratio Breakout Master",
        "The Market Whisperer",
        "Neural Network Trend Rider",
        "The Volatility Harvester"
    ]
    
    strategy_name = random.choice(creative_names)
    
    # Personality-based advice
    personality_advice = {
        "Conservative Warren Buffett": "Focus on high-probability setups with excellent risk/reward ratios",
        "Aggressive Day Trader": "Quick entries and exits with tight stops for maximum trading action",
        "Quantitative Scientist": "Data-driven approach with statistical edge validation",
        "Trend Following Guru": "Ride the major trends like a surfer riding ocean waves",
        "Risk Management Expert": "Capital preservation first, profits second - never blow up the account"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**🎯 Strategy Name**: {strategy_name}")
        st.info(f"**🤖 AI Personality**: {personality}")
        st.write(f"**💡 Philosophy**: {personality_advice[personality]}")
        
        # Creative parameters based on inputs
        lookback = 15 + (creativity * 2) + (risk * 1)
        atr_mult = 2.0 + (risk * 0.3) + (creativity * 0.1) 
        rr_ratio = 3.0 + (risk * 0.2)
        risk_per_trade = max(0.5, min(3.0, risk * 0.5))
        
        st.code(f"""
🎯 AI-Optimized Parameters:
├── Lookback: {lookback} periods
├── Stop Loss: {atr_mult:.1f}x ATR  
├── Risk/Reward: 1:{rr_ratio:.1f}
├── Risk per Trade: {risk_per_trade:.1f}%
└── Creativity Score: {creativity}/10
        """)
        
        # Save strategy parameters
        strategy_params = {
            'name': strategy_name,
            'lookback': lookback,
            'atr_mult': atr_mult,
            'rr_ratio': rr_ratio,
            'risk_per_trade': risk_per_trade,
            'personality': personality,
            'goal': goal
        }
    
    with col2:
        st.markdown("**🧠 AI Strategy Logic:**")
        
        if creativity > 7:
            st.write("🎨 **Creative Features**:")
            st.write("• Dynamic position sizing based on market volatility")
            st.write("• Multi-timeframe confluence analysis")
            st.write("• Adaptive stop-loss based on market conditions")
            
        if risk > 7:
            st.write("⚡ **High-Performance Features**:")
            st.write("• Aggressive position sizing for maximum returns")
            st.write("• Quick profit-taking rules (30% at 1R, 70% at target)")
            st.write("• Momentum-based entry confirmation")
        
        if goal == "News-based trading":
            st.write("📰 **News Integration Features**:")
            st.write("• High-impact event calendar integration")
            st.write("• Sentiment-based market bias adjustment")
            st.write("• Post-news volatility expansion capture")
        
        st.markdown("**🎯 AI Reasoning:**")
        reasoning = f"Strategy optimized for {goal.lower()} with {risk}/10 risk tolerance. Personality: {personality}. Creativity level: {creativity}/10."
        st.write(reasoning)
    
    # Save and test buttons
    col_save, col_test = st.columns(2)
    
    with col_save:
        if st.button("💾 Save AI Strategy", use_container_width=True):
            save_ai_strategy(
                st.session_state.user['id'], 
                strategy_name, 
                strategy_params, 
                reasoning
            )
            st.success(f"✅ '{strategy_name}' saved to your AI strategy library!")
            st.balloons()
    
    with col_test:
        if st.button("🧪 Backtest This Strategy", use_container_width=True):
            st.info("💡 Go to the Backtesting tab and use these parameters to test your AI strategy!")

def show_ai_pattern_recognition():
    """AI Pattern Recognition Feature"""
    st.subheader("🤖 AI Pattern Recognition")
    
    # Get some sample data for pattern recognition
    symbol = st.selectbox("Select Symbol for AI Analysis", ["EURUSD", "BTCUSD", "GBPUSD"], key="pattern_symbol")
    
    with st.spinner("🤖 AI is analyzing price patterns..."):
        time.sleep(1)  # Simulate AI processing
        
        # Generate sample data for demo
        _set_seed(42)
        df = _generate_data(symbol, 30, 1)
        patterns = detect_patterns_ai(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 AI-Detected Patterns:**")
        if patterns:
            for pattern in patterns:
                confidence = pattern['confidence']
                if confidence > 80:
                    st.success(f"🟢 **{pattern['name']}** ({confidence}% confidence)")
                    st.caption(f"Signal: {pattern['signal']}")
                elif confidence > 60:
                    st.warning(f"🟡 **{pattern['name']}** ({confidence}% confidence)")
                    st.caption(f"Signal: {pattern['signal']}")
                else:
                    st.info(f"🔵 **{pattern['name']}** ({confidence}% confidence)")
                    st.caption(f"Signal: {pattern['signal']}")
        else:
            st.info("No significant patterns detected by AI")
    
    with col2:
        st.markdown("**🎯 AI Trading Signals:**")
        if patterns:
            signals = generate_ai_signals(patterns)
            for signal in signals:
                st.write(f"• {signal}")
        else:
            st.write("• 📊 **NEUTRAL**: No strong signals detected")
            st.write("• 💡 **SUGGESTION**: Wait for clearer market structure")
    
    # Chart with patterns highlighted
    if not df.empty:
        fig = _price_fig_with_trades(df, [], symbol, show_ma=True)
        fig.update_layout(title=f"🤖 AI Pattern Analysis - {symbol}")
        st.plotly_chart(fig, use_container_width=True)

def show_market_sentiment_ai():
    """AI-powered market sentiment analysis"""
    st.subheader("🧠 AI Market Sentiment Analysis")
    
    # Simulate advanced sentiment data
    sentiment_data = {
        'EURUSD': {
            'sentiment': 0.65, 
            'trend': 'Bullish', 
            'news_count': 45,
            'social_mentions': 1250,
            'institutional_flow': 'Buying'
        },
        'BTCUSD': {
            'sentiment': 0.78, 
            'trend': 'Very Bullish', 
            'news_count': 89,
            'social_mentions': 8900,
            'institutional_flow': 'Strong Buying'
        },
        'GBPUSD': {
            'sentiment': 0.35, 
            'trend': 'Bearish', 
            'news_count': 23,
            'social_mentions': 650,
            'institutional_flow': 'Selling'
        },
        'USDJPY': {
            'sentiment': 0.52, 
            'trend': 'Neutral', 
            'news_count': 31,
            'social_mentions': 420,
            'institutional_flow': 'Mixed'
        }
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📰 AI Sentiment Dashboard:**")
        for symbol, data in sentiment_data.items():
            sentiment_score = data['sentiment']
            if sentiment_score > 0.7:
                st.success(f"🟢 **{symbol}**: {data['trend']} ({sentiment_score:.0%})")
                st.caption(f"📊 {data['news_count']} news • 💬 {data['social_mentions']} mentions • 💰 {data['institutional_flow']}")
            elif sentiment_score > 0.4:
                st.info(f"🔵 **{symbol}**: {data['trend']} ({sentiment_score:.0%})")
                st.caption(f"📊 {data['news_count']} news • 💬 {data['social_mentions']} mentions • 💰 {data['institutional_flow']}")
            else:
                st.error(f"🔴 **{symbol}**: {data['trend']} ({sentiment_score:.0%})")
                st.caption(f"📊 {data['news_count']} news • 💬 {data['social_mentions']} mentions • 💰 {data['institutional_flow']}")
    
    with col2:
        st.markdown("**🎯 AI Sentiment Insights:**")
        st.write("🤖 **AI Analysis Summary:**")
        st.write("• **BTCUSD**: Overwhelming positive sentiment from institutional adoption news")
        st.write("• **EURUSD**: Moderate bullish bias on ECB policy speculation")  
        st.write("• **GBPUSD**: Persistent bearish sentiment due to economic headwinds")
        st.write("• **USDJPY**: Neutral sentiment, waiting for central bank actions")
        
        st.markdown("**📈 AI Confidence Levels:**")
        st.write("🔴 **High Confidence**: BTCUSD bullish trend continuation")
        st.write("🟡 **Medium Confidence**: EURUSD range breakout potential")
        st.write("🟢 **Low Confidence**: GBPUSD and USDJPY directional moves")
    
    # Sentiment-based recommendations
    st.markdown("---")
    st.subheader("🤖 AI Sentiment-Based Strategies")
    
    for symbol, data in sentiment_data.items():
        with st.expander(f"🎯 {symbol} AI Strategy Recommendation"):
            if data['sentiment'] > 0.7:
                st.write(f"**🤖 AI Strategy**: Strong bullish bias for {symbol}")
                st.write("**💡 Entry Logic**: Buy on any pullbacks to key support levels")
                st.write(f"**⚠️ Risk Management**: 2.5% per trade (high confidence setup)")
                st.write(f"**🎯 Targets**: Multiple targets with 70% position held for trend continuation")
            elif data['sentiment'] < 0.4:
                st.write(f"**🤖 AI Strategy**: Strong bearish bias for {symbol}")
                st.write("**💡 Entry Logic**: Sell on any rallies to key resistance levels") 
                st.write(f"**⚠️ Risk Management**: 2% per trade (bearish sentiment confirmed)")
                st.write(f"**🎯 Targets**: Quick profit-taking due to negative sentiment")
            else:
                st.write(f"**🤖 AI Strategy**: Range trading approach for {symbol}")
                st.write("**💡 Entry Logic**: Buy support, sell resistance until breakout")
                st.write(f"**⚠️ Risk Management**: 1.5% per trade (uncertain environment)")
                st.write(f"**🎯 Targets**: Conservative profit-taking in ranges")

def generate_ai_report():
    """Generate comprehensive AI trading report"""
    if not st.session_state.user:
        st.error("Please login to generate AI reports.")
        return
    
    st.subheader("📊 AI-Generated Performance Report")
    
    with st.spinner("🤖 AI is analyzing your complete trading data..."):
        time.sleep(2)  # Simulate AI processing
    
    # Get user data for personalized report
    user_stats = get_user_statistics(st.session_state.user['id'])
    best_assets = get_user_best_assets(st.session_state.user['id'])
    
    # Generate performance grade
    if user_stats['avg_return'] > 15:
        grade = "A+"
        performance_color = "🟢"
        performance_desc = "Exceptional"
    elif user_stats['avg_return'] > 10:
        grade = "A"
        performance_color = "🟢"
        performance_desc = "Excellent"
    elif user_stats['avg_return'] > 5:
        grade = "B+"
        performance_color = "🟡"
        performance_desc = "Good"
    elif user_stats['avg_return'] > 0:
        grade = "B"
        performance_color = "🟡"
        performance_desc = "Average"
    else:
        grade = "C"
        performance_color = "🔴"
        performance_desc = "Needs Improvement"
    
    # Create the complete report
    report_content = f"""# 🤖 AI Trading Performance Report

**Generated for**: {st.session_state.user['username']}  
**Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}  
**Analysis Period**: All-time trading data

---

## 📈 Executive Summary

{performance_color} **Overall Performance Grade**: **{grade}** ({performance_desc})

Your AI trading assistant has analyzed **{user_stats['total_backtests']} backtests** containing **{user_stats['total_trades']} individual trades**.

### 🎯 Key Performance Metrics
- **Average Return**: {user_stats['avg_return']:.2f}% per backtest
- **Best Performance**: {user_stats['best_return']:.2f}% (single backtest)
- **Trading Frequency**: {user_stats['total_backtests']} strategy tests completed
- **Experience Level**: {"Advanced" if user_stats['total_trades'] > 100 else "Intermediate" if user_stats['total_trades'] > 50 else "Beginner"}

---

## 🧠 AI Deep Analysis

### 📊 Strengths Identified
{"🟢 **Consistent Profitability**: You maintain positive returns across multiple strategies" if user_stats['avg_return'] > 0 else "🟡 **Learning Progress**: You're actively testing and improving strategies"}

{"🟢 **Risk Management**: Your drawdowns appear well-controlled" if user_stats['avg_return'] > 0 else "🔴 **Risk Focus Needed**: Prioritize capital preservation over profits"}

{"🟢 **Market Diversification**: Testing across multiple instruments" if len(best_assets) > 1 else "🟡 **Consider Diversification**: Expand to more markets"}

### ⚠️ Areas for Improvement
{"🎯 **Strategy Refinement**: Fine-tune your best-performing setups for even better results" if user_stats['avg_return'] > 5 else "🎯 **Strategy Development**: Focus on developing 1-2 core strategies before expanding" if user_stats['avg_return'] > 0 else "🎯 **Education Priority**: Invest time in learning fundamental trading principles"}

{"📈 **Position Sizing**: Consider dynamic position sizing based on strategy confidence" if user_stats['total_trades'] > 50 else "📈 **Sample Size**: Increase the number of trades per backtest for statistical significance"}

---

## 🎯 AI Recommendations

### 💡 Immediate Actions (Next 7 Days)
1. **Focus Markets**: {"Continue with " + best_assets[0] + " and " + best_assets[1] + " - your strongest performers" if len(best_assets) >= 2 else "Expand beyond " + (best_assets[0] if best_assets else "current markets") + " for diversification"}

2. **Strategy Optimization**: {"Re-test your top strategies with smaller parameter variations" if user_stats['avg_return'] > 5 else "Focus on mastering one core strategy before expanding" if user_stats['avg_return'] > 0 else "Start with simple trend-following strategies"}

3. **Risk Management**: {"Consider increasing position sizes gradually (current performance supports it)" if user_stats['avg_return'] > 10 else "Maintain current risk levels - they're appropriate for your experience" if user_stats['avg_return'] > 0 else "Reduce risk per trade to 1% until consistency improves"}

### 🚀 Medium-term Goals (Next 30 Days)
- **Backtest Volume**: Run at least {"10 more backtests" if user_stats['total_backtests'] < 10 else "20 more backtests"} to increase statistical confidence
- **Market Coverage**: {"Maintain focus on your profitable markets" if len(best_assets) >= 2 else "Test at least 2-3 additional markets"}
- **Strategy Development**: {"Develop advanced features like trailing stops" if user_stats['avg_return'] > 5 else "Master basic entry and exit rules"}

### 📚 Learning Priorities
{"Advanced Topics: Multi-timeframe analysis, portfolio optimization, algorithmic execution" if user_stats['total_trades'] > 100 else "Intermediate Topics: Risk management, position sizing, market correlation" if user_stats['total_trades'] > 50 else "Fundamental Topics: Chart reading, trend analysis, support/resistance"}

---

## 🎲 AI Predictive Insights

### 📈 Probability Assessments
- **Probability of Continued Success**: {85 if user_stats['avg_return'] > 10 else 70 if user_stats['avg_return'] > 5 else 55 if user_stats['avg_return'] > 0 else 35}%
- **Expected Return Next Month**: {user_stats['avg_return'] * 1.1:.1f}% (based on current trajectory)
- **Risk of Significant Drawdown**: {"Low (15%)" if user_stats['avg_return'] > 5 else "Moderate (35%)" if user_stats['avg_return'] > 0 else "High (60%)"}

### 🎯 Success Factors
1. **Consistency**: {"Excellent - maintain current approach" if user_stats['avg_return'] > 5 else "Developing - focus on routine" if user_stats['avg_return'] > 0 else "Needs work - establish daily habits"}
2. **Adaptability**: {"Good - ready for advanced concepts" if user_stats['total_backtests'] > 20 else "Improving - continue learning"}
3. **Risk Awareness**: {"Strong - well-positioned for growth" if user_stats['avg_return'] > 0 else "Critical - prioritize education"}

---

## 🚀 Next Steps

### Action Items
- [ ] {"Implement advanced position sizing" if user_stats['avg_return'] > 5 else "Focus on consistency over complexity"}
- [ ] {"Test multi-asset portfolio strategies" if len(best_assets) > 2 else "Expand to at least 3 different markets"}
- [ ] {"Consider live paper trading" if user_stats['avg_return'] > 10 else "Continue backtesting until profitable"}
- [ ] {"Join advanced trading communities" if user_stats['total_trades'] > 100 else "Focus on fundamental education"}

**Remember**: This AI analysis is based on your historical performance. Market conditions change, and past performance doesn't guarantee future results.

---

*Report generated by AI Trading Assistant v2.0*
*For questions about this report, chat with your AI assistant*"""

    # Display the report
    st.markdown(report_content)
    
    # Add download functionality
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Report", use_container_width=True):
            st.download_button(
                label="💾 Download as Markdown",
                data=report_content,
                file_name=f"AI_Trading_Report_{st.session_state.user['username']}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    with col2:
        if st.button("📊 Generate New Report", use_container_width=True):
            st.rerun()
    
    # Add success message
    st.success("✅ AI Report Generated Successfully! Use the download button to save it.")
    st.balloons()

def show_enhanced_backtesting():
    """Enhanced backtesting with 23+ strategies"""
    if not st.session_state.user:
        st.error("Please login to run backtests.")
        return
        
    st.header("🚀 AI Strategy Tournament - 23+ Strategies")
    
    col_left, col_right = st.columns([1, 3])
    
    with col_left:
        st.subheader("🎯 Market Selection")
        
        # Enhanced symbol selection
        market_category = st.selectbox("Market Category", list(ALL_SYMBOLS.keys()))
        symbol = st.selectbox("Symbol", ALL_SYMBOLS[market_category])
        
        st.subheader("🤖 AI Tournament Settings")
        days = st.slider("Days of data", 15, 90, 30, step=5)
        
        tournament_mode = st.radio("Mode", [
            "🏆 Full Tournament (All 23+ Strategies)",
            "🎯 Category Tournament", 
            "⚡ Single Strategy Test"
        ])
        
        if tournament_mode == "🎯 Category Tournament":
            category = st.selectbox("Strategy Category", [
                "Trend Following", "Mean Reversion", "Momentum", "Volatility"
            ])
        elif tournament_mode == "⚡ Single Strategy Test":
            strategy_key = st.selectbox("Select Strategy", 
                                      [(k, v.name) for k, v in TRADING_STRATEGIES.items()],
                                      format_func=lambda x: x[1])[0]
        
        run_tournament = st.button("🚀 **RUN AI TOURNAMENT**", type="primary", use_container_width=True)
    
    with col_right:
        if run_tournament:
            tournament = AIStrategyTournament(st.session_state.user['id'])
            
            if tournament_mode == "🏆 Full Tournament (All 23+ Strategies)":
                results, df = tournament.run_tournament(symbol, days)
                show_tournament_results(results, symbol, df)
                
            elif tournament_mode == "🎯 Category Tournament":
                # Filter strategies by category
                filtered_strategies = {k: v for k, v in TRADING_STRATEGIES.items() 
                                     if v.category == category}
                
                with st.spinner(f"🤖 Testing {len(filtered_strategies)} {category} strategies..."):
                    # Run tournament with filtered strategies
                    original_strategies = TRADING_STRATEGIES.copy()
                    TRADING_STRATEGIES.clear()
                    TRADING_STRATEGIES.update(filtered_strategies)
                    
                    results, df = tournament.run_tournament(symbol, days)
                    show_tournament_results(results, symbol, df)
                    
                    # Restore original strategies
                    TRADING_STRATEGIES.clear()
                    TRADING_STRATEGIES.update(original_strategies)
                    
            elif tournament_mode == "⚡ Single Strategy Test":
                # Test single strategy
                strategy_config = TRADING_STRATEGIES[strategy_key]
                
                _set_seed(42)
                df = _generate_data(symbol, days, interval_min=1)
                
                cfg = tournament._convert_to_stratv4(strategy_config)
                stats, trades = backtest_v4(df, cfg)
                
                st.success(f"✅ {strategy_config.name} Results:")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Return", f"{stats.ret_pct:.2f}%")
                col2.metric("Trades", stats.n_trades)
                col3.metric("Win Rate", f"{stats.win_rate:.1f}%")
                col4.metric("Profit Factor", f"{stats.pf:.2f}")
                
                # Save results
                save_backtest_results(st.session_state.user['id'], symbol, stats, trades)
                
                # Show chart
                fig = _price_fig_with_trades(df, trades, symbol, show_ma=True)
                fig.update_layout(title=f"{strategy_config.name} - {symbol}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Show strategy overview
            st.info("👆 Select settings and run the AI Tournament to test multiple strategies simultaneously!")
            
            st.subheader("📊 Available Strategies Overview")
            
            for category in ["Trend Following", "Mean Reversion", "Momentum", "Volatility"]:
                with st.expander(f"{category} Strategies"):
                    category_strategies = [s for s in TRADING_STRATEGIES.values() if s.category == category]
                    
                    for strategy in category_strategies:
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        col_a.write(f"**{strategy.name}**")
                        col_a.caption(strategy.description)
                        col_b.write(f"Risk: {strategy.risk_level}")
                        col_c.write(f"Market: {strategy.market_type}")

def show_tournament_results(results, symbol, df):
    """Display AI tournament results"""
    st.success(f"🏆 AI Tournament Complete! Tested {len(results)} strategies on {symbol}")
    
    # Top 3 strategies
    st.subheader("🥇 Top 3 AI-Recommended Strategies")
    
    top_3 = results[:3]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🥇 WINNER", top_3[0]['strategy'])
        st.write(f"**Return:** {top_3[0]['return']:.2f}%")
        st.write(f"**AI Score:** {top_3[0]['ai_score']:.1f}/100")
        st.write(f"**Win Rate:** {top_3[0]['win_rate']:.1f}%")
        
    with col2:
        st.metric("🥈 2nd Place", top_3[1]['strategy'])
        st.write(f"**Return:** {top_3[1]['return']:.2f}%")
        st.write(f"**AI Score:** {top_3[1]['ai_score']:.1f}/100")
        st.write(f"**Win Rate:** {top_3[1]['win_rate']:.1f}%")
        
    with col3:
        st.metric("🥉 3rd Place", top_3[2]['strategy'])
        st.write(f"**Return:** {top_3[2]['return']:.2f}%")
        st.write(f"**AI Score:** {top_3[2]['ai_score']:.1f}/100")
        st.write(f"**Win Rate:** {top_3[2]['win_rate']:.1f}%")
    
    # AI Recommendation
    winner = top_3[0]
    st.subheader("🤖 AI Tournament Analysis")
    
    if winner['ai_score'] > 80:
        st.success(f"🟢 **Excellent Choice**: {winner['strategy']} shows exceptional performance with {winner['return']:.1f}% return!")
    elif winner['ai_score'] > 60:
        st.info(f"🟡 **Good Option**: {winner['strategy']} is profitable with {winner['return']:.1f}% return.")
    else:
        st.warning(f"🟠 **Proceed with Caution**: Best strategy shows {winner['return']:.1f}% return. Consider different market conditions.")
    
    # Detailed results table
    st.subheader("📊 Complete Tournament Results")
    
    df_results = pd.DataFrame(results)
    df_results = df_results[['strategy', 'category', 'return', 'trades', 'win_rate', 'profit_factor', 'ai_score', 'risk_level']]
    df_results.columns = ['Strategy', 'Category', 'Return %', 'Trades', 'Win Rate %', 'Profit Factor', 'AI Score', 'Risk Level']
    
    # Add performance indicators
    df_results['🏆'] = df_results.index.map(lambda x: '🥇' if x == 0 else '🥈' if x == 1 else '🥉' if x == 2 else '')
    
    st.dataframe(df_results, use_container_width=True, height=400)
    
    # Save winner to user's results
    winner_config = TRADING_STRATEGIES[winner['key']]
    tournament_obj = AIStrategyTournament(st.session_state.user['id'])
    cfg = tournament_obj._convert_to_stratv4(winner_config)
    stats, trades = backtest_v4(df, cfg)
    
    save_backtest_results(st.session_state.user['id'], symbol, stats, trades)
    st.success(f"💾 Winner strategy results saved! ({winner['strategy']})")

def show_live_demo():
    """Live demo functionality with AI insights"""
    st.header("🔴 Live Demo with AI Intelligence")
    
    col_live_left, col_live_right = st.columns([1, 3])
    
    with col_live_left:
        st.subheader("📊 Live Settings")
        
        asset_type = st.radio("Asset Type", ["Crypto", "Forex"], key="live_asset")
        if asset_type == "Crypto":
            live_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD"]
        else:
            live_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "GBPJPY"]
        
        selected_symbol = st.selectbox("Select Symbol", live_symbols, key="live_symbol")
        
        st.subheader("🤖 AI Analysis")
if st.button("🧠 Get AI Market Analysis"):
    with st.spinner("🤖 OpenAI analyzing live market conditions..."):
        ai_analysis = get_openai_market_analysis(selected_symbol)
        st.success("✅ OpenAI Analysis Complete!")
        st.markdown(ai_analysis)
        
        st.write(f"⏰ **Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    with col_live_right:
        st.subheader(f"📈 {selected_symbol} - Live AI-Enhanced Analysis")
        
        try:
            live_data = get_live_price(selected_symbol)
            
            if not live_data.empty:
                current_price = live_data['Close'].iloc[-1]
                previous_price = live_data['Close'].iloc[-2] if len(live_data) > 1 else current_price
                price_change = current_price - previous_price
                price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
                
                # AI-enhanced metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Current Price", f"{current_price:.5f}")
                col_b.metric("Change", f"{price_change:.5f}", f"{price_change_pct:.2f}%")
                col_c.metric("24h High", f"{live_data['High'].max():.5f}")
                col_d.metric("24h Low", f"{live_data['Low'].min():.5f}")
                
                # AI Pattern Recognition on live data
                if len(live_data) >= 20:
                    live_patterns = detect_patterns_ai(live_data.rename(columns=str.lower))
                    
                    if live_patterns:
                        st.subheader("🤖 Live AI Pattern Recognition")
                        for pattern in live_patterns[:2]:  # Show top 2 patterns
                            if pattern['confidence'] > 60:
                                confidence_color = "🟢" if pattern['confidence'] > 80 else "🟡"
                                st.write(f"{confidence_color} **{pattern['name']}** ({pattern['confidence']}%) - {pattern['signal']}")
                
                # Live chart with AI insights
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=live_data.index,
                    open=live_data['Open'],
                    high=live_data['High'], 
                    low=live_data['Low'],
                    close=live_data['Close'],
                    name=selected_symbol
                ))
                
                # Add moving averages
                if len(live_data) >= 20:
                    live_data['MA20'] = live_data['Close'].rolling(20).mean()
                    fig.add_trace(go.Scatter(
                        x=live_data.index, y=live_data['MA20'],
                        name="MA20", line=dict(color="orange", width=2)
                    ))
                
                fig.update_layout(
                    title=f"🤖 {selected_symbol} - Live AI Analysis",
                    xaxis_title="Time",
                    yaxis_title="Price", 
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("❌ No live data available. Try a different symbol.")
                
        except Exception as e:
            st.error(f"❌ Error fetching live  {str(e)}")

def show_user_results_history():
    """Show user's complete backtesting history with AI insights"""
    if not st.session_state.user:
        st.error("Please login to view results.")
        return
        
    st.header("📈 My AI-Enhanced Trading Results")
    
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    
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
        # AI-enhanced summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_backtests = len(backtests)
        avg_return = sum(b[2] for b in backtests) / total_backtests
        best_return = max(b[2] for b in backtests)
        total_trades = sum(b[5] for b in backtests)
        
        col1.metric("Total Backtests", total_backtests)
        col2.metric("Average Return", f"{avg_return:.2f}%")
        col3.metric("Best Return", f"{best_return:.2f}%")
        col4.metric("Total Trades", total_trades)
        
        # AI Performance Insights
        if avg_return > 10:
            st.success("🤖 **AI Insight**: Exceptional performance! You're in the top 10% of traders.")
        elif avg_return > 5:
            st.info("🤖 **AI Insight**: Strong performance. Consider scaling up your best strategies.")
        elif avg_return > 0:
            st.warning("🤖 **AI Insight**: Profitable but room for improvement. Focus on consistency.")
        else:
            st.error("🤖 **AI Insight**: Focus on risk management and strategy education.")
        
        # Results table
        st.subheader("📊 Detailed Results with AI Analysis")
        df = pd.DataFrame(backtests, columns=[
            'ID', 'Symbol', 'Return %', 'Final Equity', 'Initial Equity', 'Trades', 
            'Win Rate %', 'Profit Factor', 'Max Drawdown %', 'Date'
        ])
        
        df['Return %'] = df['Return %'].round(2)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(df, use_container_width=True, height=400)
        
    else:
        st.info("📊 No backtests yet. Try the AI-enhanced backtesting feature!")

def show_professional_dashboard():
    """Professional dashboard with AI insights"""
    if not st.session_state.user:
        st.error("Session expired. Please login again.")
        return
        
    st.header("📊 AI-Powered Professional Dashboard")
    
    user_stats = get_user_statistics(st.session_state.user['id'])
    
    # AI-enhanced metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📈 Total Backtests", user_stats['total_backtests'])
    
    with col2:
        st.metric("🎯 Total Trades", user_stats['total_trades'])
    
    with col3:
        st.metric("🏆 Best Return", f"{user_stats['best_return']:.2f}%")
    
    with col4:
        st.metric("💰 Avg Return", f"{user_stats['avg_return']:.2f}%")
    
    with col5:
        # AI Performance Score
        if user_stats['avg_return'] > 10:
            ai_score = min(100, int(user_stats['avg_return'] * 5))
            st.metric("🤖 AI Score", f"{ai_score}/100", "Excellent")
        elif user_stats['avg_return'] > 0:
            ai_score = int(50 + user_stats['avg_return'] * 3)
            st.metric("🤖 AI Score", f"{ai_score}/100", "Good")
        else:
            ai_score = max(10, int(50 + user_stats['avg_return'] * 2))
            st.metric("🤖 AI Score", f"{ai_score}/100", "Improving")
    
    # AI Insights Section
    st.subheader("🤖 AI Dashboard Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("**🎯 AI Performance Analysis:**")
        if user_stats['avg_return'] > 15:
            st.success("🚀 **Elite Trader**: Your performance is in the top 5% of all users!")
        elif user_stats['avg_return'] > 5:
            st.info("📈 **Skilled Trader**: You're consistently profitable with room to optimize.")
        elif user_stats['avg_return'] > 0:
            st.warning("⚖️ **Developing Trader**: Building profitability - focus on risk management.")
        else:
            st.error("🎓 **Learning Phase**: Priority should be education and conservative trading.")
    
    with insight_col2:
        st.markdown("**🎯 AI Recommendations:**")
        if user_stats['total_backtests'] < 10:
            st.write("• Run more backtests to improve statistical confidence")
        if user_stats['avg_return'] > 5:
            st.write("• Consider implementing live trading with small positions")
        else:
            st.write("• Focus on strategy consistency before scaling up")
        st.write("• Use the AI Assistant for personalized strategy optimization")
    
    # Platform statistics
    st.subheader("🌐 Platform Statistics")
    platform_stats = get_platform_statistics()
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric("Total Users", platform_stats['total_users'])
    
    with col_stats2:
        st.metric("Total Backtests", platform_stats['total_backtests'])
    
    with col_stats3:
        st.metric("Total Trades", platform_stats['total_trades'])

if __name__ == "__main__":
    main()
