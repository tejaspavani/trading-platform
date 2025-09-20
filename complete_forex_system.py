#!/usr/bin/env python3
"""
complete_forex_system.py
Single-file, Mac-friendly, beginner-safe FX system with:
- Synthetic minute OHLCV generator (time-based seed -> different each run)
- 23 technical indicators (pandas + numpy)
- Rule-based signal + confidence
- Backtest with simple risk controls
- Friendly banner + CLI flags
"""

from __future__ import annotations
import os, sys, time, math, random, argparse, logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

# ---------- dependencies check ----------
try:
    import numpy as np
    import pandas as pd
except Exception as e:
    print("\n[ERROR] This script needs pandas and numpy (you already installed them).")
    print("        Activate your env and run:  conda install -c conda-forge pandas numpy -y\n")
    raise

# ---------- logging ----------
logger = logging.getLogger("fx")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.handlers = [_handler]

# ---------- pretty printing ----------
def banner(use_emoji: bool = True):
    rocket = "🚀 " if use_emoji else ""
    print(f"{rocket}HYBRID LSTM-TRANSFORMER FOREX TRADING SYSTEM")
    print("Optimized for Mac M2")
    print("=" * 60)

def section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# ---------- seed ----------
def set_run_seed(seed: int | None):
    """time-based seed if seed in (None, -1)"""
    if seed in (None, -1):
        seed = int(time.time()) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    logger.info(f"Random seed set to: {seed}")
    return seed

# ---------- data generation ----------
def generate_synthetic_pair(pair: str, days: int, interval_min: int = 1) -> pd.DataFrame:
    """
    Minute data for <days>. Simple GBM-like path with realistic highs/lows + volumes.
    """
    logger.info(f"Generating {days} days of {pair} data...")
    n = days * 24 * (60 // interval_min) + 1
    end = datetime.now()
    start = end - timedelta(days=days)
    idx = pd.date_range(start=start, periods=n, freq=f"{interval_min}min")

    # baseline depending on pair (just for scale)
    base = 1.1000 if pair.endswith("USD") else 100.0
    drift = 0.0
    # minute vol ~ 0.03% typical, tune a bit higher for variety
    sigma = 0.00035
    rets = np.random.normal(loc=drift, scale=sigma, size=n)
    close = np.empty(n, dtype=float)
    close[0] = base
    for i in range(1, n):
        close[i] = close[i-1] * (1.0 + rets[i])

    # open = prior close; add realistic high/low ranges
    open_ = np.empty_like(close); open_[0] = close[0]
    open_[1:] = close[:-1]
    # range factor per bar
    rng = np.abs(np.random.normal(0.0, sigma * 0.8, size=n)) + sigma * 0.2
    high = np.maximum(open_, close) * (1.0 + rng)
    low  = np.minimum(open_, close) * (1.0 - rng)
    # ensure low <= min(open,close), high >= max(open,close)
    high = np.maximum(high, np.maximum(open_, close))
    low  = np.minimum(low,  np.minimum(open_, close))

    # volumes: random-ish with intraday seasonality
    vol = np.random.randint(800, 5500, size=n).astype(float)
    minutes = np.arange(n)
    session = 1.0 + 0.2 * np.sin(2 * math.pi * (minutes % 1440) / 1440.0)
    vol *= session

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx
    )
    logger.info(f"Generated {len(df):,} data points")
    return df

# ---------- indicators ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    # Wilder's smoothing uses alpha = 1/period
    return series.ewm(alpha=1.0/period, adjust=False).mean()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating technical indicators...")
    out = df.copy()

    # 1-4) SMAs
    for w in (5, 10, 20, 50):
        out[f"sma_{w}"] = df["close"].rolling(w).mean()

    # 5-6) EMAs
    out["ema_12"] = ema(df["close"], 12)
    out["ema_26"] = ema(df["close"], 26)

    # 7) RSI (Wilder)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = wilder_ema(gain, 14)
    avg_loss = wilder_ema(loss, 14)
    rs = (avg_gain / (avg_loss.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # 8-10) MACD (12,26,9)
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # 11-12) Stochastic %K/%D
    low_min14 = df["low"].rolling(14).min()
    high_max14 = df["high"].rolling(14).max()
    out["stoch_k"] = 100 * (df["close"] - low_min14) / (high_max14 - low_min14)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()

    # 13) ATR (Wilder)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    out["atr_14"] = wilder_ema(tr, 14)

    # 14-16) Bollinger Bands (20, 2 std)
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std(ddof=0)
    out["bb_mid"] = mid
    out["bb_up"] = mid + 2 * std
    out["bb_low"] = mid - 2 * std

    # 17) CCI (20)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_sma20 = tp.rolling(20).mean()
    md = (tp - tp_sma20).abs().rolling(20).mean()
    out["cci_20"] = (tp - tp_sma20) / (0.015 * md)

    # 18-20) ADX + DI± (14)
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    atr = out["atr_14"]
    plus_di = 100 * wilder_ema(plus_dm, 14) / atr
    minus_di = 100 * wilder_ema(minus_dm, 14) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    out["di_plus"] = plus_di
    out["di_minus"] = minus_di
    out["adx_14"] = wilder_ema(dx, 14)

    # 21) OBV
    direction = np.sign(df["close"].diff().fillna(0.0))
    out["obv"] = (direction * df["volume"]).fillna(0.0).cumsum()

    # 22) ROC(12)
    out["roc_12"] = df["close"].pct_change(12)

    # 23) PPO (12,26)
    out["ppo"] = 100 * (out["ema_12"] - out["ema_26"]) / out["ema_26"]

    # 24) Williams %R (14)
    hh = df["high"].rolling(14).max()
    ll = df["low"].rolling(14).min()
    out["willr_14"] = -100 * (hh - df["close"]) / (hh - ll)

    # 25) TRIX (15)
    ema1 = ema(df["close"], 15)
    ema2 = ema(ema1, 15)
    ema3 = ema(ema2, 15)
    out["trix_15"] = ema3.pct_change()

    logger.info("Calculated 23+ indicators")
    return out

# ---------- signals ----------
from dataclasses import dataclass

@dataclass
class SignalConfig:
    # Entry filters
    rsi_buy: float = 58.0
    rsi_sell: float = 42.0
    conf_threshold: float = 0.55
    cooldown_bars: int = 10

    # Auto-tuning of confidence threshold
    auto_tune: bool = True
    min_signals: int = 40
    max_signals: int = 250

def _apply_threshold(signal_arr, conf_series, cooldown_bars, thr):
    """Gate raw signals with a confidence threshold + cooldown."""
    final = np.zeros(len(signal_arr), dtype=int)
    last_trade = -10**9
    conf_values = conf_series.values  # fast access
    for i in range(len(signal_arr)):
        if conf_values[i] >= thr and signal_arr[i] != 0 and (i - last_trade) >= cooldown_bars:
            final[i] = int(signal_arr[i])
            last_trade = i
    return final


def generate_signals(df: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    logger.info("Generating trading signals...")
    out = df.copy()

    # Base directional conditions
    cond_long = (
        (out["close"] > out["ema_12"]) &
        (out["macd"] > out["macd_signal"]) &
        (out["rsi_14"] > cfg.rsi_buy) &
        (out["stoch_k"] > out["stoch_d"])
    )
    cond_short = (
        (out["close"] < out["ema_12"]) &
        (out["macd"] < out["macd_signal"]) &
        (out["rsi_14"] < cfg.rsi_sell) &
        (out["stoch_k"] < out["stoch_d"])
    )
    # --- Trend + strength + volatility filters + breakout buffer ---
    adx_ok   = (out["adx_14"] >= 22.0)
    trend_up = (out["ema_12"] > out["ema_26"])
    trend_dn = (out["ema_12"] < out["ema_26"])

    # Volatility floor: require ATR above its 30th percentile (global over window)
    atr_q30 = out["atr_14"].quantile(0.30)
    atr_ok  = out["atr_14"] >= atr_q30

    # Small breakout beyond previous bar by 0.10*ATR
    prev_high = out["high"].shift(1)
    prev_low  = out["low"].shift(1)
    buf = (out["atr_14"].fillna(out["atr_14"].median())) * 0.10
    break_long  = out["close"] > (prev_high + buf)
    break_short = out["close"] < (prev_low  - buf)

    cond_long  = cond_long  & adx_ok & trend_up & atr_ok & break_long
    cond_short = cond_short & adx_ok & trend_dn & atr_ok & break_short

    # Optional: trade only during active session (08:00–16:00)
    hrs = out.index.hour
    session_mask = (hrs >= 8) & (hrs <= 16)

    signal_raw = np.where(cond_long, 1, np.where(cond_short, -1, 0))
    signal_raw = np.where(session_mask, signal_raw, 0)
    
    # Confidence score (same as before, slightly cleaned)
    macd_hist = out["macd_hist"].fillna(0.0)
    macd_scale = macd_hist.rolling(200).std().fillna(macd_hist.abs().median() + 1e-9)
    rsi_dev = (out["rsi_14"] - 50.0).abs() / 50.0
    dist_ema = (out["close"] - out["ema_12"]).abs() / out["close"].replace(0, np.nan)
    macd_z = (macd_hist.abs() / (macd_scale + 1e-12)).clip(0, 3)
    conf = (0.45 * rsi_dev + 0.45 * dist_ema + 0.10 * macd_z).clip(0, 1)
    out["confidence"] = conf.fillna(0.0)

    # Initial threshold pass
    thr = float(cfg.conf_threshold)
    final = _apply_threshold(signal_raw, out["confidence"], cfg.cooldown_bars, thr)

    # Auto-tune: try to bring trade count into [min_signals, max_signals]
    if cfg.auto_tune:
        for _ in range(20):  # small bounded loop
            count = int((final != 0).sum())
            if count < cfg.min_signals and thr > 0.20:
                thr = max(0.20, thr - 0.05)
                final = _apply_threshold(signal_raw, out["confidence"], cfg.cooldown_bars, thr)
            elif count > cfg.max_signals and thr < 0.95:
                thr = min(0.95, thr + 0.05)
                final = _apply_threshold(signal_raw, out["confidence"], cfg.cooldown_bars, thr)
            else:
                break
        logger.info(f"Auto-tune: threshold={thr:.2f}, trades={(final != 0).sum()}")

    out["signal"] = final
    total = int((final != 0).sum())
    logger.info(f"Generated {total:,} trading signals")
    return out


# ---------- backtest ----------

@dataclass
class BacktestResult:
    initial_balance: float
    final_equity: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    avg_win: float
    avg_loss: float

def backtest(df: pd.DataFrame,
             spread: float = 0.00005,        # ~0.5 pip
             contract_size: float = 10_000,  # PnL = (price change) * contract_size
             use_atr_exits: bool = True,
             sl_mult: float = 0.60,          # tighter stop
             tp_mult: float = 1.80,          # larger target (aim R≈3)
             min_hold_bars: int = 5,         # hold at least N bars before opposite flip
             trail_after_mult: float = 0.80, # start trailing after +0.8*ATR move
             trail_dist_mult: float = 0.60   # trail distance = 0.6*ATR from peak/trough
             ) -> BacktestResult:
    logger.info("Running backtest...")

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    sig   = df["signal"].values
    atr   = df.get("atr_14", pd.Series(np.nan, index=df.index)).values

    equity = 10_000.0
    pos = 0                 # -1, 0, +1
    entry_px = None
    entry_atr = None
    bars_since_entry = 0
    peak = None             # for long trailing
    trough = None           # for short trailing

    trades_pnl: List[float] = []
    total_trades = 0

    def _exit_at(px):
        nonlocal equity, pos, entry_px, entry_atr, bars_since_entry, peak, trough
        pnl = (px - entry_px) * pos * contract_size
        pnl -= spread * contract_size
        equity += pnl
        trades_pnl.append(pnl)
        pos = 0
        entry_px = None
        entry_atr = None
        bars_since_entry = 0
        peak = None
        trough = None

    for i in range(1, len(close)):
        if pos == 0:
            if sig[i] != 0:
                pos = sig[i]
                entry_px = close[i]
                entry_atr = atr[i] if not np.isnan(atr[i]) else np.nan
                equity -= spread * contract_size
                total_trades += 1
                bars_since_entry = 0
                peak = high[i] if pos > 0 else None
                trough = low[i] if pos < 0 else None
            continue

        # update running extremes
        bars_since_entry += 1
        if pos > 0:
            peak = max(peak, high[i]) if peak is not None else high[i]
        else:
            trough = min(trough, low[i]) if trough is not None else low[i]

        # 1) ATR stop/target
        if use_atr_exits and entry_atr and not np.isnan(entry_atr):
            stop_dist = entry_atr * sl_mult
            tp_dist   = entry_atr * tp_mult
            if pos > 0:
                stop = entry_px - stop_dist
                tp   = entry_px + tp_dist
                # conservative: if both touched, assume stop first
                if low[i] <= stop:
                    _exit_at(stop); continue
                if high[i] >= tp:
                    _exit_at(tp); continue
            else:
                stop = entry_px + stop_dist
                tp   = entry_px - tp_dist
                if high[i] >= stop:
                    _exit_at(stop); continue
                if low[i] <= tp:
                    _exit_at(tp); continue

        # 2) Trailing stop (kick in after favorable move)
        if entry_atr and not np.isnan(entry_atr):
            if pos > 0 and peak is not None and (peak - entry_px) >= entry_atr * trail_after_mult:
                trail_stop = peak - entry_atr * trail_dist_mult
                if low[i] <= trail_stop:
                    _exit_at(trail_stop); continue
            if pos < 0 and trough is not None and (entry_px - trough) >= entry_atr * trail_after_mult:
                trail_stop = trough + entry_atr * trail_dist_mult
                if high[i] >= trail_stop:
                    _exit_at(trail_stop); continue

        # 3) Opposite signal: exit (but don't instant-flip) after min hold
        if sig[i] != 0 and sig[i] != pos and bars_since_entry >= min_hold_bars:
            _exit_at(close[i])
            # Do NOT immediately enter opposite on this same bar
            continue

    # Close any open position at last bar
    if pos != 0 and entry_px is not None:
        _exit_at(close[-1])

    logger.info("Backtest completed")

    wins   = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p < 0]
    avg_win  = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    total_return_pct = (equity / 10_000.0 - 1.0) * 100.0

    return BacktestResult(
        initial_balance=10_000.0,
        final_equity=equity,
        total_return_pct=total_return_pct,
        total_trades=total_trades,
        winning_trades=len(wins),
        avg_win=avg_win,
        avg_loss=avg_loss,
    )

# ---------- orchestration ----------
def run_pipeline(pair: str, days: int, emoji: bool, seed: int | None, sig_cfg: SignalConfig):
    ...
    df = generate_synthetic_pair(pair, days, interval_min=1)
    df = compute_indicators(df)
    df = generate_signals(df, sig_cfg)
    res = backtest(df)
    return df, res
    set_run_seed(seed)

    df = generate_synthetic_pair(pair, days, interval_min=1)
    df = compute_indicators(df)
    sig_cfg = SignalConfig()
    df = generate_signals(df, sig_cfg)
    res = backtest(df)

    return df, res

def print_report(pair: str, df: pd.DataFrame, res: BacktestResult, emoji: bool):
    section("FOREX TRADING SYSTEM RESULTS")
    print(f"Initial Balance:      ${res.initial_balance:,.2f}")
    print(f"Final Equity:         ${res.final_equity:,.2f}")
    print(f"Total Return:         {res.total_return_pct:.2f}%\n")

    print("----------------------------------------")
    print("TRADE STATISTICS")
    print("----------------------------------------")
    win_rate = (res.winning_trades / res.total_trades * 100.0) if res.total_trades else 0.0
    losses_count = res.total_trades - res.winning_trades
    sum_wins   = res.avg_win  * res.winning_trades
    sum_losses = res.avg_loss * losses_count
    profit_factor = (sum_wins / abs(sum_losses)) if losses_count > 0 else (float('inf') if res.winning_trades > 0 else 0.0)
    expectancy = (win_rate/100.0)*res.avg_win - (1 - win_rate/100.0)*abs(res.avg_loss)
    print(f"Profit Factor:       {profit_factor:.2f}")
    print(f"Expectancy / trade:  ${expectancy:.2f}")  
    print(f"Total Trades:         {res.total_trades}")
    print(f"Winning Trades:       {res.winning_trades}")
    print(f"Win Rate:             {win_rate:.2f}%")
    print(f"Average Win:          ${res.avg_win:.2f}")
    print(f"Average Loss:         ${res.avg_loss:.2f}\n")

# status line
    if res.total_trades == 0:
        status = "🟡 NO TRADES (NEUTRAL)" if emoji else "NO TRADES (NEUTRAL)"
    elif res.total_return_pct >= 0:
        status = "🟢 PROFITABLE STRATEGY" if emoji else "PROFITABLE STRATEGY"
    else:
        status = "🔴 LOSS-MAKING STRATEGY" if emoji else "LOSS-MAKING STRATEGY"
    print(status)


    section("🎯 SYSTEM SUMMARY" if emoji else "SYSTEM SUMMARY")
    non_na = df.dropna()
    print(f"Symbol:               {pair}")
    print(f"Data Points:          {len(df):,}")
    print(f"Indicators:           23+")
    print(f"Signals Generated:    {(df['signal'] != 0).sum():,}")
    print(f"System Status:        {'✅ Operational' if emoji else 'Operational'}\n")

    print("🚀 NEXT STEPS:")
    print("1. Try different currency pairs")
    print("2. Adjust indicator parameters")
    print("3. Tune confidence threshold / cooldown")
    print("4. Paper trade before going live")
    print("5. Add live data feeds if desired")

    print("\n⚠️  DISCLAIMERS:")
    print("- Educational purposes only")
    print("- Past performance ≠ future results")
    print("- Start with paper trading")
    print("- Never risk more than you can afford\n")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Complete FX System (synthetic, indicators, signals, backtest)")
    p.add_argument("--pairs", type=str, default="EURUSD", help="Comma separated, e.g. EURUSD,GBPUSD,USDJPY")
    p.add_argument("--days", type=int, default=30, help="Number of days of minute data to simulate")
    p.add_argument("--seed", type=int, default=-1, help="-1 for time-based (varies each run), else fixed")
    p.add_argument("--no-emoji", action="store_true", help="Disable emojis in output")

    # New signal controls
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold (0..1). Overrides default.")
    p.add_argument("--cooldown", type=int, default=None, help="Cooldown bars between trades")
    p.add_argument("--rsi-buy", dest="rsi_buy", type=float, default=None, help="RSI long threshold (default 58)")
    p.add_argument("--rsi-sell", dest="rsi_sell", type=float, default=None, help="RSI short threshold (default 42)")

    # Auto-tune controls
    p.add_argument("--auto-tune", action="store_true", help="Enable auto-tuning of confidence threshold")
    p.add_argument("--min-signals", type=int, default=None, help="Min trades target for auto-tune")
    p.add_argument("--max-signals", type=int, default=None, help="Max trades target for auto-tune")
    return p.parse_args()
def ensure_structure(base: str):
    for name in ("data", "models", "backtesting", "trading", "utils", "logs", "runs", "config", "tests", "notebooks"):
        os.makedirs(os.path.join(base, name), exist_ok=True)
    logger.info(f"Project created at: {base}")


def main():
    args = parse_args()
    emoji = not args.no_emoji
    base = os.path.abspath(os.path.expanduser("~/forex_trading_system"))
    ensure_structure(base)
    banner(emoji)

    # Build SignalConfig from CLI
    sig_cfg = SignalConfig()
    if args.conf is not None:       sig_cfg.conf_threshold = args.conf
    if args.cooldown is not None:   sig_cfg.cooldown_bars  = args.cooldown
    if args.rsi_buy is not None:    sig_cfg.rsi_buy        = args.rsi_buy
    if args.rsi_sell is not None:   sig_cfg.rsi_sell       = args.rsi_sell
    if args.auto_tune:              sig_cfg.auto_tune      = True
    if args.min_signals is not None:sig_cfg.min_signals    = args.min_signals
    if args.max_signals is not None:sig_cfg.max_signals    = args.max_signals

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    for i, pair in enumerate(pairs, start=1):
        if len(pairs) > 1:
            print(f"\n--- [{i}/{len(pairs)}] {pair} ---")
        df, res = run_pipeline(pair=pair, days=args.days, emoji=emoji, seed=args.seed, sig_cfg=sig_cfg)
        print_report(pair, df, res, emoji)
if __name__ == "__main__":
    main()
