#!/usr/bin/env python3
# fx_helper_cli.py — friendly launcher for complete_forex_system.py (built-ins only)

import os, sys, subprocess, time

APP_FILE = "complete_forex_system.py"
LOG_DIR = "runs"

def banner():
    print("🚀 HYBRID LSTM-TRANSFORMER FOREX TRADING SYSTEM")
    print("Optimized for Mac M2")
    print("="*60)

def note(msg):  print(f"\033[36m[NOTE]\033[0m {msg}")
def ok(msg):    print(f"\033[32m[OK]\033[0m {msg}")
def warn(msg):  print(f"\033[33m[WARN]\033[0m {msg}")
def err(msg):   print(f"\033[31m[ERR]\033[0m {msg}")

def cmd_supports_args():
    # If main script supports argparse, "--help" should print usage/flags.
    try:
        r = subprocess.run(["python3", APP_FILE, "--help"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = (r.stdout or "").lower()
        return ("usage:" in out) or ("--pairs" in out) or ("--steps" in out)
    except Exception:
        return False

def run_main(pass_args=False, pairs="EURUSD", days=30, indicators=23, steps="collect,indicators,train,backtest,live", emoji=True, log=True):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    logfile = os.path.join(LOG_DIR, f"run_{ts}.log")

    cmd = ["python3"]
    if emoji:
        cmd += ["-X", "utf8"]  # better emoji support in console
    cmd += [APP_FILE]

    if pass_args:
        cmd += ["--pairs", pairs,
                "--days", str(days),
                "--indicators", str(indicators),
                "--steps", steps]
        if not emoji:
            cmd += ["--no-emoji"]

    note("Command: " + " ".join(cmd))
    if log:
        note(f"Saving live output to: {logfile}")

    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
            f = open(logfile, "w", encoding="utf-8") if log else None
            try:
                for line in p.stdout:
                    print(line, end="")
                    if f: f.write(line)
            finally:
                if f: f.close()
        ok("Process finished.")
    except FileNotFoundError:
        err(f"Could not find '{APP_FILE}'. Make sure this file is in the SAME folder as fx_helper_cli.py.")
    except PermissionError:
        err("Permission error. Try running: chmod +x fx_helper_cli.py (or just run with python3).")
    except Exception as e:
        err(f"Unexpected error: {e}")

def menu():
    if not os.path.exists(APP_FILE):
        err(f"'{APP_FILE}' not found in current folder:\n  " + os.getcwd())
        sys.exit(1)

    banner()
    print("1) Quick Run (default pipeline)")
    print("2) Guided Run (attempt to pass options)")
    print("3) Overview-only banner (no heavy work)")
    print("4) Exit")

    choice = input("\nEnter choice [1-4]: ").strip() or "1"

    if choice == "1":
        # Just run your script and log output
        run_main(pass_args=False, emoji=True, log=True)

    elif choice == "2":
        # Try to pass options if your main script supports argparse
        pairs = input("Pairs (comma-separated) [EURUSD]: ").strip() or "EURUSD"
        days = input("Days of data [30]: ").strip() or "30"
        indicators = input("Indicator count [23]: ").strip() or "23"
        steps = input("Steps (collect,indicators,train,backtest,live) [collect,indicators,train,backtest,live]: ").strip() or "collect,indicators,train,backtest,live"
        use_emoji = input("Show emojis? [Y/n]: ").strip().lower() not in ("n","no")
        supports = cmd_supports_args()
        if not supports:
            warn("Your script does not expose CLI flags yet. Running default pipeline instead.")
        run_main(pass_args=supports, pairs=pairs, days=int(days), indicators=int(indicators), steps=steps, emoji=use_emoji, log=True)

    elif choice == "3":
        banner()
        print("\n🎯 SYSTEM OVERVIEW")
        print("="*60)
        items = [
            "1. Data Collection & Storage (SQLite with 7 major forex pairs)",
            "2. Technical Indicators (23+ indicators: SMA, EMA, RSI, MACD, etc.)",
            "3. Hybrid LSTM-Transformer Model Architecture",
            "4. Mac M2 Optimizations (MPS acceleration, mixed precision)",
            "5. Multi-task Learning (Classification + Regression + Confidence)",
            "6. Training Pipeline (Optimized for Apple Silicon)",
            "7. Comprehensive Backtesting System",
            "8. Live Trading Framework with Risk Management",
            "9. Complete Documentation & Setup Guides",
        ]
        for s in items:
            print("✅ " + s)
        print("\n(Overview only; no modules were executed.)")

    else:
        print("Goodbye!")

if __name__ == "__main__":
    menu()
