#!/usr/bin/env python3
"""
Trading Utilities Module
Provides utility functions for the forex trading platform.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import json
import logging

logger = logging.getLogger(__name__)

class TradingDatabase:
    """Handles database operations for the trading platform."""
    
    def __init__(self, db_path: str = "trading_platform.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        pair TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        status TEXT DEFAULT 'OPEN',
                        pnl REAL DEFAULT 0,
                        commission REAL DEFAULT 0,
                        notes TEXT
                    )
                ''')
                
                # Signals table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        pair TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        price REAL NOT NULL,
                        indicators TEXT,
                        reasons TEXT
                    )
                ''')
                
                # Portfolio history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        balance REAL NOT NULL,
                        equity REAL NOT NULL,
                        margin_used REAL DEFAULT 0,
                        free_margin REAL DEFAULT 0,
                        drawdown REAL DEFAULT 0
                    )
                ''')
                
                # Configuration table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS configuration (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def save_trade(self, trade_data: Dict) -> Optional[int]:
        """Save a trade to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (pair, action, quantity, entry_price, stop_loss, take_profit, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data['pair'],
                    trade_data['action'],
                    trade_data['quantity'],
                    trade_data['entry_price'],
                    trade_data.get('stop_loss', 0),
                    trade_data.get('take_profit', 0),
                    trade_data.get('status', 'OPEN')
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return None
    
    def update_trade(self, trade_id: int, updates: Dict) -> bool:
        """Update an existing trade."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build UPDATE query dynamically
                set_clauses = []
                values = []
                for key, value in updates.items():
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
                
                if not set_clauses:
                    return False
                
                query = f"UPDATE trades SET {', '.join(set_clauses)} WHERE id = ?"
                values.append(trade_id)
                
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating trade {trade_id}: {e}")
            return False
    
    def get_trades(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get trades from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if status:
                    cursor.execute('''
                        SELECT * FROM trades WHERE status = ? 
                        ORDER BY timestamp DESC LIMIT ?
                    ''', (status, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?
                    ''', (limit,))
                
                columns = [description[0] for description in cursor.description]
                trades = []
                
                for row in cursor.fetchall():
                    trade = dict(zip(columns, row))
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []
    
    def save_signal(self, signal_data: Dict) -> Optional[int]:
        """Save a trading signal to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO signals (pair, action, confidence, price, indicators, reasons)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['pair'],
                    signal_data['action'],
                    signal_data['confidence'],
                    signal_data['price'],
                    json.dumps(signal_data.get('indicators', {})),
                    json.dumps(signal_data.get('reasons', []))
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return None
    
    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Get recent trading signals."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))
                
                columns = [description[0] for description in cursor.description]
                signals = []
                
                for row in cursor.fetchall():
                    signal = dict(zip(columns, row))
                    # Parse JSON fields
                    if signal['indicators']:
                        signal['indicators'] = json.loads(signal['indicators'])
                    if signal['reasons']:
                        signal['reasons'] = json.loads(signal['reasons'])
                    signals.append(signal)
                
                return signals
                
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            return []


class PortfolioManager:
    """Manages portfolio operations and calculations."""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.db = TradingDatabase()
    
    def calculate_portfolio_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate portfolio performance metrics."""
        if not trades:
            return {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        # Calculate basic metrics
        closed_trades = [t for t in trades if t['status'] == 'CLOSED' and t['pnl'] is not None]
        
        if not closed_trades:
            return {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_trades': len(trades),
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        pnls = [float(t['pnl']) for t in closed_trades]
        total_pnl = sum(pnls)
        
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(pnls) * 100 if pnls else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Calculate drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        if len(pnls) > 1:
            daily_returns = np.array(pnls) / self.initial_balance
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
    
    def calculate_position_size(self, balance: float, risk_pct: float, stop_loss_pips: float, 
                              pair: str = "EUR/USD") -> float:
        """Calculate position size based on risk management rules."""
        try:
            # Risk amount in currency
            risk_amount = balance * (risk_pct / 100)
            
            # Pip value calculation (simplified)
            pip_values = {
                'EUR/USD': 10, 'GBP/USD': 10, 'AUD/USD': 10, 'NZD/USD': 10,
                'USD/JPY': 9.09, 'USD/CHF': 10.87, 'USD/CAD': 7.46
            }
            
            pip_value = pip_values.get(pair, 10)
            
            # Position size = Risk amount / (Stop loss in pips × Pip value)
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Round to reasonable lot size
            return round(position_size, 2)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Minimum position size


class RiskManager:
    """Handles risk management calculations and validations."""
    
    def __init__(self, max_risk_per_trade: float = 2.0, max_total_exposure: float = 10.0):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_exposure = max_total_exposure
    
    def validate_trade(self, trade_data: Dict, current_balance: float, 
                      open_positions: List[Dict]) -> Tuple[bool, str]:
        """Validate if a trade meets risk management criteria."""
        try:
            # Check position size vs balance
            position_value = trade_data['quantity'] * trade_data['entry_price'] * 10000  # Convert to base currency
            risk_pct = (position_value / current_balance) * 100
            
            if risk_pct > self.max_risk_per_trade:
                return False, f"Position size ({risk_pct:.2f}%) exceeds max risk per trade ({self.max_risk_per_trade}%)"
            
            # Check total exposure
            total_exposure = sum([pos['quantity'] * pos['entry_price'] * 10000 for pos in open_positions])
            new_total_exposure = (total_exposure + position_value) / current_balance * 100
            
            if new_total_exposure > self.max_total_exposure:
                return False, f"Total exposure ({new_total_exposure:.2f}%) would exceed maximum ({self.max_total_exposure}%)"
            
            # Check if stop loss is set
            if 'stop_loss' not in trade_data or not trade_data['stop_loss']:
                return False, "Stop loss must be set for all trades"
            
            # Validate stop loss distance
            entry_price = trade_data['entry_price']
            stop_loss = trade_data['stop_loss']
            
            if trade_data['action'] == 'BUY':
                if stop_loss >= entry_price:
                    return False, "Stop loss for BUY order must be below entry price"
            else:  # SELL
                if stop_loss <= entry_price:
                    return False, "Stop loss for SELL order must be above entry price"
            
            return True, "Trade validation passed"
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {str(e)}"
    
    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, 
                                   take_profit: float, action: str) -> float:
        """Calculate risk/reward ratio for a trade."""
        try:
            if action == 'BUY':
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:  # SELL
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            if risk <= 0:
                return 0
            
            return reward / risk
            
        except Exception as e:
            logger.error(f"Error calculating risk/reward ratio: {e}")
            return 0


class BacktestEngine:
    """Handles backtesting operations."""
    
    def __init__(self):
        self.portfolio_manager = PortfolioManager()
        self.risk_manager = RiskManager()
    
    def run_backtest(self, historical_data: pd.DataFrame, strategy_config: Dict, 
                    initial_balance: float = 10000.0) -> Dict:
        """Run a backtest on historical data."""
        try:
            results = {
                'trades': [],
                'daily_balance': [],
                'metrics': {},
                'equity_curve': []
            }
            
            balance = initial_balance
            open_positions = []
            
            # Simple moving average crossover strategy for demo
            short_window = strategy_config.get('short_ma', 12)
            long_window = strategy_config.get('long_ma', 26)
            
            # Calculate moving averages
            historical_data['ma_short'] = historical_data['close'].rolling(short_window).mean()
            historical_data['ma_long'] = historical_data['close'].rolling(long_window).mean()
            
            # Generate signals
            historical_data['signal'] = 0
            historical_data.loc[historical_data['ma_short'] > historical_data['ma_long'], 'signal'] = 1
            historical_data.loc[historical_data['ma_short'] < historical_data['ma_long'], 'signal'] = -1
            
            # Process each day
            for idx, row in historical_data.iterrows():
                if pd.isna(row['ma_short']) or pd.isna(row['ma_long']):
                    continue
                
                # Close existing positions if signal changes
                for position in open_positions[:]:
                    if (position['action'] == 'BUY' and row['signal'] == -1) or \
                       (position['action'] == 'SELL' and row['signal'] == 1):
                        
                        # Close position
                        exit_price = row['close']
                        if position['action'] == 'BUY':
                            pnl = (exit_price - position['entry_price']) * position['quantity'] * 10000
                        else:
                            pnl = (position['entry_price'] - exit_price) * position['quantity'] * 10000
                        
                        # Apply spread/commission
                        pnl -= abs(position['quantity']) * 2  # 2 pip spread
                        
                        balance += pnl
                        
                        trade_record = {
                            'entry_date': position['entry_date'],
                            'exit_date': row['datetime'] if 'datetime' in row else idx,
                            'pair': position['pair'],
                            'action': position['action'],
                            'quantity': position['quantity'],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'status': 'CLOSED'
                        }
                        
                        results['trades'].append(trade_record)
                        open_positions.remove(position)
                
                # Open new positions
                if row['signal'] != 0 and not open_positions:  # Only one position at a time
                    action = 'BUY' if row['signal'] == 1 else 'SELL'
                    quantity = 0.1  # Standard lot size
                    
                    position = {
                        'entry_date': row['datetime'] if 'datetime' in row else idx,
                        'pair': 'EUR/USD',  # Default pair
                        'action': action,
                        'quantity': quantity,
                        'entry_price': row['close']
                    }
                    
                    open_positions.append(position)
                
                # Record daily balance
                results['daily_balance'].append({
                    'date': row['datetime'] if 'datetime' in row else idx,
                    'balance': balance,
                    'equity': balance,
                    'positions': len(open_positions)
                })
            
            # Calculate metrics
            results['metrics'] = self.portfolio_manager.calculate_portfolio_metrics(results['trades'])
            results['metrics']['initial_balance'] = initial_balance
            results['metrics']['final_balance'] = balance
            results['metrics']['total_return'] = ((balance - initial_balance) / initial_balance) * 100
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'trades': [], 'daily_balance': [], 'metrics': {}, 'equity_curve': []}

# Global instances
trading_db = TradingDatabase()
portfolio_manager = PortfolioManager()
risk_manager = RiskManager()
backtest_engine = BacktestEngine()