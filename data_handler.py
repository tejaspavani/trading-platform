#!/usr/bin/env python3
"""
Data Handler Module for Forex Trading Platform
Handles real-time data fetching, processing, and integration with the trading system.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from config import API_KEYS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataHandler:
    """Handles data fetching and processing for forex pairs."""
    
    def __init__(self):
        self.alpha_vantage_key = API_KEYS.get('alpha_vantage', '')
        self.twelve_data_key = API_KEYS.get('twelve_data', '')
        self.cache = {}
        self.cache_timeout = 60  # seconds
    
    def get_forex_pairs(self) -> List[str]:
        """Get list of supported forex pairs."""
        return [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CHF', 'EUR/GBP',
            'NZD/USD', 'USD/CAD', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'CHF/JPY'
        ]
    
    def format_pair_for_api(self, pair: str, api_type: str = 'yahoo') -> str:
        """Format currency pair for specific API."""
        if api_type == 'yahoo':
            return pair.replace('/', '') + '=X'
        elif api_type == 'alpha_vantage':
            return pair.replace('/', '')
        return pair
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current price for a forex pair."""
        try:
            # Use Yahoo Finance for real-time data
            symbol = self.format_pair_for_api(pair, 'yahoo')
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            else:
                logger.warning(f"No data available for {pair}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching price for {pair}: {e}")
            # Return simulated data as fallback
            return self._simulate_price(pair)
    
    def get_historical_data(self, pair: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """Get historical data for a forex pair."""
        try:
            # Check cache first
            cache_key = f"{pair}_{period}_{interval}"
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_timeout:
                    return cached_data
            
            # Fetch from Yahoo Finance
            symbol = self.format_pair_for_api(pair, 'yahoo')
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                # Clean and format data
                data = data.reset_index()
                data.columns = [col.lower() for col in data.columns]
                
                # Cache the result
                self.cache[cache_key] = (datetime.now(), data)
                return data
            else:
                logger.warning(f"No historical data available for {pair}")
                return self._simulate_historical_data(pair, period)
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {pair}: {e}")
            return self._simulate_historical_data(pair, period)
    
    def get_multiple_prices(self, pairs: List[str]) -> Dict[str, float]:
        """Get current prices for multiple forex pairs."""
        prices = {}
        for pair in pairs:
            price = self.get_current_price(pair)
            if price:
                prices[pair] = price
        return prices
    
    async def get_prices_async(self, pairs: List[str]) -> Dict[str, float]:
        """Get prices asynchronously for better performance."""
        async def fetch_price(session, pair):
            try:
                # For demo purposes, we'll simulate async data fetching
                await asyncio.sleep(0.1)  # Simulate network delay
                return pair, self.get_current_price(pair)
            except Exception as e:
                logger.error(f"Error fetching price for {pair}: {e}")
                return pair, None
        
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_price(session, pair) for pair in pairs]
            results = await asyncio.gather(*tasks)
            
        return {pair: price for pair, price in results if price is not None}
    
    def get_market_summary(self) -> Dict:
        """Get market summary with key metrics."""
        major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
        prices = self.get_multiple_prices(major_pairs)
        
        summary = {
            'timestamp': datetime.now(),
            'pairs': {},
            'market_sentiment': 'NEUTRAL',
            'volatility': 'MODERATE'
        }
        
        for pair, price in prices.items():
            # Get 24h change (simulated for demo)
            change_pct = np.random.uniform(-2, 2)
            change_value = price * (change_pct / 100)
            
            summary['pairs'][pair] = {
                'price': price,
                'change': change_value,
                'change_pct': change_pct,
                'volume': np.random.randint(1000000, 10000000)
            }
        
        return summary
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators for the given data."""
        if data.empty or 'close' not in data.columns:
            return {}
        
        close = data['close']
        high = data['high'] if 'high' in data.columns else close
        low = data['low'] if 'low' in data.columns else close
        
        indicators = {}
        
        try:
            # Moving averages
            indicators['sma_20'] = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
            indicators['sma_50'] = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
            indicators['ema_12'] = close.ewm(span=12).mean().iloc[-1] if len(close) >= 12 else None
            indicators['ema_26'] = close.ewm(span=26).mean().iloc[-1] if len(close) >= 26 else None
            
            # RSI
            if len(close) >= 14:
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            if indicators['ema_12'] and indicators['ema_26']:
                macd_line = indicators['ema_12'] - indicators['ema_26']
                indicators['macd'] = macd_line
            
            # Bollinger Bands
            if len(close) >= 20:
                sma_20 = close.rolling(20).mean()
                std_20 = close.rolling(20).std()
                indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
                indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
                indicators['bb_middle'] = sma_20.iloc[-1]
            
            # Support and Resistance (simplified)
            if len(high) >= 10 and len(low) >= 10:
                recent_high = high.tail(10).max()
                recent_low = low.tail(10).min()
                indicators['resistance'] = recent_high
                indicators['support'] = recent_low
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    def generate_trading_signal(self, pair: str, data: pd.DataFrame = None) -> Dict:
        """Generate trading signal based on technical analysis."""
        if data is None:
            data = self.get_historical_data(pair, period='3mo', interval='1h')
        
        indicators = self.calculate_technical_indicators(data)
        
        signal = {
            'pair': pair,
            'timestamp': datetime.now(),
            'action': 'HOLD',
            'confidence': 50,
            'reasons': [],
            'price': self.get_current_price(pair),
            'indicators': indicators
        }
        
        try:
            confidence_factors = []
            
            # RSI-based signals
            if 'rsi' in indicators and indicators['rsi']:
                rsi = indicators['rsi']
                if rsi > 70:
                    signal['reasons'].append('RSI oversold')
                    confidence_factors.append(-20)  # Bearish
                elif rsi < 30:
                    signal['reasons'].append('RSI overbought')
                    confidence_factors.append(20)  # Bullish
            
            # Moving Average signals
            if all(k in indicators for k in ['sma_20', 'sma_50']) and all(indicators[k] for k in ['sma_20', 'sma_50']):
                if indicators['sma_20'] > indicators['sma_50']:
                    signal['reasons'].append('Short MA above Long MA')
                    confidence_factors.append(15)  # Bullish
                else:
                    signal['reasons'].append('Short MA below Long MA')
                    confidence_factors.append(-15)  # Bearish
            
            # MACD signals
            if 'macd' in indicators and indicators['macd']:
                if indicators['macd'] > 0:
                    signal['reasons'].append('MACD bullish')
                    confidence_factors.append(10)
                else:
                    signal['reasons'].append('MACD bearish')
                    confidence_factors.append(-10)
            
            # Bollinger Bands signals
            current_price = signal['price']
            if current_price and all(k in indicators for k in ['bb_upper', 'bb_lower']) and all(indicators[k] for k in ['bb_upper', 'bb_lower']):
                if current_price > indicators['bb_upper']:
                    signal['reasons'].append('Price above Bollinger Upper')
                    confidence_factors.append(-15)  # Bearish
                elif current_price < indicators['bb_lower']:
                    signal['reasons'].append('Price below Bollinger Lower')
                    confidence_factors.append(15)  # Bullish
            
            # Calculate overall signal
            if confidence_factors:
                total_confidence = sum(confidence_factors)
                
                if total_confidence > 15:
                    signal['action'] = 'BUY'
                    signal['confidence'] = min(50 + abs(total_confidence), 95)
                elif total_confidence < -15:
                    signal['action'] = 'SELL'
                    signal['confidence'] = min(50 + abs(total_confidence), 95)
                else:
                    signal['action'] = 'HOLD'
                    signal['confidence'] = 50 + abs(total_confidence) // 2
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
        
        return signal
    
    def _simulate_price(self, pair: str) -> float:
        """Generate simulated price data for demo purposes."""
        # Base prices for major pairs
        base_prices = {
            'EUR/USD': 1.0850, 'GBP/USD': 1.2650, 'USD/JPY': 149.50,
            'AUD/USD': 0.6450, 'USD/CHF': 0.9150, 'EUR/GBP': 0.8650,
            'NZD/USD': 0.5950, 'USD/CAD': 1.3650, 'EUR/JPY': 162.30,
            'GBP/JPY': 189.20, 'AUD/JPY': 96.40, 'CHF/JPY': 163.50
        }
        
        base_price = base_prices.get(pair, 1.0000)
        # Add some random variation
        variation = np.random.uniform(-0.01, 0.01)
        return base_price * (1 + variation)
    
    def _simulate_historical_data(self, pair: str, period: str) -> pd.DataFrame:
        """Generate simulated historical data for demo purposes."""
        # Determine number of days based on period
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }
        days = period_map.get(period, 365)
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Start with base price
        base_price = self._simulate_price(pair)
        
        # Generate price series with random walk
        returns = np.random.normal(0, 0.01, len(dates))
        prices = [base_price]
        
        for return_rate in returns[1:]:
            new_price = prices[-1] * (1 + return_rate)
            prices.append(new_price)
        
        # Create OHLC data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC
            variation = abs(np.random.normal(0, 0.005))
            high = close + variation
            low = close - variation
            open_price = close + np.random.normal(0, 0.002)
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'datetime': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)

# Global instance for easy access
data_handler = DataHandler()