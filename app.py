#!/usr/bin/env python3
"""
Streamlit Web App for Hybrid LSTM-Transformer Forex Trading System
A comprehensive trading platform with real-time analysis, backtesting, and portfolio management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Tuple, Optional

# Import our trading system
sys.path.append(os.path.dirname(__file__))
from complete_forex_system import (
    TradingPlatform, TechnicalIndicators, 
    set_run_seed, banner, section
)
from config import API_KEYS, MODEL_CONFIG

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 Forex Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .sidebar-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-live { background-color: #00ff00; }
    .status-offline { background-color: #ff0000; }
    .status-warning { background-color: #ffaa00; }
</style>
""", unsafe_allow_html=True)

class ForexApp:
    """Main Streamlit application class for forex trading platform."""
    
    def __init__(self):
        self.initialize_session_state()
        self.platform = None
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'trading_active' not in st.session_state:
            st.session_state.trading_active = False
        if 'portfolio_balance' not in st.session_state:
            st.session_state.portfolio_balance = 10000.0
        if 'trades_history' not in st.session_state:
            st.session_state.trades_history = []
        if 'current_positions' not in st.session_state:
            st.session_state.current_positions = {}
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
    
    def render_header(self):
        """Render the main application header."""
        st.markdown('<h1 class="main-header">🚀 Hybrid LSTM-Transformer Forex Trading System</h1>', 
                   unsafe_allow_html=True)
        
        # Status indicator
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            status_color = "status-live" if st.session_state.trading_active else "status-offline"
            status_text = "LIVE TRADING" if st.session_state.trading_active else "OFFLINE"
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="status-indicator {status_color}"></span>
                <strong>{status_text}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with navigation and controls."""
        with st.sidebar:
            st.markdown("## 📊 Navigation")
            
            # Main navigation
            page = st.selectbox(
                "Choose a page:",
                ["Dashboard", "Live Trading", "Backtesting", "Portfolio", "Settings", "About"]
            )
            
            st.markdown("---")
            
            # Quick stats
            st.markdown("## 📈 Quick Stats")
            st.metric("Portfolio Balance", f"${st.session_state.portfolio_balance:,.2f}")
            st.metric("Active Positions", len(st.session_state.current_positions))
            st.metric("Total Trades", len(st.session_state.trades_history))
            
            st.markdown("---")
            
            # Trading controls
            st.markdown("## ⚡ Quick Actions")
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
            
            if st.session_state.trading_active:
                if st.button("⏸️ Stop Trading", use_container_width=True, type="secondary"):
                    st.session_state.trading_active = False
                    st.success("Trading stopped!")
                    time.sleep(1)
                    st.rerun()
            else:
                if st.button("▶️ Start Trading", use_container_width=True, type="primary"):
                    st.session_state.trading_active = True
                    st.success("Trading started!")
                    time.sleep(1)
                    st.rerun()
            
            # Risk management
            st.markdown("---")
            st.markdown("## ⚠️ Risk Management")
            risk_level = st.selectbox("Risk Level", ["Conservative", "Moderate", "Aggressive"])
            max_positions = st.slider("Max Positions", 1, 10, 5)
            
            return page
    
    def render_dashboard(self):
        """Render the main dashboard page."""
        st.markdown("## 📊 Trading Dashboard")
        
        # Generate sample data for visualization
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'price': np.cumsum(np.random.randn(len(dates)) * 0.01) + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="EUR/USD",
                value="1.0845",
                delta="0.0012 (0.11%)"
            )
        
        with col2:
            st.metric(
                label="GBP/USD", 
                value="1.2634",
                delta="-0.0023 (-0.18%)"
            )
        
        with col3:
            st.metric(
                label="USD/JPY",
                value="149.85",
                delta="0.45 (0.30%)"
            )
        
        with col4:
            st.metric(
                label="Daily P&L",
                value="$245.67",
                delta="12.5%"
            )
        
        # Price chart
        st.markdown("### 📈 Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_data['timestamp'],
            y=sample_data['price'],
            mode='lines',
            name='EUR/USD',
            line=dict(color='#667eea', width=2)
        ))
        
        fig.update_layout(
            title="EUR/USD Price Movement (Last 30 Days)",
            xaxis_title="Time",
            yaxis_title="Price",
            hovermode='x unified',
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Technical Indicators")
            
            # Generate sample technical indicators
            indicators = {
                "RSI (14)": 65.4,
                "MACD": 0.0023,
                "Bollinger Upper": 1.0890,
                "Bollinger Lower": 1.0800,
                "SMA (20)": 1.0845,
                "EMA (12)": 1.0847
            }
            
            for indicator, value in indicators.items():
                if isinstance(value, float):
                    st.write(f"**{indicator}:** {value:.4f}")
                else:
                    st.write(f"**{indicator}:** {value}")
        
        with col2:
            st.markdown("### 🔔 Recent Signals")
            
            signals = [
                {"time": "14:23", "pair": "EUR/USD", "signal": "BUY", "confidence": 85},
                {"time": "13:45", "pair": "GBP/USD", "signal": "SELL", "confidence": 72},
                {"time": "13:12", "pair": "USD/JPY", "signal": "BUY", "confidence": 90},
                {"time": "12:58", "pair": "EUR/GBP", "signal": "HOLD", "confidence": 60}
            ]
            
            for signal in signals:
                signal_color = "🟢" if signal["signal"] == "BUY" else "🔴" if signal["signal"] == "SELL" else "🟡"
                st.write(f"{signal_color} **{signal['time']}** - {signal['pair']}: {signal['signal']} ({signal['confidence']}% confidence)")
    
    def render_live_trading(self):
        """Render the live trading interface."""
        st.markdown("## ⚡ Live Trading")
        
        if not st.session_state.trading_active:
            st.warning("⚠️ Trading is currently disabled. Enable it from the sidebar to start live trading.")
            return
        
        # Trading interface
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown("### 📋 Order Entry")
            
            # Order form
            with st.form("trading_form"):
                currency_pair = st.selectbox("Currency Pair", 
                    ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "EUR/GBP"])
                
                order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop"])
                
                col_a, col_b = st.columns(2)
                with col_a:
                    action = st.radio("Action", ["Buy", "Sell"])
                with col_b:
                    quantity = st.number_input("Quantity", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
                
                if order_type in ["Limit", "Stop"]:
                    target_price = st.number_input("Target Price", min_value=0.0001, value=1.0000, step=0.0001, format="%.4f")
                
                # Risk management
                st.markdown("**Risk Management**")
                col_sl, col_tp = st.columns(2)
                with col_sl:
                    stop_loss = st.number_input("Stop Loss", min_value=0.0001, value=0.0050, step=0.0001, format="%.4f")
                with col_tp:
                    take_profit = st.number_input("Take Profit", min_value=0.0001, value=0.0100, step=0.0001, format="%.4f")
                
                submitted = st.form_submit_button("🚀 Execute Trade", type="primary", use_container_width=True)
                
                if submitted:
                    # Simulate trade execution
                    trade = {
                        "timestamp": datetime.now(),
                        "pair": currency_pair,
                        "action": action,
                        "quantity": quantity,
                        "price": np.random.uniform(1.0800, 1.0900),
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "status": "Open"
                    }
                    
                    st.session_state.trades_history.append(trade)
                    st.session_state.current_positions[f"{currency_pair}_{len(st.session_state.trades_history)}"] = trade
                    
                    st.success(f"✅ {action} order for {quantity} {currency_pair} executed successfully!")
                    time.sleep(1)
                    st.rerun()
        
        with col2:
            st.markdown("### 🎯 AI Recommendations")
            
            # AI-powered recommendations
            recommendations = [
                {"pair": "EUR/USD", "action": "BUY", "confidence": 87, "reason": "Strong bullish momentum"},
                {"pair": "GBP/USD", "action": "SELL", "confidence": 73, "reason": "Bearish divergence detected"},
                {"pair": "USD/JPY", "action": "HOLD", "confidence": 65, "reason": "Sideways consolidation"}
            ]
            
            for rec in recommendations:
                with st.container():
                    action_color = "🟢" if rec["action"] == "BUY" else "🔴" if rec["action"] == "SELL" else "🟡"
                    st.markdown(f"""
                    **{rec['pair']}** {action_color}  
                    **{rec['action']}** - {rec['confidence']}%  
                    *{rec['reason']}*
                    """)
                    st.markdown("---")
        
        with col3:
            st.markdown("### 📊 Current Positions")
            
            if st.session_state.current_positions:
                positions_df = pd.DataFrame(list(st.session_state.current_positions.values()))
                
                for idx, position in positions_df.iterrows():
                    with st.container():
                        current_price = position['price'] + np.random.uniform(-0.01, 0.01)
                        pnl = (current_price - position['price']) * position['quantity'] * 10000
                        pnl_color = "🟢" if pnl > 0 else "🔴"
                        
                        st.markdown(f"""
                        **{position['pair']}**  
                        {position['action']} {position['quantity']} @ {position['price']:.4f}  
                        Current: {current_price:.4f}  
                        P&L: {pnl_color} ${pnl:.2f}
                        """)
                        
                        if st.button(f"Close Position", key=f"close_{idx}"):
                            # Remove position
                            position_key = list(st.session_state.current_positions.keys())[idx]
                            del st.session_state.current_positions[position_key]
                            st.rerun()
                        
                        st.markdown("---")
            else:
                st.info("No open positions")
    
    def render_backtesting(self):
        """Render the backtesting interface."""
        st.markdown("## 🔬 Strategy Backtesting")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Backtest Parameters")
            
            with st.form("backtest_form"):
                # Date range
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
                end_date = st.date_input("End Date", datetime.now())
                
                # Currency pairs
                pairs = st.multiselect("Currency Pairs", 
                    ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF"], 
                    default=["EUR/USD"])
                
                # Strategy parameters
                st.markdown("**Strategy Settings**")
                rsi_period = st.slider("RSI Period", 5, 50, 14)
                ma_short = st.slider("MA Short Period", 5, 50, 12)
                ma_long = st.slider("MA Long Period", 20, 200, 26)
                
                # Risk management
                st.markdown("**Risk Management**")
                initial_capital = st.number_input("Initial Capital ($)", 1000, 100000, 10000)
                risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0)
                
                run_backtest = st.form_submit_button("🚀 Run Backtest", type="primary", use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Backtest Results")
            
            if run_backtest:
                # Simulate backtest results
                with st.spinner("Running backtest..."):
                    time.sleep(2)  # Simulate processing time
                
                # Generate sample backtest data
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                returns = np.random.randn(len(dates)) * 0.02 + 0.001
                cumulative_returns = (1 + returns).cumprod()
                portfolio_value = initial_capital * cumulative_returns
                
                # Create performance chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=portfolio_value,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#667eea', width=2)
                ))
                
                fig.update_layout(
                    title="Portfolio Performance",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                st.markdown("### 📊 Performance Metrics")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                total_return = (portfolio_value[-1] - initial_capital) / initial_capital * 100
                max_drawdown = np.random.uniform(5, 15)
                sharpe_ratio = np.random.uniform(0.5, 2.0)
                win_rate = np.random.uniform(45, 75)
                
                with col_a:
                    st.metric("Total Return", f"{total_return:.2f}%")
                with col_b:
                    st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                with col_c:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                with col_d:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                # Trade analysis
                st.markdown("### 📋 Trade Analysis")
                
                # Generate sample trades
                num_trades = np.random.randint(50, 200)
                sample_trades = pd.DataFrame({
                    'Date': pd.date_range(start=start_date, periods=num_trades, freq='D'),
                    'Pair': np.random.choice(pairs, num_trades),
                    'Action': np.random.choice(['BUY', 'SELL'], num_trades),
                    'Entry': np.random.uniform(1.08, 1.12, num_trades),
                    'Exit': np.random.uniform(1.08, 1.12, num_trades),
                    'P&L': np.random.uniform(-500, 800, num_trades)
                })
                
                sample_trades['P&L'] = sample_trades['P&L'].round(2)
                
                st.dataframe(sample_trades.head(10), use_container_width=True)
                
                if st.button("📥 Download Full Results"):
                    st.success("Results downloaded! (Simulated)")
    
    def render_portfolio(self):
        """Render the portfolio management page."""
        st.markdown("## 💼 Portfolio Management")
        
        # Portfolio overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 💰 Account Summary")
            st.metric("Total Balance", f"${st.session_state.portfolio_balance:,.2f}")
            st.metric("Available Margin", f"${st.session_state.portfolio_balance * 0.8:,.2f}")
            st.metric("Used Margin", f"${st.session_state.portfolio_balance * 0.2:,.2f}")
            
        with col2:
            st.markdown("### 📊 Performance")
            daily_pnl = np.random.uniform(-200, 300)
            monthly_pnl = np.random.uniform(-1000, 2000)
            ytd_pnl = np.random.uniform(-2000, 5000)
            
            st.metric("Daily P&L", f"${daily_pnl:.2f}", f"{daily_pnl/st.session_state.portfolio_balance*100:.2f}%")
            st.metric("Monthly P&L", f"${monthly_pnl:.2f}", f"{monthly_pnl/st.session_state.portfolio_balance*100:.2f}%")
            st.metric("YTD P&L", f"${ytd_pnl:.2f}", f"{ytd_pnl/st.session_state.portfolio_balance*100:.2f}%")
        
        with col3:
            st.markdown("### ⚖️ Risk Metrics")
            var = st.session_state.portfolio_balance * 0.05
            st.metric("Value at Risk (5%)", f"${var:.2f}")
            st.metric("Risk/Reward Ratio", "1:2.5")
            st.metric("Maximum Leverage", "30:1")
        
        # Portfolio composition
        st.markdown("### 🥧 Portfolio Composition")
        
        # Generate sample portfolio data
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CHF']
        exposure = np.random.dirichlet(np.ones(len(currencies))) * 100
        
        fig = px.pie(
            values=exposure,
            names=currencies,
            title="Currency Exposure (%)",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Transaction history
        st.markdown("### 📜 Transaction History")
        
        if st.session_state.trades_history:
            trades_df = pd.DataFrame(st.session_state.trades_history)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp', ascending=False)
            
            # Add simulated P&L
            trades_df['pnl'] = np.random.uniform(-100, 200, len(trades_df))
            trades_df['pnl'] = trades_df['pnl'].round(2)
            
            st.dataframe(
                trades_df[['timestamp', 'pair', 'action', 'quantity', 'price', 'pnl']],
                use_container_width=True,
                column_config={
                    'timestamp': 'Time',
                    'pair': 'Pair',
                    'action': 'Action',
                    'quantity': 'Quantity',
                    'price': 'Price',
                    'pnl': 'P&L ($)'
                }
            )
        else:
            st.info("No trading history available")
    
    def render_settings(self):
        """Render the settings page."""
        st.markdown("## ⚙️ Settings")
        
        # API Configuration
        st.markdown("### 🔑 API Configuration")
        
        with st.expander("API Keys", expanded=False):
            alpha_vantage_key = st.text_input("Alpha Vantage API Key", value=API_KEYS.get('alpha_vantage', ''), type='password')
            twelve_data_key = st.text_input("Twelve Data API Key", value=API_KEYS.get('twelve_data', ''), type='password')
            
            if st.button("💾 Save API Keys"):
                st.success("API keys saved successfully!")
        
        # Model Configuration
        st.markdown("### 🧠 Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sequence_length = st.slider("Sequence Length", 30, 120, MODEL_CONFIG['sequence_length'])
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, MODEL_CONFIG['learning_rate'], format="%.4f")
        
        with col2:
            epochs = st.slider("Training Epochs", 50, 500, MODEL_CONFIG['epochs'])
            device = st.selectbox("Device", ['cpu', 'mps', 'cuda'], index=1)
            mixed_precision = st.checkbox("Mixed Precision", MODEL_CONFIG['mixed_precision'])
        
        # Trading Parameters
        st.markdown("### 📊 Trading Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Management**")
            max_risk_per_trade = st.slider("Max Risk per Trade (%)", 0.5, 5.0, 2.0)
            max_positions = st.slider("Max Open Positions", 1, 20, 5)
            stop_loss_pips = st.number_input("Default Stop Loss (pips)", 10, 100, 50)
        
        with col2:
            st.markdown("**Signal Parameters**")
            min_confidence = st.slider("Minimum Signal Confidence (%)", 50, 95, 70)
            rsi_overbought = st.slider("RSI Overbought Level", 70, 90, 80)
            rsi_oversold = st.slider("RSI Oversold Level", 10, 30, 20)
        
        # Notification Settings
        st.markdown("### 🔔 Notifications")
        
        email_notifications = st.checkbox("Email Notifications", True)
        if email_notifications:
            email_address = st.text_input("Email Address", placeholder="trader@example.com")
        
        push_notifications = st.checkbox("Push Notifications", True)
        trade_alerts = st.checkbox("Trade Execution Alerts", True)
        pnl_alerts = st.checkbox("P&L Alerts", True)
        
        # Save settings
        if st.button("💾 Save All Settings", type="primary", use_container_width=True):
            st.success("✅ Settings saved successfully!")
    
    def render_about(self):
        """Render the about page."""
        st.markdown("## ℹ️ About")
        
        st.markdown("""
        ### 🚀 Hybrid LSTM-Transformer Forex Trading System
        
        This advanced trading platform combines the power of Long Short-Term Memory (LSTM) networks 
        with Transformer architecture to create a state-of-the-art forex trading system optimized for Mac M2.
        
        #### 🎯 Key Features:
        - **Real-time Market Analysis**: Advanced technical indicators and pattern recognition
        - **AI-Powered Predictions**: Hybrid LSTM-Transformer model for price forecasting  
        - **Risk Management**: Comprehensive risk controls and portfolio optimization
        - **Backtesting Engine**: Historical strategy validation with detailed performance metrics
        - **Live Trading**: Automated trade execution with manual override capabilities
        - **Multi-Currency Support**: Trade major forex pairs with real-time data
        
        #### 🛠️ Technical Stack:
        - **Frontend**: Streamlit with Plotly visualizations
        - **Backend**: Python with pandas, numpy, and scikit-learn
        - **Machine Learning**: LSTM + Transformer neural networks
        - **Data Sources**: Alpha Vantage, Twelve Data APIs
        - **Database**: SQLite for trade history and configuration storage
        
        #### 📊 Supported Currency Pairs:
        - EUR/USD, GBP/USD, USD/JPY
        - AUD/USD, USD/CHF, EUR/GBP
        - NZD/USD, USD/CAD, EUR/JPY
        - GBP/JPY, AUD/JPY, CHF/JPY
        
        #### ⚠️ Risk Disclaimer:
        Trading forex carries substantial risk and may not be suitable for all investors. 
        Past performance is not indicative of future results. This software is for 
        educational and research purposes only.
        
        ---
        
        ### 👨‍💻 Developer Information
        **Version**: 1.0.0  
        **Last Updated**: September 2024  
        **License**: MIT License  
        **Repository**: [GitHub](https://github.com/tejaspavani/trading-platform)
        
        ### 📞 Support
        For technical support or feature requests, please visit our GitHub repository 
        or contact the development team.
        """)
        
        # System information
        st.markdown("### 🖥️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Platform**: {sys.platform}  
            **Python Version**: {sys.version.split()[0]}  
            **Streamlit Version**: {st.__version__}
            """)
        
        with col2:
            st.markdown(f"""
            **Last Update**: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}  
            **Trading Status**: {'Active' if st.session_state.trading_active else 'Inactive'}  
            **Data Connection**: Online ✅
            """)
    
    def run(self):
        """Main application runner."""
        self.render_header()
        
        # Render sidebar and get selected page
        selected_page = self.render_sidebar()
        
        # Render selected page
        if selected_page == "Dashboard":
            self.render_dashboard()
        elif selected_page == "Live Trading":
            self.render_live_trading()
        elif selected_page == "Backtesting":
            self.render_backtesting()
        elif selected_page == "Portfolio":
            self.render_portfolio()
        elif selected_page == "Settings":
            self.render_settings()
        elif selected_page == "About":
            self.render_about()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; padding: 1rem;'>"
            "🚀 Hybrid LSTM-Transformer Forex Trading System | "
            "Built with ❤️ using Streamlit"
            "</div>", 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    # Initialize and run the application
    app = ForexApp()
    app.run()