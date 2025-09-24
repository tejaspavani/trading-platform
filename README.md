# 🚀 Hybrid LSTM-Transformer Forex Trading Platform

A comprehensive web-based forex trading platform that combines advanced machine learning techniques with real-time market analysis and automated trading capabilities.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## 🌟 Features

### 📊 Real-time Trading Dashboard
- Live forex price monitoring for 12+ currency pairs
- Interactive charts with technical indicators
- Real-time P&L tracking and portfolio management
- AI-powered trading signals with confidence levels

### 🤖 Advanced AI Trading System
- Hybrid LSTM-Transformer neural network architecture
- 23+ technical indicators for comprehensive market analysis
- Automated signal generation with customizable confidence thresholds
- Risk-adjusted position sizing and portfolio optimization

### 📈 Comprehensive Backtesting Engine
- Historical strategy validation with detailed performance metrics
- Customizable date ranges and currency pair selection
- Risk management simulation with drawdown analysis
- Exportable results and trade analysis

### 💼 Portfolio Management
- Real-time balance and equity tracking
- Position management with stop-loss and take-profit orders
- Risk exposure monitoring and compliance checks
- Detailed transaction history and performance analytics

### ⚙️ Configuration Management
- API key management for data providers
- Trading parameter customization
- Risk management settings
- Notification preferences and alerts

## 🏗️ Architecture

```
trading-platform/
├── app.py                      # Main Streamlit application
├── complete_forex_system.py    # Core trading system (existing)
├── data_handler.py            # Data fetching and processing
├── trading_utils.py           # Trading utilities and database
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── trading_platform.db       # SQLite database
└── README.md                  # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for real-time data

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tejaspavani/trading-platform.git
   cd trading-platform
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys** (Optional)
   
   Edit `config.py` to add your API keys:
   ```python
   API_KEYS = {
       'alpha_vantage': 'your_alpha_vantage_key',
       'twelve_data': 'your_twelve_data_key',
   }
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the platform**
   
   Open your browser and navigate to `http://localhost:8501`

## 💻 Usage

### Dashboard Overview
The main dashboard provides:
- Real-time currency pair prices with change indicators
- Interactive price charts with technical analysis
- Recent trading signals with confidence levels
- Portfolio metrics and performance statistics

### Live Trading
1. Navigate to the "Live Trading" page
2. Enable trading from the sidebar
3. Configure trade parameters:
   - Currency pair selection
   - Order type (Market, Limit, Stop)
   - Position size and risk management
4. Execute trades manually or follow AI recommendations

### Backtesting
1. Go to the "Backtesting" page
2. Select date range and currency pairs
3. Configure strategy parameters:
   - RSI period (5-50)
   - Moving average periods
   - Risk per trade percentage
4. Run backtest and analyze results

### Portfolio Management
- Monitor account balance and equity
- Track open positions and P&L
- View transaction history
- Analyze performance metrics

## 📊 Supported Currency Pairs

- **Major Pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, EUR/GBP
- **Cross Pairs**: NZD/USD, USD/CAD, EUR/JPY, GBP/JPY, AUD/JPY, CHF/JPY

## 🔧 Technical Indicators

The platform includes 23+ technical indicators:

### Trend Indicators
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Moving Average Convergence Divergence (MACD)
- Parabolic SAR

### Momentum Indicators
- Relative Strength Index (RSI)
- Stochastic Oscillator
- Williams %R
- Rate of Change (ROC)

### Volatility Indicators
- Bollinger Bands
- Average True Range (ATR)
- Standard Deviation
- Commodity Channel Index (CCI)

### Volume Indicators
- On-Balance Volume (OBV)
- Volume Rate of Change
- Accumulation/Distribution Line

## ⚠️ Risk Management

### Built-in Risk Controls
- **Position Sizing**: Automatic calculation based on account balance and risk percentage
- **Stop Loss**: Mandatory stop-loss orders for all positions
- **Maximum Exposure**: Configurable limits on total portfolio exposure
- **Drawdown Protection**: Real-time monitoring and alerts

### Best Practices
1. **Never risk more than 2% per trade**
2. **Set stop-loss orders for all positions**
3. **Diversify across multiple currency pairs**
4. **Regularly review and adjust risk parameters**
5. **Use the backtesting engine to validate strategies**

## 🔌 API Integration

### Supported Data Providers
- **Yahoo Finance**: Free real-time and historical data
- **Alpha Vantage**: Professional-grade forex data (API key required)
- **Twelve Data**: Comprehensive market data (API key required)

### Data Sources
- Real-time price feeds
- Historical OHLCV data
- Economic indicators
- Market sentiment data

## 🛠️ Development

### Project Structure
```
├── Core Components
│   ├── app.py              # Main Streamlit application
│   ├── complete_forex_system.py  # Trading algorithm engine
│   └── config.py           # Configuration management
├── Data Layer
│   ├── data_handler.py     # Data fetching and processing
│   └── trading_platform.db # SQLite database
├── Utilities
│   ├── trading_utils.py    # Trading utilities and helpers
│   └── requirements.txt    # Dependencies
└── Documentation
    └── README.md           # This file
```

### Key Classes
- **ForexApp**: Main Streamlit application controller
- **DataHandler**: Real-time and historical data management
- **TradingDatabase**: Persistent storage for trades and signals
- **PortfolioManager**: Portfolio calculations and metrics
- **RiskManager**: Risk validation and compliance
- **BacktestEngine**: Historical strategy testing

### Adding New Features
1. Create feature branch from main
2. Implement functionality with proper error handling
3. Add unit tests where applicable
4. Update documentation
5. Submit pull request for review

## 📈 Performance Optimization

### Mac M2 Optimizations
- **MPS Device**: Utilizes Apple's Metal Performance Shaders
- **Mixed Precision**: Faster training with reduced memory usage
- **Optimized Batch Sizes**: Configured for M2 architecture
- **Efficient Data Loading**: Streamlined data pipelines

### Caching Strategy
- API response caching for reduced latency
- Historical data caching for improved performance
- Session state management for user preferences

## 🔒 Security

### Data Protection
- API keys stored securely in configuration files
- SQLite database with proper access controls
- Input validation and sanitization
- Error handling without sensitive data exposure

### Trading Security
- Position validation before execution
- Risk limit enforcement
- Audit trail for all trading activities
- Secure session management

## 🧪 Testing

### Manual Testing
1. **Dashboard Functionality**: Verify all metrics and charts load correctly
2. **Data Integration**: Test real-time data feeds and historical data
3. **Trading Operations**: Validate trade execution and position management
4. **Risk Controls**: Verify risk management rules are enforced
5. **Backtesting**: Test strategy validation with known datasets

### Performance Testing
- Load testing with multiple concurrent users
- Memory usage optimization
- Database performance optimization
- API rate limit handling

## 🚨 Troubleshooting

### Common Issues

**Data Loading Errors**
```bash
# Check internet connection and API keys
# Verify config.py settings
# Review logs for specific error messages
```

**Database Errors**
```bash
# Reset database (warning: loses data)
rm trading_platform.db
python -c "from trading_utils import trading_db; trading_db.init_database()"
```

**Streamlit Performance**
```bash
# Clear Streamlit cache
streamlit cache clear
```

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run app.py
```

## 📋 Roadmap

### Version 1.1 (Planned)
- [ ] Real-time WebSocket data feeds
- [ ] Advanced chart patterns recognition
- [ ] Social trading features
- [ ] Mobile-responsive design improvements

### Version 1.2 (Planned)
- [ ] Multi-timeframe analysis
- [ ] Custom indicator builder
- [ ] Automated strategy optimization
- [ ] Integration with popular brokers

### Version 2.0 (Future)
- [ ] Cryptocurrency trading support
- [ ] Advanced AI models (GPT integration)
- [ ] Cloud deployment options
- [ ] Professional trader collaboration tools

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper documentation
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/trading-platform.git
cd trading-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run application
streamlit run app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important Risk Warning**: Trading foreign exchange (forex) carries a high level of risk and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade forex, you should carefully consider your investment objectives, level of experience, and risk appetite.

This software is provided for educational and research purposes only. The developers are not responsible for any financial losses incurred through the use of this platform. Always consult with qualified financial advisors before making trading decisions.

Past performance is not indicative of future results. All trading involves risk, and you could lose more than your initial investment.

## 📞 Support

- **Documentation**: This README and inline code comments
- **Issues**: [GitHub Issues](https://github.com/tejaspavani/trading-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tejaspavani/trading-platform/discussions)
- **Email**: For private inquiries (check repository for contact information)

## 🙏 Acknowledgments

- **Streamlit Team**: For the excellent web app framework
- **Plotly**: For interactive charting capabilities
- **Yahoo Finance**: For providing free market data
- **Alpha Vantage & Twelve Data**: For professional data APIs
- **Open Source Community**: For the various Python libraries that make this project possible

---

**Built with ❤️ by the Trading Platform Team**

*Last Updated: September 2024*