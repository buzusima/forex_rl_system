# config.py - Central Configuration สำหรับ Forex RL Trading System
"""
ไฟล์นี้เป็นหัวใจของการตั้งค่าทั้งระบบ
แก้ไขไฟล์นี้เมื่อ: เปลี่ยน symbols, timeframes, หรือ trading parameters
"""

import os
from typing import List, Dict, Any
from datetime import datetime, timedelta

class ForexRLConfig:
    """Configuration class สำหรับ Forex RL Trading System"""
    
    # ============= TRADING SYMBOLS =============
    # 28 คู่เงินที่เลือกใช้ (ปรับจาก 26 เป็น 28)
    TRADING_SYMBOLS: List[str] = [
        # Major Pairs (7 คู่)
        'EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 
        'AUDUSD', 'NZDUSD', 'USDCAD',
        
        # EUR Crosses (6 คู่)
        'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 
        'EURJPY', 'EURNZD',
        
        # GBP Crosses (5 คู่)
        'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD',
        
        # Other Crosses (10 คู่)
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD',
        'CADCHF', 'CADJPY', 'CHFJPY', 
        'NZDCAD', 'NZDCHF', 'NZDJPY'
        
        
    ]
    
    # ============= TIMEFRAMES =============
    # 5 timeframes ที่ใช้
    TIMEFRAMES: Dict[str, int] = {
        'M5': 5,     # 5 minutes
        'M15': 15,   # 15 minutes  
        'M30': 30,   # 30 minutes
        'H1': 60,    # 1 hour
        'H4': 240    # 4 hours
    }
    
    # MT5 Timeframe constants
    MT5_TIMEFRAMES = {
        'M5': 16385,   # TIMEFRAME_M5
        'M15': 16386,  # TIMEFRAME_M15
        'M30': 16387,  # TIMEFRAME_M30
        'H1': 16388,   # TIMEFRAME_H1
        'H4': 16390    # TIMEFRAME_H4
    }
    
    # ============= DATA SETTINGS =============
    DATA_CONFIG = {
        # Historical data settings
        'HISTORICAL_DAYS': 1095,  # 3 years of data
        'MIN_DATA_POINTS': 10000,  # Minimum data points per symbol
        'DATA_VALIDATION': True,   # Enable data validation
        
        # Real-time data settings
        'REALTIME_COLLECTION': True,
        'COLLECTION_INTERVAL': 5,  # seconds
        'MAX_MISSING_TICKS': 10,   # Maximum allowed missing ticks
        
        # Storage settings
        'COMPRESS_DATA': True,
        'BACKUP_FREQUENCY': 'daily',
        'RETENTION_DAYS': 2555,  # 7 years
    }
    
    # ============= DATABASE SETTINGS =============
    DATABASE_CONFIG = {
        'TYPE': 'timescaledb',  # timescaledb, influxdb, postgresql
        'HOST': 'localhost',
        'PORT': 5432,
        'NAME': 'forex_rl_db',
        'USER': 'forex_user',
        'PASSWORD': 'secure_password_here',  # แก้ไขตามความเหมาะสม
        
        # Performance settings
        'CONNECTION_POOL_SIZE': 20,
        'MAX_CONNECTIONS': 100,
        'QUERY_TIMEOUT': 30,
        
        # Table settings
        'CHUNK_TIME_INTERVAL': '1 day',  # TimescaleDB hypertable chunk interval
        'COMPRESSION_AFTER': '7 days',   # Compress data older than 7 days
    }
    
    # ============= RL SETTINGS =============
    RL_CONFIG = {
        # Environment settings
        'MAX_STEPS_PER_EPISODE': 10000,
        'OBSERVATION_WINDOW': 100,  # lookback window
        'ACTION_SPACE_SIZE': 10,    # number of possible actions
        
        # Multi-agent settings
        'USE_MULTI_AGENT': True,
        'AGENTS_PER_CURRENCY_BLOCK': 1,
        'PORTFOLIO_AGENT': True,
        
        # Training settings
        'ALGORITHM': 'PPO',  # PPO, SAC, TD3
        'LEARNING_RATE': 3e-4,
        'BATCH_SIZE': 256,
        'BUFFER_SIZE': 1000000,
        'GAMMA': 0.99,  # discount factor
        
        # Model architecture
        'HIDDEN_LAYERS': [512, 256, 128],
        'ACTIVATION': 'relu',
        'USE_LSTM': True,
        'LSTM_UNITS': 64,
        
        # Feature settings
        'USE_TECHNICAL_INDICATORS': True,
        'USE_CORRELATION_FEATURES': True,
        'USE_MARKET_REGIME_FEATURES': True,
        'FEATURE_NORMALIZATION': 'minmax',  # minmax, zscore, robust
    }
    
    # ============= RISK MANAGEMENT =============
    RISK_CONFIG = {
        # Position sizing
        'MAX_RISK_PER_TRADE': 0.01,    # 1% risk per trade
        'MAX_RISK_PER_CURRENCY': 0.04,  # 4% risk per currency
        'MAX_TOTAL_RISK': 0.12,         # 12% total portfolio risk
        'MIN_POSITION_SIZE': 0.01,       # Minimum lot size
        'MAX_POSITION_SIZE': 2.0,        # Maximum lot size
        
        # Portfolio limits
        'MAX_POSITIONS': 15,             # Maximum concurrent positions
        'MAX_CORRELATION_EXPOSURE': 0.06, # 6% max exposure to correlated pairs
        
        # Stop loss and take profit
        'DYNAMIC_STOPS': True,
        'MIN_STOP_LOSS_PIPS': 10,
        'MAX_STOP_LOSS_PIPS': 100,
        'RISK_REWARD_RATIO': 2.0,
        
        # Emergency stops
        'MAX_DAILY_DRAWDOWN': 0.05,     # 5% daily drawdown limit
        'MAX_TOTAL_DRAWDOWN': 0.15,     # 15% total drawdown limit
        'CIRCUIT_BREAKER_LOSS': 0.08,   # 8% loss triggers circuit breaker
    }
    
    # ============= BACKTESTING SETTINGS =============
    BACKTEST_CONFIG = {
        'START_DATE': '2020-01-01',
        'END_DATE': '2023-12-31',
        'INITIAL_BALANCE': 100000,  # $100,000
        'COMMISSION': 0.00003,      # 3 pips equivalent
        'SPREAD_MODEL': 'realistic', # realistic, optimistic, pessimistic
        'SLIPPAGE_MODEL': 'normal',  # none, normal, high
        
        # Performance metrics
        'BENCHMARK': 'EURUSD_BUY_HOLD',
        'RISK_FREE_RATE': 0.02,  # 2% annual
        'TARGET_SHARPE': 2.0,
        'TARGET_CALMAR': 3.0,
    }
    
    # ============= LIVE TRADING SETTINGS =============
    LIVE_CONFIG = {
        'PAPER_TRADING': True,      # Start with paper trading
        'REAL_TRADING': False,      # Enable real trading when ready
        'MAX_ORDERS_PER_MINUTE': 10,
        'ORDER_TIMEOUT': 30,        # seconds
        'EXECUTION_DELAY': 0.1,     # seconds
        
        # Monitoring
        'HEALTH_CHECK_INTERVAL': 60,  # seconds
        'PERFORMANCE_LOG_INTERVAL': 300,  # 5 minutes
        'DAILY_REPORT': True,
        'TELEGRAM_ALERTS': False,    # Enable when needed
    }
    
    # ============= TECHNICAL INDICATORS =============
    INDICATORS_CONFIG = {
        'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
        'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
        'ATR': {'period': 14},
        'BOLLINGER_BANDS': {'period': 20, 'std': 2},
        'EMA': {'periods': [9, 21, 50, 200]},
        'SMA': {'periods': [20, 50, 100]},
        'STOCHASTIC': {'k_period': 14, 'd_period': 3},
        'ADX': {'period': 14},
        'CCI': {'period': 20},
        'WILLIAMS_R': {'period': 14}
    }
    
    # ============= CORRELATION SETTINGS =============
    CORRELATION_CONFIG = {
        'ROLLING_WINDOW': 50,       # periods for rolling correlation
        'HIGH_CORRELATION_THRESHOLD': 0.8,
        'CORRELATION_BREAK_THRESHOLD': 0.3,  # alert if correlation changes by this much
        'UPDATE_FREQUENCY': 'realtime',      # realtime, hourly, daily
        
        # Correlation groups
        'CURRENCY_BLOCKS': {
            'USD': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'],
            'EUR': ['EURUSD', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD'],
            'GBP': ['GBPUSD', 'EURGBP', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD'],
            'JPY': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY', 'NZDJPY'],
            'COMMODITY': ['AUDUSD', 'NZDUSD', 'USDCAD']  # Commodity currencies
        }
    }
    
    # ============= LOGGING SETTINGS =============
    LOGGING_CONFIG = {
        'LEVEL': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        'LOG_TO_FILE': True,
        'LOG_TO_CONSOLE': True,
        'LOG_ROTATION': 'daily',
        'MAX_LOG_FILES': 30,
        'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        
        # Log directories
        'LOG_DIR': 'logs/',
        'TRADE_LOG_DIR': 'logs/trades/',
        'ERROR_LOG_DIR': 'logs/errors/',
        'PERFORMANCE_LOG_DIR': 'logs/performance/'
    }
    
    # ============= DIRECTORIES =============
    DIRECTORIES = {
        'DATA_DIR': 'data/',
        'RAW_DATA_DIR': 'data/raw/',
        'PROCESSED_DATA_DIR': 'data/processed/',
        'FEATURES_DIR': 'data/features/',
        'MODELS_DIR': 'models/',
        'BACKTEST_RESULTS_DIR': 'results/backtests/',
        'LIVE_RESULTS_DIR': 'results/live/',
        'TEMP_DIR': 'temp/'
    }
    
    @classmethod
    def create_directories(cls):
        """สร้าง directories ที่จำเป็นทั้งหมด"""
        for dir_path in cls.DIRECTORIES.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # สร้าง log directories
        for log_dir in [cls.LOGGING_CONFIG['LOG_DIR'], 
                       cls.LOGGING_CONFIG['TRADE_LOG_DIR'],
                       cls.LOGGING_CONFIG['ERROR_LOG_DIR'],
                       cls.LOGGING_CONFIG['PERFORMANCE_LOG_DIR']]:
            os.makedirs(log_dir, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """ตรวจสอบ configuration ให้ถูกต้อง"""
        # ตรวจสอบ symbols
        assert len(cls.TRADING_SYMBOLS) == 28, f"Expected 28 symbols, got {len(cls.TRADING_SYMBOLS)}"
        
        # ตรวจสอบ timeframes
        assert len(cls.TIMEFRAMES) == 5, f"Expected 5 timeframes, got {len(cls.TIMEFRAMES)}"
        
        # ตรวจสอบ risk parameters
        assert cls.RISK_CONFIG['MAX_RISK_PER_TRADE'] <= 0.02, "Risk per trade too high"
        assert cls.RISK_CONFIG['MAX_TOTAL_RISK'] <= 0.20, "Total risk too high"
        
        print("✅ Configuration validation passed!")
        return True
    
    @classmethod
    def get_symbol_info(cls, symbol: str) -> Dict[str, Any]:
        """ข้อมูล symbol ที่ใช้บ่อย"""
        if symbol not in cls.TRADING_SYMBOLS:
            raise ValueError(f"Symbol {symbol} not in trading list")
        
        # กำหนด pip value และ lot size ตาม symbol
        if 'JPY' in symbol:
            pip_value = 0.01
            decimal_places = 3
        else:
            pip_value = 0.0001
            decimal_places = 5
        
        # กำหนด currency block
        currency_block = None
        for block, symbols in cls.CORRELATION_CONFIG['CURRENCY_BLOCKS'].items():
            if symbol in symbols:
                currency_block = block
                break
        
        return {
            'symbol': symbol,
            'pip_value': pip_value,
            'decimal_places': decimal_places,
            'currency_block': currency_block,
            'is_major': symbol in cls.TRADING_SYMBOLS[:7],
            'has_usd': 'USD' in symbol,
            'is_cross': 'USD' not in symbol
        }

# Initialize configuration
if __name__ == "__main__":
    # Test configuration
    config = ForexRLConfig()
    config.create_directories()
    config.validate_config()
    
    print(f"📊 Trading Symbols: {len(config.TRADING_SYMBOLS)}")
    print(f"⏰ Timeframes: {list(config.TIMEFRAMES.keys())}")
    print(f"🎯 Max Risk per Trade: {config.RISK_CONFIG['MAX_RISK_PER_TRADE']*100}%")
    print(f"💾 Database: {config.DATABASE_CONFIG['TYPE']}")
    print(f"🤖 RL Algorithm: {config.RL_CONFIG['ALGORITHM']}")
    
