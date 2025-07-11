# core/data_manager.py
"""
Multi-Currency Data Manager
จัดการข้อมูล 28 คู่เงิน x 4 timeframes พร้อม real-time updates
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import time
import logging
from dataclasses import dataclass
import MetaTrader5 as mt5

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
class DataManager:
    """
    Multi-Currency Data Manager
    - จัดการข้อมูล 28 คู่เงิน
    - 4 timeframes: M5, M15, H1, H4
    - Real-time data collection
    - SQLite storage with efficient indexing
    """
    
    def __init__(self, db_path: str = "data/correlation_data.db"):
        self.db_path = db_path
        self.connection = None
        
        # Create data directory if not exists
        import os
        data_dir = os.path.dirname(self.db_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f" Created directory: {data_dir}")
        
        # 21 Most common currency pairs (guaranteed availability)
        self.currency_pairs = [
            # Major pairs (7)
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
            # Cross pairs (14)
            "EURGBP", "EURJPY", "EURCHF", "EURCAD", "EURAUD", "EURNZD",
            "GBPJPY", "GBPCHF", "GBPCAD", "GBPAUD", "GBPNZD",
            "CHFJPY", "CADJPY", "AUDNZD"
        ]
        
        # Timeframes mapping
        self.timeframes = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4
        }
        
        # Data cache for fast access
        self.data_cache = {}
        self.last_update = {}
        
        # Threading control
        self.is_running = False
        self.update_thread = None
        self.lock = threading.Lock()
        
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize SQLite database with optimized schema"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.connection.cursor()
            
            # Create main data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            # Create indexes for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
                ON market_data(symbol, timeframe, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON market_data(timestamp)
            """)
            
            # Create metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_metadata (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    last_update INTEGER NOT NULL,
                    total_bars INTEGER DEFAULT 0,
                    PRIMARY KEY(symbol, timeframe)
                )
            """)
            
            self.connection.commit()
            logger.info(" Database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        try:
            if not mt5.initialize():
                logger.error("❌ MT5 initialization failed")
                return False
            
            # Check connection
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("❌ MT5 account info failed")
                return False
            
            logger.info(f" MT5 connected - Account: {account_info.login}")
            return True
            
        except Exception as e:
            logger.error(f"❌ MT5 connection error: {e}")
            return False
    
    def get_historical_data(self, symbol: str, timeframe: str, bars: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical data from MT5"""
        try:
            if timeframe not in self.timeframes:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            # Get data from MT5
            rates = mt5.copy_rates_from_pos(symbol, self.timeframes[timeframe], 0, bars)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df.drop('time', axis=1)
            df = df.rename(columns={'tick_volume': 'volume'})
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error getting historical data {symbol} {timeframe}: {e}")
            return None
    
    def store_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Store data to SQLite database"""
        try:
            with self.lock:
                cursor = self.connection.cursor()
                
                # Prepare data for insertion
                records = []
                for _, row in data.iterrows():
                    timestamp_unix = int(row['timestamp'].timestamp())
                    records.append((
                        symbol, timeframe, timestamp_unix,
                        row['open'], row['high'], row['low'], row['close'], row['volume']
                    ))
                
                # Insert with conflict resolution
                cursor.executemany("""
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                # Update metadata
                last_timestamp = int(data['timestamp'].iloc[-1].timestamp())
                cursor.execute("""
                    INSERT OR REPLACE INTO data_metadata 
                    (symbol, timeframe, last_update, total_bars)
                    VALUES (?, ?, ?, ?)
                """, (symbol, timeframe, last_timestamp, len(data)))
                
                self.connection.commit()
                
                # Update cache
                cache_key = f"{symbol}_{timeframe}"
                self.data_cache[cache_key] = data.copy()
                self.last_update[cache_key] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error storing data {symbol} {timeframe}: {e}")
    
    def get_data(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """Get data from cache or database"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            
            # Check cache first (if updated within last minute)
            if (cache_key in self.data_cache and 
                cache_key in self.last_update and
                (datetime.now() - self.last_update[cache_key]).seconds < 60):
                
                cached_data = self.data_cache[cache_key]
                return cached_data.tail(bars).copy()
            
            # Get from database
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute("""
                    SELECT timestamp, open, high, low, close, volume
                    FROM market_data 
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (symbol, timeframe, bars))
                
                rows = cursor.fetchall()
                
                if not rows:
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Update cache
                self.data_cache[cache_key] = df.copy()
                self.last_update[cache_key] = datetime.now()
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting data {symbol} {timeframe}: {e}")
            return None
    
    def update_single_pair(self, symbol: str, timeframe: str):
        """Update data for single currency pair"""
        try:
            # Get latest data from MT5
            new_data = self.get_historical_data(symbol, timeframe, bars=100)
            
            if new_data is not None and len(new_data) > 0:
                # Store to database
                self.store_data(symbol, timeframe, new_data)
                logger.debug(f" Updated {symbol} {timeframe}: {len(new_data)} bars")
            else:
                logger.warning(f"⚠️ No new data for {symbol} {timeframe}")
                
        except Exception as e:
            logger.error(f"❌ Update failed {symbol} {timeframe}: {e}")
    
    def update_all_data(self):
        """Update all currency pairs and timeframes"""
        logger.info(" Starting data update cycle...")
        
        update_count = 0
        error_count = 0
        
        for symbol in self.currency_pairs:
            for timeframe in self.timeframes.keys():
                try:
                    self.update_single_pair(symbol, timeframe)
                    update_count += 1
                    time.sleep(0.1)  # Prevent MT5 overload
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"❌ Failed to update {symbol} {timeframe}: {e}")
        
        logger.info(f" Data update completed: {update_count} updated, {error_count} errors")
    
    def start_real_time_updates(self, interval: int = 60):
        """Start real-time data updates in background thread"""
        if self.is_running:
            logger.warning("⚠️ Real-time updates already running")
            return
        
        logger.info(f" Starting real-time updates (every {interval} seconds)")
        self.is_running = True
        
        def update_loop():
            while self.is_running:
                try:
                    self.update_all_data()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"❌ Update loop error: {e}")
                    time.sleep(5)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def stop_real_time_updates(self):
        """Stop real-time data updates"""
        if self.is_running:
            logger.info("🛑 Stopping real-time updates...")
            self.is_running = False
            if self.update_thread:
                self.update_thread.join(timeout=5)
    
    def get_data_summary(self) -> Dict:
        """Get data summary statistics"""
        try:
            with self.lock:
                cursor = self.connection.cursor()
                
                # Get total records
                cursor.execute("SELECT COUNT(*) FROM market_data")
                total_records = cursor.fetchone()[0]
                
                # Get data by symbol
                cursor.execute("""
                    SELECT symbol, timeframe, COUNT(*) as bars, 
                           MAX(timestamp) as last_update
                    FROM market_data 
                    GROUP BY symbol, timeframe
                    ORDER BY symbol, timeframe
                """)
                
                symbol_data = cursor.fetchall()
                
                return {
                    'total_records': total_records,
                    'total_pairs': len(self.currency_pairs),
                    'total_timeframes': len(self.timeframes),
                    'symbol_data': symbol_data,
                    'cache_size': len(self.data_cache)
                }
                
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}
    
    def initialize_historical_data(self, days: int = 30):
        """Initialize historical data for all pairs"""
        logger.info(f" Initializing historical data ({days} days)...")
        
        if not self.connect_mt5():
            logger.error("❌ Cannot initialize without MT5 connection")
            return
        
        total_pairs = len(self.currency_pairs) * len(self.timeframes)
        completed = 0
        
        for symbol in self.currency_pairs:
            for timeframe in self.timeframes.keys():
                try:
                    # Calculate bars needed based on timeframe
                    if timeframe == "M5":
                        bars = days * 288  # 288 bars per day
                    elif timeframe == "M15":
                        bars = days * 96   # 96 bars per day
                    elif timeframe == "H1":
                        bars = days * 24   # 24 bars per day
                    else:  # H4
                        bars = days * 6    # 6 bars per day
                    
                    # Get historical data
                    data = self.get_historical_data(symbol, timeframe, bars)
                    
                    if data is not None and len(data) > 0:
                        self.store_data(symbol, timeframe, data)
                        completed += 1
                        logger.info(f" {symbol} {timeframe}: {len(data)} bars loaded ({completed}/{total_pairs})")
                    else:
                        logger.warning(f"⚠️ No data for {symbol} {timeframe}")
                    
                    time.sleep(0.1)  # Prevent MT5 overload
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {symbol} {timeframe}: {e}")
        
        logger.info(f" Historical data initialization completed: {completed}/{total_pairs}")
    
    def close(self):
        """Clean shutdown"""
        logger.info(" Shutting down Data Manager...")
        
        # Stop real-time updates
        self.stop_real_time_updates()
        
        # Close database connection
        if self.connection:
            self.connection.close()
        
        # Shutdown MT5
        mt5.shutdown()
        
        logger.info(" Data Manager shutdown completed")

# Example usage and testing
if __name__ == "__main__":
    print("🔧 Data Manager Test")
    print("=" * 50)
    
    # Create data manager
    dm = DataManager()
    
    # Test connection
    if dm.connect_mt5():
        print(" MT5 connection successful")
        
        # Initialize some historical data
        print("📊 Loading initial data...")
        dm.initialize_historical_data(days=7)  # 7 days for testing
        
        # Get data summary
        summary = dm.get_data_summary()
        print(f"\n📈 Data Summary:")
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Cache Size: {summary['cache_size']}")
        
        # Test data retrieval
        print("\n🔍 Testing data retrieval...")
        eurusd_h1 = dm.get_data("EURUSD", "H1", bars=10)
        if eurusd_h1 is not None:
            print(f"EURUSD H1 - Last 10 bars:")
            print(eurusd_h1.tail().to_string())
        
        # Start real-time updates for testing (uncomment to test)
        # print("\n Starting real-time updates...")
        # dm.start_real_time_updates(interval=30)
        # time.sleep(60)  # Run for 1 minute
        
    else:
        print("❌ MT5 connection failed")
    
    # Cleanup
    dm.close()
    print("\n✅ Test completed")