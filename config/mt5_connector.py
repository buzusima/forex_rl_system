# mt5_connector.py - MT5 Connection และ Data Retrieval
"""
ไฟล์นี้จัดการการเชื่อมต่อกับ MetaTrader 5 และดึงข้อมูลราคา
แก้ไขไฟล์นี้เมื่อ: มีปัญหา connection, เปลี่ยนโบรกเกอร์, หรือ MT5 update
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import threading
from dataclasses import dataclass

from config.config import ForexRLConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MT5ConnectionConfig:
    """MT5 Connection Configuration"""
    login: int = 0              # ใส่ account number
    password: str = ""          # ใส่ password  
    server: str = ""            # ใส่ server name
    path: str = ""              # ใส่ MT5 installation path
    timeout: int = 60000        # Connection timeout (ms)
    
class MT5Connector:
    """MT5 Connection และ Data Management Class"""
    
    def __init__(self, connection_config: Optional[MT5ConnectionConfig] = None):
        self.config = ForexRLConfig()
        self.connection_config = connection_config
        self.is_connected = False
        self.symbols_info = {}
        self.last_tick_time = {}
        self.connection_lock = threading.Lock()
        
        # Initialize connection
        if connection_config:
            self.connect()
    
    def connect(self) -> bool:
        """เชื่อมต่อกับ MT5 - Auto detect existing login"""
        try:
            with self.connection_lock:
                # Initialize MT5
                if not mt5.initialize():
                    logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False
                
                # ตรวจสอบ account ที่ login อยู่แล้ว
                account_info = mt5.account_info()
                
                if account_info is None:
                    # ไม่มี account login อยู่ - ลอง login ถ้ามี config
                    if self.connection_config and self.connection_config.login:
                        logger.info("No active account found, attempting login...")
                        authorized = mt5.login(
                            login=self.connection_config.login,
                            password=self.connection_config.password,
                            server=self.connection_config.server
                        )
                        
                        if not authorized:
                            logger.error(f"MT5 login failed: {mt5.last_error()}")
                            return False
                        
                        account_info = mt5.account_info()
                        logger.info(f"✅ Logged in to MT5 account: {account_info.login}")
                    else:
                        logger.error("❌ No active MT5 account found and no login config provided")
                        logger.info("Please login to MT5 manually or provide login credentials")
                        return False
                else:
                    # มี account login อยู่แล้ว
                    logger.info(f"✅ Found active MT5 account: {account_info.login}")
                    logger.info(f"   Server: {account_info.server}")
                    logger.info(f"   Balance: ${account_info.balance:,.2f}")
                    logger.info(f"   Equity: ${account_info.equity:,.2f}")
                    logger.info(f"   Currency: {account_info.currency}")
                
                self.is_connected = True
                self._load_symbols_info()
                return True
                
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """ตัดการเชื่อมต่อ MT5"""
        try:
            with self.connection_lock:
                if self.is_connected:
                    mt5.shutdown()
                    self.is_connected = False
                    logger.info("MT5 disconnected")
        except Exception as e:
            logger.error(f"MT5 disconnect error: {e}")
    
    def _load_symbols_info(self):
        """โหลดข้อมูล symbols ทั้งหมด"""
        try:
            for symbol in self.config.TRADING_SYMBOLS:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.warning(f"Symbol {symbol} not found")
                    continue
                
                # Enable symbol ใน Market Watch
                if not mt5.symbol_select(symbol, True):
                    logger.warning(f"Failed to select symbol {symbol}")
                    continue
                
                self.symbols_info[symbol] = {
                    'point': symbol_info.point,
                    'digits': symbol_info.digits,
                    'spread': symbol_info.spread,
                    'trade_allowed': symbol_info.trade_mode != 0,
                    'min_lot': symbol_info.volume_min,
                    'max_lot': symbol_info.volume_max,
                    'lot_step': symbol_info.volume_step,
                    'tick_size': symbol_info.trade_tick_size,
                    'tick_value': symbol_info.trade_tick_value
                }
            
            logger.info(f"Loaded info for {len(self.symbols_info)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading symbols info: {e}")
    
    def get_historical_data(self, 
                          symbol: str, 
                          timeframe: str, 
                          start_date: datetime, 
                          end_date: datetime) -> Optional[pd.DataFrame]:
        """ดึงข้อมูล historical data"""
        
        if not self.is_connected:
            logger.error("MT5 not connected")
            return None
        
        if symbol not in self.config.TRADING_SYMBOLS:
            logger.error(f"Symbol {symbol} not in trading list")
            return None
        
        if timeframe not in self.config.MT5_TIMEFRAMES:
            logger.error(f"Timeframe {timeframe} not supported")
            return None
        
        try:
            # Get MT5 timeframe constant
            mt5_timeframe = self.config.MT5_TIMEFRAMES[timeframe]
            
            # Request data from MT5
            rates = mt5.copy_rates_range(
                symbol, 
                mt5_timeframe, 
                start_date, 
                end_date
            )
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Add symbol and timeframe info
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # Add spread info if available
            if symbol in self.symbols_info:
                df['spread'] = self.symbols_info[symbol]['spread']
                df['point'] = self.symbols_info[symbol]['point']
            
            # Sort by time
            df.sort_index(inplace=True)
            
            logger.info(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def get_multiple_symbols_data(self, 
                                 symbols: List[str], 
                                 timeframe: str, 
                                 start_date: datetime, 
                                 end_date: datetime) -> Dict[str, pd.DataFrame]:
        """ดึงข้อมูลหลาย symbols พร้อมกัน"""
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"Getting data for {symbol}...")
            data = self.get_historical_data(symbol, timeframe, start_date, end_date)
            
            if data is not None:
                results[symbol] = data
            else:
                logger.warning(f"Failed to get data for {symbol}")
            
            # Small delay to avoid overloading
            time.sleep(0.1)
        
        logger.info(f"Retrieved data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_all_timeframes_data(self, 
                               symbol: str, 
                               start_date: datetime, 
                               end_date: datetime) -> Dict[str, pd.DataFrame]:
        """ดึงข้อมูลทุก timeframes สำหรับ symbol หนึ่ง"""
        
        results = {}
        
        for timeframe in self.config.TIMEFRAMES.keys():
            logger.info(f"Getting {symbol} {timeframe} data...")
            data = self.get_historical_data(symbol, timeframe, start_date, end_date)
            
            if data is not None:
                results[timeframe] = data
            else:
                logger.warning(f"Failed to get {symbol} {timeframe} data")
            
            time.sleep(0.1)
        
        return results
    
    def get_current_tick(self, symbol: str) -> Optional[Dict]:
        """ดึงข้อมูล tick ปัจจุบัน"""
        
        if not self.is_connected:
            return None
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            tick_data = {
                'symbol': symbol,
                'time': datetime.fromtimestamp(tick.time),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'spread': tick.ask - tick.bid,
            }
            
            # Update last tick time
            self.last_tick_time[symbol] = tick_data['time']
            
            return tick_data
            
        except Exception as e:
            logger.error(f"Error getting current tick for {symbol}: {e}")
            return None
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """ดึงราคาปัจจุบันของหลาย symbols"""
        
        prices = {}
        
        for symbol in symbols:
            tick = self.get_current_tick(symbol)
            if tick:
                prices[symbol] = tick
        
        return prices
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """ตรวจสอบคุณภาพข้อมูล"""
        
        issues = []
        
        try:
            # ตรวจสอบ missing data
            if df.isnull().any().any():
                null_cols = df.columns[df.isnull().any()].tolist()
                issues.append(f"Missing data in columns: {null_cols}")
            
            # ตรวจสอบ duplicate timestamps
            if df.index.duplicated().any():
                issues.append("Duplicate timestamps found")
            
            # ตรวจสอบ price consistency
            if (df['high'] < df['low']).any():
                issues.append("High < Low inconsistency")
            
            if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
                issues.append("High price inconsistency")
            
            if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
                issues.append("Low price inconsistency")
            
            # ตรวจสอบ outliers (price changes > 10%)
            price_changes = df['close'].pct_change().abs()
            if (price_changes > 0.1).any():
                outlier_count = (price_changes > 0.1).sum()
                issues.append(f"Found {outlier_count} potential outliers (>10% price change)")
            
            # ตรวจสอบ zero volume
            if 'tick_volume' in df.columns:
                if (df['tick_volume'] == 0).any():
                    zero_vol_count = (df['tick_volume'] == 0).sum()
                    issues.append(f"Found {zero_vol_count} bars with zero volume")
            
            # ตรวจสอบ spread
            if 'spread' in df.columns:
                if symbol in self.symbols_info:
                    normal_spread = self.symbols_info[symbol]['spread']
                    wide_spreads = (df['spread'] > normal_spread * 5).sum()
                    if wide_spreads > 0:
                        issues.append(f"Found {wide_spreads} bars with abnormally wide spreads")
            
            is_valid = len(issues) == 0
            
            if is_valid:
                logger.info(f"✅ Data quality check passed for {symbol}")
            else:
                logger.warning(f"⚠️ Data quality issues for {symbol}: {issues}")
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return False, [f"Validation error: {e}"]
    
    def get_market_hours_info(self) -> Dict[str, Dict]:
        """ข้อมูลเวลาการเปิด-ปิดตลาด"""
        
        return {
            'ASIAN': {
                'start': '22:00',  # UTC
                'end': '07:00',    # UTC
                'major_pairs': ['USDJPY', 'AUDUSD', 'NZDUSD']
            },
            'EUROPEAN': {
                'start': '07:00',  # UTC
                'end': '16:00',    # UTC  
                'major_pairs': ['EURUSD', 'GBPUSD', 'USDCHF']
            },
            'AMERICAN': {
                'start': '13:00',  # UTC
                'end': '22:00',    # UTC
                'major_pairs': ['EURUSD', 'GBPUSD', 'USDCAD']
            }
        }
    
    def is_market_open(self, symbol: str = None) -> bool:
        """ตรวจสอบว่าตลาดเปิดอยู่หรือไม่"""
        
        try:
            current_time = datetime.now()
            
            # วันเสาร์-อาทิตย์ ตลาดปิด
            if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # วันศุกร์หลัง 22:00 UTC ตลาดปิด
            if current_time.weekday() == 4 and current_time.hour >= 22:
                return False
            
            # วันจันทร์ก่อน 22:00 UTC ตลาดยังไม่เปิด
            if current_time.weekday() == 0 and current_time.hour < 22:
                return False
            
            # ตรวจสอบ symbol specific
            if symbol and self.is_connected:
                tick = self.get_current_tick(symbol)
                if tick:
                    # ถ้าได้ tick data แสดงว่าตลาดเปิด
                    return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return True  # Default to open if can't determine
    
    def get_connection_status(self) -> Dict[str, Any]:
        """สถานะการเชื่อมต่อพร้อมข้อมูลละเอียด"""
        
        status = {
            'connected': self.is_connected,
            'symbols_loaded': len(self.symbols_info),
            'last_update': datetime.now(),
            'market_open': self.is_market_open(),
            'mt5_initialized': False,
            'account_info': None,
            'terminal_info': None
        }
        
        if self.is_connected:
            try:
                # ตรวจสอบว่า MT5 ยัง initialize อยู่หรือไม่
                account_info = mt5.account_info()
                if account_info:
                    status['mt5_initialized'] = True
                    status['account_info'] = {
                        'login': account_info.login,
                        'server': account_info.server,
                        'name': account_info.name,
                        'company': account_info.company,
                        'currency': account_info.currency,
                        'balance': account_info.balance,
                        'equity': account_info.equity,
                        'margin': account_info.margin,
                        'margin_free': account_info.margin_free,
                        'margin_level': account_info.margin_level,
                        'leverage': account_info.leverage,
                        'trade_allowed': account_info.trade_allowed,
                        'trade_expert': account_info.trade_expert
                    }
                
                # ข้อมูล terminal
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    status['terminal_info'] = {
                        'community_account': terminal_info.community_account,
                        'community_connection': terminal_info.community_connection,
                        'connected': terminal_info.connected,
                        'dlls_allowed': terminal_info.dlls_allowed,
                        'trade_allowed': terminal_info.trade_allowed,
                        'tradeapi_disabled': terminal_info.tradeapi_disabled,
                        'email_enabled': terminal_info.email_enabled,
                        'ftp_enabled': terminal_info.ftp_enabled,
                        'notifications_enabled': terminal_info.notifications_enabled,
                        'company': terminal_info.company,
                        'name': terminal_info.name,
                        'path': terminal_info.path
                    }
                
            except Exception as e:
                status['connection_error'] = str(e)
                status['connected'] = False
        
        return status

# Utility functions
def create_mt5_connector(login: int = None, 
                        password: str = None, 
                        server: str = None) -> MT5Connector:
    """สร้าง MT5Connector - Auto detect existing login if no credentials provided"""
    
    if login and password and server:
        # ใช้ credentials ที่ระบุ
        config = MT5ConnectionConfig(
            login=login,
            password=password, 
            server=server
        )
        logger.info(f"Creating MT5 connector with provided credentials (Account: {login})")
        return MT5Connector(config)
    else:
        # Auto detect existing login
        logger.info("Creating MT5 connector with auto-detection of existing login")
        return MT5Connector()

def test_connection(symbols: List[str] = None) -> bool:
    """ทดสอบการเชื่อมต่อ MT5 - Auto detect existing login"""
    
    print("🔄 Testing MT5 Connection...")
    
    connector = create_mt5_connector()
    
    if not connector.connect():
        print("❌ Failed to connect to MT5")
        print("💡 Make sure:")
        print("   1. MT5 is running")
        print("   2. You are logged into an account")
        print("   3. Account has access to forex symbols")
        return False
    
    print("✅ Connected to MT5")
    
    # แสดงข้อมูล account
    try:
        account_info = mt5.account_info()
        if account_info:
            print(f"📊 Account: {account_info.login}")
            print(f"🏦 Server: {account_info.server}")
            print(f"💰 Balance: ${account_info.balance:,.2f}")
            print(f"📈 Equity: ${account_info.equity:,.2f}")
    except:
        pass
    
    # Test symbols
    test_symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY']
    print(f"\n🧪 Testing {len(test_symbols)} symbols...")
    
    for symbol in test_symbols:
        tick = connector.get_current_tick(symbol)
        if tick:
            spread = tick['ask'] - tick['bid']
            print(f"✅ {symbol}: {tick['bid']:.5f}/{tick['ask']:.5f} (spread: {spread*10000:.1f} pips)")
        else:
            print(f"❌ {symbol}: No data available")
    
    # Test historical data
    print(f"\n📊 Testing historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    data = connector.get_historical_data('EURUSD', 'M15', start_date, end_date)
    if data is not None and len(data) > 0:
        print(f"✅ Historical data: {len(data)} bars retrieved")
        print(f"   Last price: {data['close'].iloc[-1]:.5f}")
        print(f"   Data range: {data.index[0]} to {data.index[-1]}")
    else:
        print("❌ Failed to get historical data")
    
    connector.disconnect()
    
    print("\n🎉 Connection test completed!")
    return True

if __name__ == "__main__":
    # Test the MT5 connection
    print("🔄 Testing MT5 Connection...")
    test_connection()