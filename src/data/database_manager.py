# src/data/database_manager.py - Database Management System
"""
ไฟล์นี้จัดการ database สำหรับเก็บข้อมูล forex และ RL training data
แก้ไขไฟล์นี้เมื่อ: เปลี่ยน database schema หรือ optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import threading
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, DateTime, Boolean, Index, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import insert
import psycopg2
from psycopg2.extras import execute_values
import warnings
warnings.filterwarnings('ignore')

# TimescaleDB support (optional - only needed for PostgreSQL with TimescaleDB extension)
TIMESCALE_AVAILABLE = False
# try:
#     import timescaledb
#     TIMESCALE_AVAILABLE = True
# except ImportError:
#     TIMESCALE_AVAILABLE = False
#     # TimescaleDB not available, will use standard PostgreSQL or SQLite

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class PriceData(Base):
    """Price data table schema"""
    __tablename__ = 'price_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(5), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    tick_volume = Column(Integer, nullable=True)
    spread = Column(Float, nullable=True)
    real_volume = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
        Index('idx_timestamp_symbol', 'timestamp', 'symbol'),
    )

class RealTimeTicks(Base):
    """Real-time tick data table"""
    __tablename__ = 'realtime_ticks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    bid = Column(Float, nullable=False)
    ask = Column(Float, nullable=False)
    last = Column(Float, nullable=True)
    volume = Column(Integer, nullable=True)
    spread = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

class FeatureData(Base):
    """Processed feature data table"""
    __tablename__ = 'feature_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(5), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    feature_name = Column(String(50), nullable=False)
    feature_value = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

class DatabaseManager:
    """Main Database Management Class"""
    
    def __init__(self, config=None):
        from config.config import ForexRLConfig
        self.config = config or ForexRLConfig()
        
        self.engine = None
        self.session_factory = None
        self.connection_string = None
        self._lock = threading.Lock()
        
        # Database type
        self.db_type = self.config.DATABASE_CONFIG.get('TYPE', 'sqlite')
        
        self._setup_database()
    
    def _setup_database(self):
        """Setup database connection และ tables"""
        try:
            if self.db_type.lower() == 'sqlite':
                self._setup_sqlite()
            elif self.db_type.lower() in ['postgresql', 'postgres', 'timescaledb']:
                self._setup_postgresql()
            else:
                logger.warning(f"Unknown database type: {self.db_type}, defaulting to SQLite")
                self._setup_sqlite()
            
            # Create tables
            self._create_tables()
            
            logger.info(f"✅ Database setup completed ({self.db_type})")
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            # Fallback to SQLite
            if self.db_type != 'sqlite':
                logger.info("🔄 Falling back to SQLite...")
                self._setup_sqlite()
                self._create_tables()
    
    def _setup_sqlite(self):
        """Setup SQLite database"""
        db_path = os.path.join(self.config.DIRECTORIES['DATA_DIR'], 'forex_data.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.connection_string = f"sqlite:///{db_path}"
        self.engine = create_engine(
            self.connection_string,
            echo=False,
            pool_pre_ping=True,
            connect_args={'check_same_thread': False}
        )
        self.session_factory = sessionmaker(bind=self.engine)
        
        logger.info(f"📁 SQLite database: {db_path}")
    
    def _setup_postgresql(self):
        """Setup PostgreSQL/TimescaleDB database"""
        db_config = self.config.DATABASE_CONFIG
        
        self.connection_string = (
            f"postgresql://{db_config['USER']}:{db_config['PASSWORD']}"
            f"@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"
        )
        
        try:
            self.engine = create_engine(
                self.connection_string,
                echo=False,
                pool_size=db_config.get('CONNECTION_POOL_SIZE', 20),
                max_overflow=db_config.get('MAX_CONNECTIONS', 100),
                pool_pre_ping=True
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.session_factory = sessionmaker(bind=self.engine)
            
            logger.info(f"🐘 PostgreSQL connected: {db_config['HOST']}:{db_config['PORT']}")
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            raise
    
    def _create_tables(self):
        """สร้าง tables ทั้งหมด"""
        try:
            # Create all tables
            Base.metadata.create_all(self.engine)
            
            # Setup TimescaleDB hypertables if available
            if self.db_type.lower() == 'timescaledb':
                self._setup_timescale_hypertables()
            
            logger.info("📊 Database tables created")
            
        except Exception as e:
            logger.error(f"❌ Table creation failed: {e}")
    
    def _setup_timescale_hypertables(self):
        """Setup TimescaleDB hypertables (only if TimescaleDB is available)"""
        if not TIMESCALE_AVAILABLE:
            logger.info("TimescaleDB not available, using standard PostgreSQL tables")
            return
            
        try:
            with self.engine.connect() as conn:
                # Check if TimescaleDB extension is available
                result = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'timescaledb'"))
                if not result.fetchone():
                    logger.warning("TimescaleDB extension not installed, using standard tables")
                    return
                
                # Create hypertable for price_data
                conn.execute(text("""
                    SELECT create_hypertable('price_data', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE);
                """))
                
                # Create hypertable for realtime_ticks
                conn.execute(text("""
                    SELECT create_hypertable('realtime_ticks', 'timestamp',
                    chunk_time_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE);
                """))
                
                # Create compression policy
                conn.execute(text("""
                    SELECT add_compression_policy('price_data', INTERVAL '7 days');
                """))
                
                conn.commit()
                
            logger.info("🕐 TimescaleDB hypertables configured")
            
        except Exception as e:
            logger.warning(f"TimescaleDB setup warning (will use standard tables): {e}")
    
    def save_price_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """บันทึกข้อมูล price data"""
        try:
            with self._lock:
                # Prepare data
                data_to_save = data.copy()
                data_to_save['symbol'] = symbol
                data_to_save['timeframe'] = timeframe
                data_to_save['timestamp'] = data_to_save.index
                
                # Select relevant columns
                columns = ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close']
                
                # Add optional columns if they exist
                if 'tick_volume' in data_to_save.columns:
                    columns.append('tick_volume')
                if 'spread' in data_to_save.columns:
                    columns.append('spread')
                if 'real_volume' in data_to_save.columns:
                    columns.append('real_volume')
                
                data_to_save = data_to_save[columns].copy()
                
                # Save to database
                data_to_save.to_sql(
                    'price_data',
                    self.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                logger.debug(f"💾 Saved {len(data_to_save)} rows for {symbol} {timeframe}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving price data: {e}")
            return False
    
    def save_realtime_tick(self, tick_data: Dict[str, Any]) -> bool:
        """บันทึก real-time tick data"""
        try:
            with self._lock:
                session = self.session_factory()
                
                tick = RealTimeTicks(
                    symbol=tick_data['symbol'],
                    timestamp=tick_data['time'],
                    bid=tick_data['bid'],
                    ask=tick_data['ask'],
                    last=tick_data.get('last'),
                    volume=tick_data.get('volume'),
                    spread=tick_data['spread']
                )
                
                session.add(tick)
                session.commit()
                session.close()
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving tick data: {e}")
            return False
    
    def get_price_data(self, 
                      symbol: str, 
                      timeframe: str, 
                      start_date: datetime = None, 
                      end_date: datetime = None,
                      limit: int = None) -> Optional[pd.DataFrame]:
        """ดึงข้อมูล price data"""
        try:
            # Build query
            query = """
                SELECT timestamp, open, high, low, close, tick_volume, spread, real_volume
                FROM price_data 
                WHERE symbol = :symbol AND timeframe = :timeframe
            """
            
            params = {'symbol': symbol, 'timeframe': timeframe}
            
            if start_date:
                query += " AND timestamp >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND timestamp <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY timestamp"
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Execute query
            df = pd.read_sql_query(query, self.engine, params=params)
            
            if len(df) > 0:
                df.set_index('timestamp', inplace=True)
                df.index = pd.to_datetime(df.index)
                
                logger.debug(f"📊 Retrieved {len(df)} rows for {symbol} {timeframe}")
                return df
            else:
                logger.warning(f"⚠️ No data found for {symbol} {timeframe}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving price data: {e}")
            return None
    
    def get_multiple_symbols_data(self, 
                                 symbols: List[str], 
                                 timeframe: str,
                                 start_date: datetime = None,
                                 end_date: datetime = None) -> Dict[str, pd.DataFrame]:
        """ดึงข้อมูลหลาย symbols พร้อมกัน"""
        
        results = {}
        
        for symbol in symbols:
            data = self.get_price_data(symbol, timeframe, start_date, end_date)
            if data is not None:
                results[symbol] = data
        
        logger.info(f"📊 Retrieved data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """หา timestamp ล่าสุดในฐานข้อมูล"""
        try:
            query = """
                SELECT MAX(timestamp) as latest_timestamp
                FROM price_data 
                WHERE symbol = :symbol AND timeframe = :timeframe
            """
            
            result = pd.read_sql_query(
                query, 
                self.engine, 
                params={'symbol': symbol, 'timeframe': timeframe}
            )
            
            if len(result) > 0 and result['latest_timestamp'].iloc[0]:
                return pd.to_datetime(result['latest_timestamp'].iloc[0])
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting latest timestamp: {e}")
            return None
    
    def get_data_summary(self) -> Dict[str, Any]:
        """สรุปข้อมูลในฐานข้อมูล"""
        try:
            summary = {
                'total_records': 0,
                'symbols': [],
                'timeframes': [],
                'date_range': {},
                'data_quality': {}
            }
            
            # Total records
            total_query = "SELECT COUNT(*) as total FROM price_data"
            total_result = pd.read_sql_query(total_query, self.engine)
            summary['total_records'] = int(total_result['total'].iloc[0])
            
            # Symbols and timeframes
            symbols_query = "SELECT DISTINCT symbol FROM price_data ORDER BY symbol"
            symbols_result = pd.read_sql_query(symbols_query, self.engine)
            summary['symbols'] = symbols_result['symbol'].tolist()
            
            timeframes_query = "SELECT DISTINCT timeframe FROM price_data ORDER BY timeframe"
            timeframes_result = pd.read_sql_query(timeframes_query, self.engine)
            summary['timeframes'] = timeframes_result['timeframe'].tolist()
            
            # Date range
            date_query = """
                SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date 
                FROM price_data
            """
            date_result = pd.read_sql_query(date_query, self.engine)
            if len(date_result) > 0:
                summary['date_range'] = {
                    'start': date_result['min_date'].iloc[0],
                    'end': date_result['max_date'].iloc[0]
                }
            
            # Data quality per symbol-timeframe
            quality_query = """
                SELECT symbol, timeframe, COUNT(*) as record_count,
                       MIN(timestamp) as start_date, MAX(timestamp) as end_date
                FROM price_data 
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """
            quality_result = pd.read_sql_query(quality_query, self.engine)
            
            for _, row in quality_result.iterrows():
                key = f"{row['symbol']}_{row['timeframe']}"
                summary['data_quality'][key] = {
                    'records': int(row['record_count']),
                    'start_date': row['start_date'],
                    'end_date': row['end_date']
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting data summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """ลบข้อมูลเก่าที่เกิน X วัน"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self.engine.connect() as conn:
                # Delete old price data
                result = conn.execute(
                    text("DELETE FROM price_data WHERE timestamp < :cutoff_date"),
                    {'cutoff_date': cutoff_date}
                )
                
                # Delete old tick data
                tick_result = conn.execute(
                    text("DELETE FROM realtime_ticks WHERE timestamp < :cutoff_date"),
                    {'cutoff_date': cutoff_date}
                )
                
                conn.commit()
                
                logger.info(f"🗑️ Cleanup completed: {result.rowcount} price records, {tick_result.rowcount} tick records deleted")
                
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def export_data(self, 
                   symbol: str, 
                   timeframe: str, 
                   start_date: datetime = None,
                   end_date: datetime = None,
                   format: str = 'csv') -> str:
        """Export ข้อมูลเป็นไฟล์"""
        try:
            data = self.get_price_data(symbol, timeframe, start_date, end_date)
            
            if data is None or len(data) == 0:
                logger.warning("No data to export")
                return None
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timeframe}_{timestamp}.{format}"
            filepath = os.path.join(self.config.DIRECTORIES['PROCESSED_DATA_DIR'], filename)
            
            # Export based on format
            if format.lower() == 'csv':
                data.to_csv(filepath)
            elif format.lower() == 'parquet':
                data.to_parquet(filepath)
            elif format.lower() == 'json':
                data.to_json(filepath, orient='index', date_format='iso')
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
            
            logger.info(f"📤 Data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return None
    
    def get_connection_info(self) -> Dict[str, Any]:
        """ข้อมูลการเชื่อมต่อฐานข้อมูล"""
        return {
            'database_type': self.db_type,
            'connection_string': self.connection_string.replace(self.config.DATABASE_CONFIG.get('PASSWORD', ''), '***'),
            'engine_info': str(self.engine.url) if self.engine else 'Not connected',
            'tables_created': True if self.engine else False
        }
    
    def close(self):
        """ปิดการเชื่อมต่อฐานข้อมูล"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("🔌 Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")

# Utility functions
def create_database_manager(config=None) -> DatabaseManager:
    """สร้าง DatabaseManager instance"""
    return DatabaseManager(config)

def test_database_connection() -> bool:
    """ทดสอบการเชื่อมต่อฐานข้อมูล"""
    try:
        print("🧪 Testing database connection...")
        
        db = create_database_manager()
        
        # Test connection
        info = db.get_connection_info()
        print(f"✅ Database type: {info['database_type']}")
        print(f"✅ Connection: {info['engine_info']}")
        
        # Test basic operations
        summary = db.get_data_summary()
        print(f"📊 Total records: {summary.get('total_records', 0)}")
        print(f"📊 Symbols: {len(summary.get('symbols', []))}")
        print(f"📊 Timeframes: {summary.get('timeframes', [])}")
        
        db.close()
        
        print("🎉 Database test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    # Run database test
    test_database_connection()