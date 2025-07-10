# data_collector.py - Automated Data Collection สำหรับ 26 คู่เงิน 5 timeframes
"""
ไฟล์นี้จัดการการดึงข้อมูลอัตโนมัติสำหรับทั้งระบบ
แก้ไขไฟล์นี้เมื่อ: ต้องการเปลี่ยน collection frequency หรือเพิ่ม data sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import schedule
import logging
from typing import Dict, List, Optional, Callable
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from dataclasses import dataclass
import queue

from config.config import ForexRLConfig
from config.mt5_connector import MT5Connector, create_mt5_connector

# Import database manager
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.database_manager import DatabaseManager, create_database_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CollectionJob:
    """Data collection job definition"""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    priority: int = 1  # 1 = high, 2 = medium, 3 = low

@dataclass
class CollectionResult:
    """Result of data collection"""
    symbol: str
    timeframe: str
    success: bool
    data: Optional[pd.DataFrame]
    error: Optional[str]
    collection_time: datetime
    rows_collected: int

class DataCollector:
    """Main Data Collection System"""
    
    def __init__(self, mt5_connector: Optional[MT5Connector] = None, database_manager: Optional[DatabaseManager] = None):
        self.config = ForexRLConfig()
        self.mt5 = mt5_connector or create_mt5_connector()
        self.db = database_manager or create_database_manager()
        
        # Collection state
        self.is_collecting = False
        self.collection_thread = None
        self.job_queue = queue.PriorityQueue()
        self.results_queue = queue.Queue()
        
        # Statistics
        self.collection_stats = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'last_collection': None,
            'errors': []
        }
        
        # Data cache
        self.data_cache = {}
        self.cache_expiry = {}
        
        # Thread pool for parallel collection
        self.max_workers = 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        self._setup_directories()
    
    def _setup_directories(self):
        """สร้าง directories สำหรับเก็บข้อมูล"""
        for directory in [
            self.config.DIRECTORIES['RAW_DATA_DIR'],
            self.config.DIRECTORIES['PROCESSED_DATA_DIR'],
            f"{self.config.DIRECTORIES['RAW_DATA_DIR']}/historical",
            f"{self.config.DIRECTORIES['RAW_DATA_DIR']}/realtime"
        ]:
            os.makedirs(directory, exist_ok=True)
    
    def collect_historical_data_all(self, 
                                   days_back: int = None,
                                   save_to_disk: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        """ดึงข้อมูล historical ทั้งหมด 26 คู่ 5 timeframes"""
        
        days_back = days_back or self.config.DATA_CONFIG['HISTORICAL_DAYS']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"🚀 Starting historical data collection for {len(self.config.TRADING_SYMBOLS)} symbols")
        logger.info(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Connect to MT5
        if not self.mt5.connect():
            logger.error("❌ Failed to connect to MT5")
            return {}
        
        all_data = {}
        jobs = []
        
        # Create collection jobs
        for symbol in self.config.TRADING_SYMBOLS:
            all_data[symbol] = {}
            for timeframe in self.config.TIMEFRAMES.keys():
                job = CollectionJob(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    priority=1 if symbol in self.config.TRADING_SYMBOLS[:8] else 2  # Major pairs first
                )
                jobs.append(job)
        
        # Execute jobs in parallel
        logger.info(f"📊 Collecting data for {len(jobs)} symbol-timeframe combinations...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._collect_single_dataset, job): job 
                for job in jobs
            }
            
            # Collect results
            completed = 0
            failed = 0
            
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                
                try:
                    result = future.result()
                    
                    if result.success and result.data is not None:
                        all_data[result.symbol][result.timeframe] = result.data
                        completed += 1
                        
                        # Save to database
                        if self.db:
                            db_success = self.db.save_price_data(result.data, result.symbol, result.timeframe)
                            if db_success:
                                logger.debug(f"💾 Saved {result.symbol} {result.timeframe} to database")
                            else:
                                logger.warning(f"⚠️ Database save failed for {result.symbol} {result.timeframe}")
                        
                        if save_to_disk:
                            self._save_data_to_disk(result.data, result.symbol, result.timeframe, 'historical')
                        
                        logger.info(f"✅ {result.symbol} {result.timeframe}: {result.rows_collected} rows")
                    else:
                        failed += 1
                        logger.error(f"❌ {result.symbol} {result.timeframe}: {result.error}")
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ {job.symbol} {job.timeframe}: {str(e)}")
                
                # Progress update
                total_progress = completed + failed
                progress_pct = (total_progress / len(jobs)) * 100
                print(f"\r🔄 Progress: {progress_pct:.1f}% ({completed} success, {failed} failed)", end='')
        
        print()  # New line after progress
        
        # Update statistics
        self.collection_stats.update({
            'total_jobs': len(jobs),
            'completed_jobs': completed,
            'failed_jobs': failed,
            'last_collection': datetime.now()
        })
        
        logger.info(f"📈 Collection completed: {completed}/{len(jobs)} successful")
        
        # Generate summary report
        self._generate_collection_report(all_data)
        
        return all_data
    
    def _collect_single_dataset(self, job: CollectionJob) -> CollectionResult:
        """ดึงข้อมูลชุดเดียว"""
        
        start_time = datetime.now()
        
        try:
            data = self.mt5.get_historical_data(
                symbol=job.symbol,
                timeframe=job.timeframe,
                start_date=job.start_date,
                end_date=job.end_date
            )
            
            if data is not None and len(data) > 0:
                # Validate data quality
                is_valid, issues = self.mt5.validate_data_quality(data, job.symbol)
                
                if not is_valid:
                    logger.warning(f"⚠️ Data quality issues for {job.symbol} {job.timeframe}: {issues}")
                
                return CollectionResult(
                    symbol=job.symbol,
                    timeframe=job.timeframe,
                    success=True,
                    data=data,
                    error=None,
                    collection_time=start_time,
                    rows_collected=len(data)
                )
            else:
                return CollectionResult(
                    symbol=job.symbol,
                    timeframe=job.timeframe,
                    success=False,
                    data=None,
                    error="No data returned",
                    collection_time=start_time,
                    rows_collected=0
                )
                
        except Exception as e:
            return CollectionResult(
                symbol=job.symbol,
                timeframe=job.timeframe,
                success=False,
                data=None,
                error=str(e),
                collection_time=start_time,
                rows_collected=0
            )
    
    def collect_realtime_data(self) -> Dict[str, Dict]:
        """ดึงข้อมูล real-time ทั้งหมด"""
        
        if not self.mt5.is_connected:
            logger.error("MT5 not connected for real-time data")
            return {}
        
        current_prices = {}
        
        try:
            # Get current prices for all symbols
            for symbol in self.config.TRADING_SYMBOLS:
                tick = self.mt5.get_current_tick(symbol)
                if tick:
                    current_prices[symbol] = {
                        'bid': tick['bid'],
                        'ask': tick['ask'],
                        'spread': tick['spread'],
                        'time': tick['time'],
                        'last': tick.get('last', tick['bid'])
                    }
            
            logger.info(f"📊 Real-time data collected for {len(current_prices)} symbols")
            return current_prices
            
        except Exception as e:
            logger.error(f"Error collecting real-time data: {e}")
            return {}
    
    def start_realtime_collection(self, interval_seconds: int = 5):
        """เริ่มการดึงข้อมูล real-time อัตโนมัติ"""
        
        if self.is_collecting:
            logger.warning("Real-time collection already running")
            return
        
        self.is_collecting = True
        
        def collection_loop():
            logger.info(f"🔄 Started real-time collection (interval: {interval_seconds}s)")
            
            while self.is_collecting:
                try:
                    if self.mt5.is_market_open():
                        data = self.collect_realtime_data()
                        
                        if data:
                            # Cache the data
                            self.data_cache['realtime'] = data
                            self.cache_expiry['realtime'] = datetime.now() + timedelta(seconds=interval_seconds)
                            
                            # Save to database
                            if self.db:
                                for symbol, tick_data in data.items():
                                    tick_data['symbol'] = symbol
                                    self.db.save_realtime_tick(tick_data)
                            
                            # Save to disk (optional)
                            self._save_realtime_tick(data)
                        
                    else:
                        logger.info("⏸️ Market closed, pausing real-time collection")
                        time.sleep(60)  # Check market status every minute
                        continue
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in real-time collection loop: {e}")
                    time.sleep(interval_seconds)
            
            logger.info("⏹️ Real-time collection stopped")
        
        self.collection_thread = threading.Thread(target=collection_loop, daemon=True)
        self.collection_thread.start()
    
    def stop_realtime_collection(self):
        """หยุดการดึงข้อมูล real-time"""
        
        if self.is_collecting:
            self.is_collecting = False
            if self.collection_thread:
                self.collection_thread.join(timeout=10)
            logger.info("⏹️ Real-time collection stopped")
    
    def _save_data_to_disk(self, 
                          data: pd.DataFrame, 
                          symbol: str, 
                          timeframe: str, 
                          data_type: str = 'historical'):
        """บันทึกข้อมูลลง disk"""
        
        try:
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timeframe}_{data_type}_{timestamp}.csv"
            filepath = os.path.join(
                self.config.DIRECTORIES['RAW_DATA_DIR'], 
                data_type, 
                filename
            )
            
            # Save to CSV only (avoid parquet issues)
            data.to_csv(filepath)
            
            logger.debug(f"💾 Saved {symbol} {timeframe} data to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving data to disk: {e}")
    
    def _save_realtime_tick(self, tick_data: Dict[str, Dict]):
        """บันทึก real-time tick data"""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"realtime_ticks_{timestamp}.json"
            filepath = os.path.join(
                self.config.DIRECTORIES['RAW_DATA_DIR'], 
                'realtime', 
                filename
            )
            
            # Add timestamp to data
            tick_data['collection_time'] = timestamp
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(tick_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error saving real-time tick: {e}")
    
    def _generate_collection_report(self, data: Dict[str, Dict[str, pd.DataFrame]]):
        """สร้างรายงานสรุปการดึงข้อมูล"""
        
        try:
            report = {
                'collection_time': datetime.now().isoformat(),
                'total_symbols': len(data),
                'total_timeframes': len(self.config.TIMEFRAMES),
                'symbols_summary': {},
                'timeframes_summary': {},
                'data_quality': {},
                'statistics': self.collection_stats
            }
            
            # Symbols summary
            for symbol, timeframes_data in data.items():
                report['symbols_summary'][symbol] = {
                    'timeframes_collected': len(timeframes_data),
                    'total_rows': sum(len(df) for df in timeframes_data.values()),
                    'success_rate': len(timeframes_data) / len(self.config.TIMEFRAMES)
                }
            
            # Timeframes summary
            for tf in self.config.TIMEFRAMES.keys():
                tf_data = [data[symbol].get(tf) for symbol in data.keys()]
                tf_data = [df for df in tf_data if df is not None]
                
                report['timeframes_summary'][tf] = {
                    'symbols_collected': len(tf_data),
                    'total_rows': sum(len(df) for df in tf_data),
                    'avg_rows_per_symbol': np.mean([len(df) for df in tf_data]) if tf_data else 0
                }
            
            # Save report
            report_path = os.path.join(
                self.config.DIRECTORIES['RAW_DATA_DIR'],
                f"collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Collection report saved to {report_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("📊 DATA COLLECTION SUMMARY")
            print("="*60)
            print(f"✅ Symbols: {report['total_symbols']}")
            print(f"⏰ Timeframes: {report['total_timeframes']}")
            print(f"📈 Success Rate: {(self.collection_stats['completed_jobs']/self.collection_stats['total_jobs'])*100:.1f}%")
            print(f"📁 Total Datasets: {self.collection_stats['completed_jobs']}")
            
            # Top performers
            successful_symbols = [
                symbol for symbol, info in report['symbols_summary'].items() 
                if info['success_rate'] == 1.0
            ]
            print(f"🏆 Perfect Collection: {len(successful_symbols)} symbols")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error generating collection report: {e}")
    
    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """ดึงข้อมูลจาก cache"""
        
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self.data_cache:
            # Check if cache is still valid
            if cache_key in self.cache_expiry:
                if datetime.now() < self.cache_expiry[cache_key]:
                    return self.data_cache[cache_key]
                else:
                    # Cache expired
                    del self.data_cache[cache_key]
                    del self.cache_expiry[cache_key]
        
        return None
    
    def get_collection_status(self) -> Dict:
        """สถานะการดึงข้อมูล"""
        
        return {
            'is_collecting': self.is_collecting,
            'mt5_connected': self.mt5.is_connected,
            'market_open': self.mt5.is_market_open(),
            'cache_size': len(self.data_cache),
            'statistics': self.collection_stats,
            'last_update': datetime.now()
        }
    
    def schedule_daily_collection(self, time_str: str = "02:00"):
        """กำหนดเวลาดึงข้อมูลอัตโนมัติรายวัน"""
        
        def daily_job():
            logger.info("🕐 Starting scheduled daily data collection")
            self.collect_historical_data_all(days_back=7)  # Last week's data
        
        schedule.every().day.at(time_str).do(daily_job)
        logger.info(f"⏰ Scheduled daily collection at {time_str}")
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        """ลบไฟล์เก่าที่เก็บเกิน X วัน"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for root, dirs, files in os.walk(self.config.DIRECTORIES['RAW_DATA_DIR']):
            for file in files:
                filepath = os.path.join(root, file)
                
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_time < cutoff_date:
                        os.remove(filepath)
                        logger.debug(f"🗑️ Removed old file: {file}")
                except Exception as e:
                    logger.error(f"Error removing file {file}: {e}")

def quick_data_collection_test():
    """ทดสอบระบบดึงข้อมูลแบบเร็ว"""
    
    print("🚀 Quick Data Collection Test")
    print("="*40)
    
    # Test with 3 major pairs and 2 timeframes
    test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    test_timeframes = ['M15', 'H1']
    
    collector = DataCollector()
    
    if not collector.mt5.connect():
        print("❌ Failed to connect to MT5")
        return False
    
    print(f"✅ Connected to MT5")
    print(f"📊 Testing {len(test_symbols)} symbols × {len(test_timeframes)} timeframes")
    
    # Test historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Last week
    
    success_count = 0
    total_tests = len(test_symbols) * len(test_timeframes)
    
    for symbol in test_symbols:
        for timeframe in test_timeframes:
            try:
                data = collector.mt5.get_historical_data(symbol, timeframe, start_date, end_date)
                
                if data is not None and len(data) > 0:
                    print(f"✅ {symbol} {timeframe}: {len(data)} bars")
                    success_count += 1
                else:
                    print(f"❌ {symbol} {timeframe}: No data")
                    
            except Exception as e:
                print(f"❌ {symbol} {timeframe}: {str(e)}")
    
    # Test real-time data
    print("\n🔴 Testing real-time data...")
    realtime_data = collector.collect_realtime_data()
    
    if realtime_data:
        print(f"✅ Real-time: {len(realtime_data)} symbols")
        for symbol in test_symbols:
            if symbol in realtime_data:
                tick = realtime_data[symbol]
                print(f"  {symbol}: {tick['bid']:.5f}/{tick['ask']:.5f}")
    else:
        print("❌ Real-time: No data")
    
    print(f"\n📈 Test Results: {success_count}/{total_tests} successful")
    
    collector.mt5.disconnect()
    return success_count == total_tests

if __name__ == "__main__":
    # Run quick test
    print("🧪 Running Data Collection Test...")
    success = quick_data_collection_test()
    
    if success:
        print("\n🎉 All tests passed! Ready for full data collection.")
        
        # Ask user if they want to run full collection
        response = input("\nRun full historical data collection? (y/N): ")
        if response.lower() == 'y':
            collector = DataCollector()
            collector.collect_historical_data_all(days_back=30)  # Last month
    else:
        print("\n⚠️ Some tests failed. Please check MT5 connection and data access.")