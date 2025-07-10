# main.py - Main System Runner
"""
ไฟล์หลักสำหรับรันระบบ Forex RL Trading
ใช้ไฟล์นี้เป็นจุดเริ่มต้นของทุกอย่าง
"""

import sys
import os
import time
from datetime import datetime, timedelta
import logging

# Add src directory to path  
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import our modules from config package
from config import ForexRLConfig, MT5Connector, create_mt5_connector, test_connection, DataCollector, quick_data_collection_test

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForexRLSystem:
    """Main Forex RL Trading System"""
    
    def __init__(self):
        self.config = ForexRLConfig()
        self.mt5_connector = None
        self.data_collector = None
        self.system_status = "Initialized"
        
        # Create necessary directories
        self.config.create_directories()
        logger.info("📁 Directories created")
    
    def initialize_system(self):
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Forex RL System...")
            
            # Validate configuration
            self.config.validate_config()
            logger.info("✅ Configuration validated")
            
            # Create MT5 connector
            self.mt5_connector = create_mt5_connector()
            logger.info("✅ MT5 connector created")
            
            # Create data collector
            self.data_collector = DataCollector(self.mt5_connector)
            logger.info("✅ Data collector created")
            
            self.system_status = "Initialized"
            logger.info("🎉 System initialization completed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            self.system_status = "Failed"
            return False
    
    def test_connections(self):
        """Test all system connections"""
        logger.info("🧪 Testing system connections...")
        
        print("\n" + "="*50)
        print("🔧 SYSTEM CONNECTION TESTS")
        print("="*50)
        
        # Test MT5 connection
        print("\n1️⃣ Testing MT5 Connection...")
        mt5_success = test_connection(['EURUSD', 'GBPUSD', 'USDJPY'])
        
        # Test data collection
        print("\n2️⃣ Testing Data Collection...")
        data_success = quick_data_collection_test()
        
        # Test market status
        print("\n3️⃣ Testing Market Status...")
        if self.mt5_connector and self.mt5_connector.connect():
            market_open = self.mt5_connector.is_market_open()
            print(f"📊 Market Status: {'🟢 OPEN' if market_open else '🔴 CLOSED'}")
            
            # Test symbols availability
            symbols_available = 0
            for symbol in self.config.TRADING_SYMBOLS[:5]:  # Test first 5
                tick = self.mt5_connector.get_current_tick(symbol)
                if tick:
                    symbols_available += 1
            
            print(f"📈 Symbols Available: {symbols_available}/5 test symbols")
            self.mt5_connector.disconnect()
        
        print("\n" + "="*50)
        
        if mt5_success and data_success:
            print("🎉 ALL TESTS PASSED! System ready for operation.")
            return True
        else:
            print("⚠️ Some tests failed. Please check configuration.")
            return False
    
    def collect_initial_data(self, days_back: int = 30):
        """Collect initial historical data"""
        logger.info(f"📊 Starting initial data collection ({days_back} days)...")
        
        if not self.data_collector:
            logger.error("Data collector not initialized")
            return False
        
        try:
            # Collect historical data
            data = self.data_collector.collect_historical_data_all(
                days_back=days_back,
                save_to_disk=True
            )
            
            if data:
                logger.info(f"✅ Initial data collection completed")
                
                # Print summary
                total_datasets = sum(len(timeframes) for timeframes in data.values())
                print(f"\n📊 DATA COLLECTION SUMMARY:")
                print(f"  • Symbols: {len(data)}")
                print(f"  • Datasets: {total_datasets}")
                print(f"  • Period: {days_back} days")
                
                return True
            else:
                logger.error("❌ No data collected")
                return False
                
        except Exception as e:
            logger.error(f"❌ Data collection failed: {e}")
            return False
    
    def start_realtime_monitoring(self):
        """Start real-time data monitoring"""
        logger.info("🔴 Starting real-time monitoring...")
        
        if not self.data_collector:
            logger.error("Data collector not initialized")
            return False
        
        try:
            self.data_collector.start_realtime_collection(interval_seconds=5)
            self.system_status = "Running"
            logger.info("✅ Real-time monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start real-time monitoring: {e}")
            return False
    
    def stop_realtime_monitoring(self):
        """Stop real-time data monitoring"""
        logger.info("⏹️ Stopping real-time monitoring...")
        
        if self.data_collector:
            self.data_collector.stop_realtime_collection()
            self.system_status = "Stopped"
            logger.info("✅ Real-time monitoring stopped")
    
    def get_system_status(self):
        """Get current system status"""
        status = {
            'system_status': self.system_status,
            'timestamp': datetime.now(),
            'components': {
                'config': 'OK' if self.config else 'Not loaded',
                'mt5_connector': 'OK' if self.mt5_connector else 'Not initialized',
                'data_collector': 'OK' if self.data_collector else 'Not initialized'
            }
        }
        
        if self.data_collector:
            status.update(self.data_collector.get_collection_status())
        
        return status
    
    def run_interactive_menu(self):
        """Run interactive menu for system control"""
        
        while True:
            print("\n" + "="*60)
            print("🤖 FOREX RL TRADING SYSTEM")
            print("="*60)
            print("Current Status:", self.system_status)
            print("-"*60)
            print("1️⃣  Initialize System")
            print("2️⃣  Test Connections") 
            print("3️⃣  Collect Initial Data (30 days)")
            print("4️⃣  Collect Initial Data (Custom days)")
            print("5️⃣  Start Real-time Monitoring")
            print("6️⃣  Stop Real-time Monitoring")
            print("7️⃣  System Status")
            print("8️⃣  Configuration Info")
            print("9️⃣  Exit")
            print("="*60)
            
            try:
                choice = input("Enter your choice (1-9): ").strip()
                
                if choice == '1':
                    self.initialize_system()
                
                elif choice == '2':
                    self.test_connections()
                
                elif choice == '3':
                    self.collect_initial_data(days_back=30)
                
                elif choice == '4':
                    days = int(input("Enter number of days (1-1095): "))
                    if 1 <= days <= 1095:
                        self.collect_initial_data(days_back=days)
                    else:
                        print("⚠️ Invalid number of days")
                
                elif choice == '5':
                    self.start_realtime_monitoring()
                
                elif choice == '6':
                    self.stop_realtime_monitoring()
                
                elif choice == '7':
                    status = self.get_system_status()
                    print("\n📊 SYSTEM STATUS:")
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                
                elif choice == '8':
                    print(f"\n⚙️ CONFIGURATION INFO:")
                    print(f"  Symbols: {len(self.config.TRADING_SYMBOLS)}")
                    print(f"  Timeframes: {list(self.config.TIMEFRAMES.keys())}")
                    print(f"  RL Algorithm: {self.config.RL_CONFIG['ALGORITHM']}")
                    print(f"  Max Risk/Trade: {self.config.RISK_CONFIG['MAX_RISK_PER_TRADE']*100}%")
                
                elif choice == '9':
                    print("👋 Exiting system...")
                    self.stop_realtime_monitoring()
                    if self.mt5_connector:
                        self.mt5_connector.disconnect()
                    break
                
                else:
                    print("⚠️ Invalid choice. Please try again.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting system...")
                self.stop_realtime_monitoring()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    print("🚀 Starting Forex RL Trading System...")
    
    # Create system instance
    system = ForexRLSystem()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            # Quick test mode
            print("🧪 Running in test mode...")
            system.initialize_system()
            system.test_connections()
        
        elif command == 'collect':
            # Data collection mode
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            print(f"📊 Running data collection mode ({days} days)...")
            system.initialize_system()
            system.collect_initial_data(days_back=days)
        
        elif command == 'monitor':
            # Real-time monitoring mode
            print("🔴 Running real-time monitoring mode...")
            system.initialize_system()
            system.start_realtime_monitoring()
            
            try:
                while True:
                    status = system.get_system_status()
                    print(f"📊 Status: {status['system_status']} | Time: {datetime.now().strftime('%H:%M:%S')}")
                    import time
                    time.sleep(30)
            except KeyboardInterrupt:
                system.stop_realtime_monitoring()
        
        else:
            print(f"⚠️ Unknown command: {command}")
            print("Available commands: test, collect [days], monitor")
    
    else:
        # Interactive mode
        system.run_interactive_menu()

if __name__ == "__main__":
    main()