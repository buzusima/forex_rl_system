import MetaTrader5 as mt5
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Callable
import threading
import os

class LoginManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        
        # Connection state
        self.is_connected = False
        self.is_logged_in = False
        self.last_login_check = None
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        
        # Account information
        self.account_info = None
        self.login_number = None
        self.broker_name = None
        self.account_balance = 0.0
        self.account_equity = 0.0
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.check_interval = 5  # seconds
        
        # Callbacks
        self.on_login_callback = None
        self.on_logout_callback = None
        self.on_connection_lost_callback = None
        self.on_connection_restored_callback = None
        
        # Logging
        self.setup_logging()
        
        # Initialize MT5
        self.initialize_mt5()
    
    def load_config(self) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file {self.config_file} not found")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            return {}
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('monitoring', {}).get('log_level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('login_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('LoginManager')
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                self.logger.error("Failed to initialize MT5")
                return False
            
            self.is_connected = True
            self.logger.info("MT5 initialized successfully")
            
            # Get initial account info
            self.update_account_info()
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {e}")
            self.is_connected = False
            return False
    
    def update_account_info(self) -> bool:
        """Update account information"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                self.is_logged_in = False
                self.account_info = None
                return False
            
            # Store account details
            self.account_info = account_info
            self.login_number = account_info.login
            self.broker_name = account_info.company
            self.account_balance = account_info.balance
            self.account_equity = account_info.equity
            self.is_logged_in = True
            self.last_login_check = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            self.is_logged_in = False
            return False
    
    def check_connection_status(self) -> Dict[str, any]:
        """Check current connection and login status"""
        status = {
            'mt5_initialized': False,
            'logged_in': False,
            'account_info': None,
            'terminal_info': None,
            'symbols_available': False,
            'last_check': datetime.now()
        }
        
        try:
            # Check if MT5 is initialized
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return status
            
            status['mt5_initialized'] = True
            status['terminal_info'] = {
                'company': terminal_info.company,
                'path': terminal_info.path,
                'build': terminal_info.build,
                'connected': terminal_info.connected
            }
            
            # Check account login
            if self.update_account_info():
                status['logged_in'] = True
                status['account_info'] = {
                    'login': self.login_number,
                    'company': self.broker_name,
                    'balance': self.account_balance,
                    'equity': self.account_equity,
                    'currency': self.account_info.currency,
                    'leverage': self.account_info.leverage,
                    'margin_mode': self.account_info.margin_mode
                }
            
            # Check symbol availability
            symbol = self.config.get('trading_settings', {}).get('symbol', 'XAUUSD')
            symbol_info = mt5.symbol_info(symbol)
            status['symbols_available'] = symbol_info is not None
            
            return status
            
        except Exception as e:
            self.logger.error(f"Connection check error: {e}")
            return status
    
    def handle_login_event(self):
        """Handle login event"""
        self.logger.info(f"Login detected - Account: {self.login_number}, Broker: {self.broker_name}")
        
        if self.on_login_callback:
            try:
                self.on_login_callback(self.account_info)
            except Exception as e:
                self.logger.error(f"Login callback error: {e}")
    
    def handle_logout_event(self):
        """Handle logout event"""
        self.logger.warning("Logout detected - Trading will be suspended")
        
        if self.on_logout_callback:
            try:
                self.on_logout_callback()
            except Exception as e:
                self.logger.error(f"Logout callback error: {e}")
    
    def handle_connection_lost(self):
        """Handle connection lost event"""
        self.logger.error("MT5 connection lost")
        self.is_connected = False
        
        if self.on_connection_lost_callback:
            try:
                self.on_connection_lost_callback()
            except Exception as e:
                self.logger.error(f"Connection lost callback error: {e}")
    
    def handle_connection_restored(self):
        """Handle connection restored event"""
        self.logger.info("MT5 connection restored")
        self.is_connected = True
        
        if self.on_connection_restored_callback:
            try:
                self.on_connection_restored_callback()
            except Exception as e:
                self.logger.error(f"Connection restored callback error: {e}")
    
    def attempt_reconnection(self) -> bool:
        """Attempt to reconnect to MT5"""
        self.logger.info(f"Attempting reconnection... (Attempt {self.connection_attempts + 1})")
        
        try:
            # Shutdown existing connection
            mt5.shutdown()
            time.sleep(2)
            
            # Reinitialize
            if mt5.initialize():
                self.connection_attempts = 0
                self.handle_connection_restored()
                return True
            else:
                self.connection_attempts += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Reconnection attempt failed: {e}")
            self.connection_attempts += 1
            return False
    
    def monitor_connection(self):
        """Monitor connection status in background thread"""
        previous_status = {
            'logged_in': False,
            'connected': False
        }
        
        while self.monitoring_active:
            try:
                current_status = self.check_connection_status()
                
                # Check for login/logout events
                if current_status['logged_in'] != previous_status['logged_in']:
                    if current_status['logged_in']:
                        self.handle_login_event()
                    else:
                        self.handle_logout_event()
                
                # Check for connection events
                mt5_connected = current_status['mt5_initialized']
                if mt5_connected != previous_status['connected']:
                    if mt5_connected:
                        self.handle_connection_restored()
                    else:
                        self.handle_connection_lost()
                
                # Update states
                self.is_logged_in = current_status['logged_in']
                previous_status = {
                    'logged_in': current_status['logged_in'],
                    'connected': mt5_connected
                }
                
                # Attempt reconnection if needed
                if not mt5_connected and self.connection_attempts < self.max_connection_attempts:
                    self.attempt_reconnection()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                time.sleep(self.check_interval)
    
    def start_monitoring(self):
        """Start connection monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self.monitor_connection, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Connection monitoring started")
    
    def stop_monitoring(self):
        """Stop connection monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        self.logger.info("Connection monitoring stopped")
    
    def set_callbacks(self, 
                     on_login: Optional[Callable] = None,
                     on_logout: Optional[Callable] = None, 
                     on_connection_lost: Optional[Callable] = None,
                     on_connection_restored: Optional[Callable] = None):
        """Set event callbacks"""
        self.on_login_callback = on_login
        self.on_logout_callback = on_logout
        self.on_connection_lost_callback = on_connection_lost
        self.on_connection_restored_callback = on_connection_restored
    
    def get_status_report(self) -> Dict[str, any]:
        """Get comprehensive status report"""
        status = self.check_connection_status()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'mt5_status': {
                'initialized': status['mt5_initialized'],
                'terminal_info': status['terminal_info']
            },
            'account_status': {
                'logged_in': status['logged_in'],
                'account_info': status['account_info']
            },
            'trading_status': {
                'symbols_available': status['symbols_available'],
                'connection_attempts': self.connection_attempts,
                'monitoring_active': self.monitoring_active
            },
            'uptime': {
                'last_check': status['last_check'].isoformat(),
                'last_login_check': self.last_login_check.isoformat() if self.last_login_check else None
            }
        }
    
    def force_logout_detection(self):
        """Force check for logout (manual trigger)"""
        self.logger.info("Forcing logout detection check...")
        if not self.update_account_info():
            self.handle_logout_event()
    
    def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        self.logger.warning("Emergency shutdown initiated")
        self.stop_monitoring()
        
        try:
            mt5.shutdown()
            self.logger.info("MT5 shutdown completed")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_monitoring()
        try:
            mt5.shutdown()
        except:
            pass

# Example usage and testing
if __name__ == "__main__":
    def on_login_event(account_info):
        print(f" LOGIN: Account {account_info.login} connected")
    
    def on_logout_event():
        print(" LOGOUT: Account disconnected")
    
    def on_connection_lost():
        print(" CONNECTION LOST: MT5 disconnected")
    
    def on_connection_restored():
        print(" CONNECTION RESTORED: MT5 reconnected")
    
    # Initialize login manager
    login_mgr = LoginManager()
    
    # Set callbacks
    login_mgr.set_callbacks(
        on_login=on_login_event,
        on_logout=on_logout_event,
        on_connection_lost=on_connection_lost,
        on_connection_restored=on_connection_restored
    )
    
    # Start monitoring
    login_mgr.start_monitoring()
    
    try:
        while True:
            # Get status report every 30 seconds
            status = login_mgr.get_status_report()
            print(f"\n Status: {datetime.now().strftime('%H:%M:%S')}")
            print(f"   MT5: {'' if status['mt5_status']['initialized'] else '❌'}")
            print(f"   Login: {'' if status['account_status']['logged_in'] else '❌'}")
            
            if status['account_status']['logged_in']:
                acc_info = status['account_status']['account_info']
                print(f"   Account: {acc_info['login']} | Balance: ${acc_info['balance']:.2f}")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        login_mgr.emergency_shutdown()