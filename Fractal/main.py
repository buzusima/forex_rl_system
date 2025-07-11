#!/usr/bin/env python3
"""
XAUUSD Multi-Timeframe Trading System
Main launcher with GUI and command line options
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_default_config():
    """Create default config.json if not exists"""
    default_config = {
        "trading_settings": {
            "symbol": "XAUUSD",
            "lot_size": 0.01,
            "max_positions": 5,
            "anti_hedge": True,
            "trading_enabled": False
        },
        "signal_parameters": {
            "rsi_period": 14,
            "rsi_up": 55,
            "rsi_down": 45,
            "fractal_period": 5,
            "fractal_lookback": 5,
            "min_signal_interval": 60
        },
        "timeframes": {
            "entry_tf": "M1",
            "trend_tf": "M15",
            "bias_tf": "H1",
            "require_tf_alignment": True,
            "tf_weights": {
                "entry": 0.4,
                "trend": 0.35,
                "bias": 0.25
            }
        },
        "multi_tf_thresholds": {
            "trend_bullish_threshold": 50,
            "trend_bearish_threshold": 50,
            "bias_strong_bull": 55,
            "bias_strong_bear": 45,
            "bias_neutral_upper": 52,
            "bias_neutral_lower": 48
        },
        "risk_management": {
            "daily_loss_limit": 100,
            "max_daily_trades": 20,
            "max_drawdown_percent": 15,
            "emergency_stop_loss": 200,
            "position_size_scaling": True,
            "risk_per_trade_percent": 2.0
        },
        "take_profit": {
            "enable_tp": False,
            "tp_points": 200,
            "dynamic_tp": False,
            "tp_multiplier": 1.5,
            "trailing_stop": False,
            "trailing_points": 50
        },
        "recovery_system": {
            "enable_recovery": True,
            "recovery_trigger_points": 100,
            "martingale_multiplier": 2.0,
            "max_recovery_levels": 3,
            "recovery_wait_bars": 3,
            "smart_recovery": True
        },
        "spread_management": {
            "max_spread_points": 50,
            "spread_filter_enabled": True,
            "spread_multiplier": 1.5,
            "spread_buffer": 5,
            "dynamic_spread_adjustment": True
        },
        "session_filters": {
            "enable_session_filter": False,
            "allowed_sessions": ["London", "NewYork"],
            "avoid_asian_session": False,
            "session_overlap_only": False,
            "weekend_trading": False
        },
        "news_filter": {
            "enable_news_filter": False,
            "news_impact_levels": ["High"],
            "stop_before_news_minutes": 30,
            "resume_after_news_minutes": 60,
            "news_symbols": ["USD", "EUR", "GBP"]
        },
        "volatility_filter": {
            "enable_volatility_filter": True,
            "atr_period": 14,
            "min_atr_threshold": 10,
            "max_atr_threshold": 100,
            "volatility_scaling": True
        },
        "execution_settings": {
            "execution_mode": "MARKET",
            "slippage_points": 20,
            "max_retries": 3,
            "retry_delay_ms": 1000,
            "fill_or_kill": False,
            "partial_fill_allowed": True
        },
        "monitoring": {
            "check_interval_seconds": 30,
            "log_level": "INFO",
            "save_signals_history": True,
            "performance_tracking": True,
            "real_time_alerts": True,
            "telegram_notifications": False
        },
        "gui_settings": {
            "theme": "dark",
            "auto_refresh_seconds": 5,
            "show_chart_overlay": True,
            "compact_mode": False,
            "always_on_top": False,
            "minimize_to_tray": True
        },
        "advanced_features": {
            "machine_learning": False,
            "pattern_recognition": False,
            "sentiment_analysis": False,
            "correlation_filter": False,
            "seasonal_adjustment": False
        },
        "presets": {
            "scalping": {
                "entry_tf": "M1",
                "trend_tf": "M5",
                "bias_tf": "M15",
                "rsi_up": 60,
                "rsi_down": 40,
                "check_interval_seconds": 15
            },
            "intraday": {
                "entry_tf": "M15",
                "trend_tf": "H1", 
                "bias_tf": "H4",
                "rsi_up": 55,
                "rsi_down": 45,
                "check_interval_seconds": 60
            },
            "swing": {
                "entry_tf": "H1",
                "trend_tf": "H4",
                "bias_tf": "D1", 
                "rsi_up": 50,
                "rsi_down": 50,
                "check_interval_seconds": 300
            }
        },
        "version": "1.0.0",
        "last_updated": datetime.now().isoformat(),
        "config_hash": ""
    }
    
    if not os.path.exists('config.json'):
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        print("Created default config.json")
    
    return default_config

def check_dependencies():
    """Check if required dependencies are installed"""
    required_modules = [
        'MetaTrader5',
        'pandas', 
        'numpy',
        'tkinter'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Missing required modules: {', '.join(missing_modules)}")
        print(f"Install with: pip install {' '.join(missing_modules)}")
        return False
    
    return True

def run_gui_mode():
    """Run with GUI interface"""
    print("🖥️ Starting GUI mode...")
    
    try:
        from gui_interface import TradingGUI
        from trading_engine import XAUUSDTradingEngine
        from login_manager import LoginManager
        from symbol_manager import SymbolManager
        
        # Initialize components
        print("⚙️ Initializing trading engine...")
        engine = XAUUSDTradingEngine()
        
        print("🖥️ Starting GUI...")
        app = TradingGUI()
        
        # Connect components
        app.set_managers(
            login_manager=engine.login_manager,
            symbol_manager=engine.symbol_manager,
            trading_engine=engine
        )
        
        print(" System ready! Opening GUI...")
        app.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📁 Make sure all files are in the same directory")
    except Exception as e:
        print(f"❌ GUI startup error: {e}")

def run_command_line():
    """Run in command line mode"""
    print("⌨️ Starting command line mode...")
    
    try:
        from trading_engine import XAUUSDTradingEngine
        import time
        
        def on_signal_event(event_type, message):
            print(f"📡 [{datetime.now().strftime('%H:%M:%S')}] {event_type}: {message}")
        
        def on_trade_event(action, volume, price, signal):
            print(f"💰 [{datetime.now().strftime('%H:%M:%S')}] TRADE: {action} {volume} lots at {price:.5f}")
        
        def on_error_event(error_type, message):
            print(f"❌ [{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_type} - {message}")
        
        # Initialize engine
        print("⚙️ Initializing trading engine...")
        engine = XAUUSDTradingEngine()
        
        # Set callbacks
        engine.set_callbacks(
            on_signal=on_signal_event,
            on_trade=on_trade_event,
            on_error=on_error_event
        )
        
        print(" Engine initialized")
        print(" Starting trading...")
        print("Press Ctrl+C to stop\n")
        
        # Start trading
        engine.start_trading()
        
        # Main loop
        try:
            while True:
                if engine.is_running and engine.is_connected:
                    engine.analyze_and_trade()
                    
                    # Status report every 5 minutes
                    if engine.daily_stats['trades'] % 10 == 0:
                        status = engine.get_status_report()
                        print(f"\n📊 Status Report [{datetime.now().strftime('%H:%M:%S')}]:")
                        print(f"   Engine: {'🟢 Running' if status['engine_status']['running'] else '🔴 Stopped'}")
                        print(f"   Symbol: {status['engine_status']['symbol']}")
                        print(f"   Trades Today: {status['daily_stats']['trades']}")
                        print(f"   P&L: ${status['daily_stats']['profit']:.2f}")
                        print(f"   Open Positions: {status['positions']['count']}")
                        if status['current_analysis']:
                            print(f"   Current RSI: {status['current_analysis'].get('rsi', 0):.2f}")
                            print(f"   Signal: {status['current_analysis'].get('signal', 'NONE')}")
                        print(f"   Spread: {status['market_data']['spread']:.1f} points")
                        print("-" * 50)
                
                # Sleep for check interval
                check_interval = engine.config.get('monitoring', {}).get('check_interval_seconds', 30)
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print(f"\n Stopping trading engine...")
            engine.cleanup()
            print(" Shutdown complete")
        
    except ImportError as e:
        print(f" Import error: {e}")
    except Exception as e:
        print(f" Command line error: {e}")

def run_test_mode():
    """Run in test mode (signal detection only)"""
    print("🧪 Starting test mode (signal detection only)...")
    
    try:
        from trading_engine import XAUUSDTradingEngine
        import time
        
        # Initialize engine
        engine = XAUUSDTradingEngine()
        
        # Disable actual trading
        engine.config['trading_settings']['trading_enabled'] = False
        
        print(" Test mode initialized")
        print(" Signal detection active (no actual trading)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                if engine.is_connected:
                    # Just analyze, don't trade
                    signal = engine.analyze_multi_timeframe()
                    
                    if signal:
                        print(f" [{datetime.now().strftime('%H:%M:%S')}] SIGNAL DETECTED:")
                        print(f"   Type: {signal.signal_type.name}")
                        print(f"   Strength: {signal.strength.name}")
                        print(f"   Price: {signal.price:.5f}")
                        print(f"   RSI: {signal.rsi_value:.2f}")
                        print(f"   Fractal: {signal.fractal_type}")
                        print(f"   Score: {signal.confirmation_score:.2f}")
                        print(f"   Reason: {signal.entry_reason}")
                        print("-" * 50)
                    else:
                        # Show current market state every 10 cycles
                        status = engine.get_status_report()
                        if status['current_analysis']:
                            analysis = status['current_analysis']
                            print(f"📊 [{datetime.now().strftime('%H:%M:%S')}] Market: "
                                  f"RSI={analysis.get('rsi', 0):.2f}, "
                                  f"Signal={analysis.get('signal', 'NONE')}, "
                                  f"Spread={status['market_data']['spread']:.1f}pts")
                
                time.sleep(30)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Test mode stopped")
        
    except Exception as e:
        print(f"❌ Test mode error: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="XAUUSD Multi-Timeframe Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with GUI (default)
  python main.py --cli              # Run in command line mode  
  python main.py --test             # Test mode (signals only)
  python main.py --setup            # Setup and exit
        """
    )
    
    parser.add_argument('--cli', action='store_true', 
                       help='Run in command line mode')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (signal detection only)')
    parser.add_argument('--setup', action='store_true',
                       help='Setup configuration and exit')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 60)
    print(" XAUUSD Multi-Timeframe Trading System v1.0")
    print(" Fractal + RSI + Multi-TF Confirmation")
    print("=" * 60)
    
    # Setup mode
    if args.setup:
        print(" Setting up configuration...")
        create_default_config()
        print(" Setup complete!")
        print(" Edit config.json to customize settings")
        print(" Run 'python main.py' to start")
        return
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Create config if not exists
    create_default_config()
    
    # Choose run mode
    if args.test:
        run_test_mode()
    elif args.cli:
        run_command_line()
    else:
        run_gui_mode()

if __name__ == "__main__":
    main()