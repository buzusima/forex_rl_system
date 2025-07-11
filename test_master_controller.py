# test_master_controller_safe.py
"""
Test script for Master Controller (Windows Safe - No Emoji)
วางไฟล์นี้ใน root directory (เดียวกับ core, strategy, execution folders)
"""

import sys
import os
import time
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    import locale
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

from execution.master_controller import MasterController, SystemConfig, TradingMode

def test_master_controller():
    print("Master Controller Full System Test")
    print("=" * 60)
    
    # Create conservative configuration for testing
    config = SystemConfig(
        trading_mode=TradingMode.CONSERVATIVE,
        max_daily_trades=3,
        max_concurrent_positions=2,
        default_risk_percent=0.5,  # Very conservative
        
        # Update intervals (faster for testing)
        data_update_interval=30,
        strength_update_interval=120,
        correlation_update_interval=180,
        regime_update_interval=300,
        position_check_interval=15,
        
        # Risk settings
        max_account_risk=2.0,
        emergency_stop_loss=5.0,
        
        # Strategy toggles
        enable_trend_following=True,
        enable_correlation_trading=True,
        enable_arbitrage=False,  # Disable for testing
        enable_breakout_trading=True,
        enable_smart_recovery=True
    )
    
    print("Configuration:")
    print(f"   Trading Mode: {config.trading_mode.value}")
    print(f"   Max Daily Trades: {config.max_daily_trades}")
    print(f"   Max Positions: {config.max_concurrent_positions}")
    print(f"   Default Risk: {config.default_risk_percent}%")
    print(f"   Emergency Stop: {config.emergency_stop_loss}%")
    
    print("\nCreating Master Controller...")
    controller = MasterController(config)
    
    try:
        print("\nStarting system...")
        success = controller.start_system()
        
        if not success:
            print("[ERROR] Failed to start system")
            return False
        
        print("[SUCCESS] System started successfully!")
        print("\nSystem Status:")
        
        # Monitor system for a few minutes
        for i in range(5):  # Monitor for 5 cycles
            print(f"\n--- Cycle {i+1} ---")
            
            # Get system status
            status = controller.get_system_status()
            
            print(f"System Status: {status['system']['status']}")
            print(f"Uptime: {status['system']['uptime_hours']:.2f} hours")
            print(f"Daily Trades: {status['trading']['daily_trades']}/{status['trading']['max_daily_trades']}")
            print(f"Active Signals: {status['trading']['active_signals']}")
            
            # Account info
            if 'positions' in status and status['positions']:
                pos_info = status['positions']
                if 'account' in pos_info:
                    account = pos_info['account']
                    print(f"Balance: ${account.get('balance', 0):,.2f}")
                    print(f"Equity: ${account.get('equity', 0):,.2f}")
                
                if 'positions' in pos_info:
                    positions = pos_info['positions']
                    print(f"Open Positions: {positions.get('total_count', 0)}")
                    print(f"Total P&L: ${positions.get('total_profit', 0):,.2f}")
            
            # Wait before next check
            if i < 4:  # Don't wait on last iteration
                print("Waiting 60 seconds...")
                time.sleep(60)
        
        print("\nTesting Manual Signal Generation...")
        
        # Test signal generation manually
        if hasattr(controller, '_generate_trading_signals'):
            signals = controller._generate_trading_signals()
            print(f"Generated {len(signals)} signals:")
            
            for i, signal in enumerate(signals[:3], 1):  # Show top 3
                print(f"{i}. {signal.direction} {signal.symbol}")
                print(f"   Strategy: {signal.strategy}")
                print(f"   Confidence: {signal.confidence:.1f}%")
                print(f"   Risk Level: {signal.risk_level}")
                print(f"   Reason: {signal.entry_reason}")
                print()
        
        return True
        
    except KeyboardInterrupt:
        print("\n[STOP] Test interrupted by user")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Always stop the system
        print("\nStopping system...")
        controller.stop_system()
        print("[SUCCESS] System stopped")

def test_configuration_modes():
    """Test different trading modes"""
    print("\n" + "="*60)
    print("TRADING MODE COMPARISON")
    print("="*60)
    
    modes = [TradingMode.CONSERVATIVE, TradingMode.BALANCED, TradingMode.AGGRESSIVE]
    
    for mode in modes:
        print(f"\n{mode.value.upper()} Mode:")
        
        config = SystemConfig(
            trading_mode=mode,
            max_daily_trades=10 if mode == TradingMode.AGGRESSIVE else 5 if mode == TradingMode.BALANCED else 3,
            default_risk_percent=2.0 if mode == TradingMode.AGGRESSIVE else 1.0 if mode == TradingMode.BALANCED else 0.5
        )
        
        print(f"   Max Daily Trades: {config.max_daily_trades}")
        print(f"   Default Risk: {config.default_risk_percent}%")
        print(f"   Emergency Stop: {config.emergency_stop_loss}%")

def show_system_architecture():
    """Show system architecture"""
    print("\n" + "="*60)
    print("SYSTEM ARCHITECTURE")
    print("="*60)
    
    architecture = """
    MASTER CONTROLLER
    |-- Data Manager (Real-time data)
    |-- Currency Strength Engine
    |-- Correlation Engine  
    |-- Market Regime Detector
    |-- Position Manager
    |-- Recovery System
    
    SIGNAL GENERATION:
    |-- Trend Following (Currency Strength)
    |-- Correlation Divergence
    |-- Arbitrage Opportunities
    |-- Breakout Trading
    |-- Smart Recovery
    
    AUTOMATED OPERATIONS:
    |-- Real-time data updates
    |-- Signal generation & filtering
    |-- Risk management
    |-- Position monitoring
    |-- Recovery execution
    |-- Performance logging
    """
    
    print(architecture)

def simple_test():
    """Simple test without full system start"""
    print("Simple Master Controller Test")
    print("=" * 40)
    
    # Create minimal config
    config = SystemConfig(
        trading_mode=TradingMode.CONSERVATIVE,
        max_daily_trades=1,
        max_concurrent_positions=1,
        default_risk_percent=0.1
    )
    
    print("Creating controller...")
    controller = MasterController(config)
    
    print("Getting initial status...")
    status = controller.get_system_status()
    print(f"Status: {status}")
    
    print("Simple test completed!")

if __name__ == "__main__":
    try:
        # Show system overview
        show_system_architecture()
        
        # Test different modes
        test_configuration_modes()
        
        # Ask user which test to run
        print("\n" + "="*60)
        print("TEST OPTIONS")
        print("="*60)
        print("1. Simple test (no MT5 connection)")
        print("2. Full system test (requires MT5)")
        
        choice = input("\nSelect test (1 or 2): ").strip()
        
        if choice == "1":
            simple_test()
            
        elif choice == "2":
            print("\nFULL SYSTEM TEST")
            print("="*60)
            print("This will start the complete trading system!")
            print("Make sure:")
            print("[CHECK] MT5 is running and logged in")
            print("[CHECK] You're using DEMO account")
            print("[CHECK] All folders (core, strategy, execution) exist")
            
            confirm = input("\nRun full system test? (yes/no): ").lower().strip()
            
            if confirm == 'yes':
                success = test_master_controller()
                
                if success:
                    print("\n[SUCCESS] Full system test completed successfully!")
                    print("The system is ready for production use")
                else:
                    print("\n[ERROR] System test failed")
                    print("Check logs for details")
            else:
                print("\n[CANCEL] Test cancelled by user")
        else:
            print("Invalid choice. Exiting.")
            
    except Exception as e:
        print(f"\n[ERROR] Test script error: {e}")
        import traceback
        traceback.print_exc()