# test_config_aggressive.py
"""
Aggressive Test Configuration
การตั้งค่าสำหรับทดสอบการออกออเดอร์จริง
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from execution.master_controller import MasterController, SystemConfig, TradingMode

def create_test_config():
    """สร้าง config สำหรับทดสอบ - ออกออเดอร์ง่ายขึ้น"""
    
    config = SystemConfig(
        # Trading mode - เปลี่ยนเป็น AGGRESSIVE เพื่อออกออเดอร์ง่ายขึ้น
        trading_mode=TradingMode.AGGRESSIVE,
        
        # เพิ่มจำนวน trades ที่อนุญาต
        max_daily_trades=10,
        max_concurrent_positions=5,
        
        # เพิ่ม risk เล็กน้อย (แต่ยังปลอดภัย)
        default_risk_percent=0.5,  # ใช้ 0.5% ต่อ trade
        max_account_risk=3.0,      # รวมไม่เกิน 3%
        emergency_stop_loss=8.0,   # Emergency stop ที่ 8%
        
        # ลด update intervals ให้เร็วขึ้น
        data_update_interval=30,
        strength_update_interval=60,    # 1 minute
        correlation_update_interval=120, # 2 minutes  
        regime_update_interval=180,     # 3 minutes
        position_check_interval=15,
        
        # ลด correlation threshold
        correlation_threshold=0.8,  # จาก 0.7 เป็น 0.8 (หลวมขึ้น)
        
        # เปิดใช้ strategies ทั้งหมด
        enable_trend_following=True,
        enable_correlation_trading=True,
        enable_arbitrage=True,
        enable_breakout_trading=True,
        enable_smart_recovery=True,
        
        # Trading hours - เปิดตลอด 24 ชั่วโมง
        trading_start_hour=0,
        trading_end_hour=23
    )
    
    return config

def start_aggressive_test():
    """เริ่มทดสอบแบบ aggressive"""
    
    print("=" * 60)
    print("AGGRESSIVE TEST MODE - DEMO ACCOUNT ONLY")
    print("=" * 60)
    print("Settings:")
    print("- Trading Mode: AGGRESSIVE")
    print("- Max Daily Trades: 10")
    print("- Risk per Trade: 0.5%")
    print("- Max Account Risk: 3.0%") 
    print("- Lower confidence thresholds")
    print("- All strategies enabled")
    
    # ยืนยันการใช้ Demo account
    print("\nSAFETY CHECK:")
    print("Make sure you are using DEMO account!")
    
    account_type = input("Confirm you are using DEMO account (yes/no): ").lower().strip()
    
    if account_type != 'yes':
        print("Test cancelled for safety. Please use DEMO account only.")
        return
    
    # สร้าง config
    config = create_test_config()
    
    # สร้าง controller
    print("\nCreating Master Controller...")
    controller = MasterController(config)
    
    # แก้ไข thresholds ให้หลวมขึ้น (hack)
    print("Adjusting thresholds for testing...")
    
    try:
        # เริ่มระบบ
        print("\nStarting system...")
        success = controller.start_system()
        
        if success:
            print("\n[SUCCESS] Aggressive test system started!")
            print("\nMonitoring for trades...")
            print("Press Ctrl+C to stop")
            
            # แสดงสถานะทุก 30 วินาที
            import time
            
            while True:
                try:
                    time.sleep(30)
                    
                    # แสดงสถานะ
                    status = controller.get_system_status()
                    
                    print(f"\n--- Status Update ---")
                    print(f"System: {status['system']['status']}")
                    print(f"Daily Trades: {status['trading']['daily_trades']}/{status['trading']['max_daily_trades']}")
                    print(f"Active Signals: {status['trading']['active_signals']}")
                    
                    # แสดงข้อมูล account
                    if 'positions' in status and status['positions']:
                        pos_info = status['positions']
                        if 'account' in pos_info:
                            account = pos_info['account']
                            print(f"Balance: ${account.get('balance', 0):,.2f}")
                            print(f"Equity: ${account.get('equity', 0):,.2f}")
                        
                        if 'positions' in pos_info:
                            positions = pos_info['positions']
                            print(f"Open Positions: {positions.get('total_count', 0)}")
                            if positions.get('total_count', 0) > 0:
                                print(f"Total P&L: ${positions.get('total_profit', 0):,.2f}")
                    
                    # ลองสร้าง signals manual
                    if hasattr(controller, '_generate_trading_signals'):
                        signals = controller._generate_trading_signals()
                        if signals:
                            print(f"\nGenerated {len(signals)} signals:")
                            for signal in signals[:3]:
                                print(f"  {signal.direction} {signal.symbol} - {signal.confidence:.1f}%")
                    
                except KeyboardInterrupt:
                    print("\n\nStopping test...")
                    break
                except Exception as e:
                    print(f"Error in monitoring: {e}")
                    continue
        
        else:
            print("[ERROR] Failed to start system")
            
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # หยุดระบบ
        print("Shutting down...")
        controller.stop_system()
        print("Test completed!")

def force_test_trade():
    """บังคับทดสอบการออกออเดอร์"""
    
    print("FORCE TEST TRADE")
    print("=" * 30)
    
    # Import engines
    from core.data_manager import DataManager
    from core.currency_strength import CurrencyStrengthEngine
    from core.correlation_engine import CorrelationEngine  
    from core.market_regime import MarketRegimeDetector
    from strategy.position_manager import PositionManager
    
    # Initialize
    dm = DataManager()
    
    if not dm.connect_mt5():
        print("Failed to connect to MT5")
        return
    
    print("Connected to MT5")
    
    # Create engines
    strength_engine = CurrencyStrengthEngine(dm)
    correlation_engine = CorrelationEngine(dm, strength_engine)
    regime_detector = MarketRegimeDetector(dm, strength_engine)
    position_manager = PositionManager(dm, strength_engine, correlation_engine, regime_detector)
    
    print("Engines created")
    
    # ทดสอบออกออเดอร์
    print("\nTesting order placement...")
    
    ticket = position_manager.open_position(
        symbol="EURUSD",
        direction="BUY",
        strategy_type="TEST",
        entry_reason="Force test trade",
        risk_percent=0.1  # ใช้ risk น้อยมาก
    )
    
    if ticket:
        print(f"[SUCCESS] Test order placed! Ticket: {ticket}")
        
        # รอ 10 วินาที แล้วปิด
        import time
        print("Waiting 10 seconds before closing...")
        time.sleep(10)
        
        # ปิด position
        success = position_manager._close_position(ticket, "Test completed")
        if success:
            print("[SUCCESS] Test position closed")
        else:
            print("[ERROR] Failed to close test position")
    else:
        print("[ERROR] Failed to place test order")
    
    # Cleanup
    dm.close()

def main():
    """Main function"""
    
    print("TRADING SYSTEM TEST OPTIONS")
    print("=" * 40)
    print("1. Aggressive Test Mode (Full system)")
    print("2. Force Test Trade (Single order)")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        start_aggressive_test()
    elif choice == '2':
        force_test_trade()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()