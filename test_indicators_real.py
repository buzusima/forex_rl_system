# test_indicators_real.py - ทดสอบ Indicators กับข้อมูลจริง
"""
ไฟล์นี้ทดสอบ technical indicators กับข้อมูลจริงที่เราดึงจาก MT5
"""

import sys
import os

# Add paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_indicators_with_real_data():
    """ทดสอบ indicators กับข้อมูลจริง"""
    try:
        print("🧪 Testing Technical Indicators with Real Data...")
        print("=" * 50)
        
        # Import modules
        from config.config import ForexRLConfig
        from src.data.database_manager import create_database_manager
        from src.features.technical_indicators import create_indicators_calculator
        
        # สร้าง instances
        config = ForexRLConfig()
        db = create_database_manager()
        calculator = create_indicators_calculator()
        
        print("✅ Modules imported successfully")
        
        # ดูข้อมูลที่มีใน database
        summary = db.get_data_summary()
        print(f"\n📊 Database Summary:")
        print(f"  Total records: {summary.get('total_records', 0)}")
        print(f"  Symbols: {len(summary.get('symbols', []))}")
        print(f"  Timeframes: {summary.get('timeframes', [])}")
        
        if summary.get('total_records', 0) == 0:
            print("\n⚠️ No data in database. Let's use sample data instead.")
            # ใช้ sample data
            import pandas as pd
            import numpy as np
            
            dates = pd.date_range(start='2024-01-01', periods=200, freq='15T')
            np.random.seed(42)
            base_price = 1.1000
            price_changes = np.random.normal(0, 0.0002, 200)
            close_prices = base_price + np.cumsum(price_changes)
            
            test_data = pd.DataFrame({
                'open': close_prices + np.random.normal(0, 0.00005, 200),
                'high': close_prices + abs(np.random.normal(0, 0.0001, 200)),
                'low': close_prices - abs(np.random.normal(0, 0.0001, 200)),
                'close': close_prices,
                'tick_volume': np.random.randint(50, 300, 200)
            }, index=dates)
            
            symbol = "EURUSD_SAMPLE"
            timeframe = "M15"
            
        else:
            # ใช้ข้อมูลจริงจาก database
            symbols = summary.get('symbols', [])
            timeframes = summary.get('timeframes', [])
            
            if symbols and timeframes:
                symbol = symbols[0]  # เอาตัวแรก
                timeframe = timeframes[0]  # เอาตัวแรก
                
                print(f"\n📈 Testing with: {symbol} {timeframe}")
                
                # ดึงข้อมูลจาก database
                test_data = db.get_price_data(symbol, timeframe)
                
                if test_data is None or len(test_data) == 0:
                    print("❌ No data retrieved from database")
                    return False
            else:
                print("❌ No symbols or timeframes found")
                return False
        
        print(f"\n📊 Data loaded: {len(test_data)} bars")
        print(f"   Period: {test_data.index[0]} to {test_data.index[-1]}")
        print(f"   Sample prices: {test_data['close'].iloc[:5].values}")
        
        # คำนวณ indicators
        print(f"\n🔄 Calculating technical indicators...")
        data_with_indicators = calculator.calculate_all_indicators(test_data)
        
        if len(data_with_indicators) == 0:
            print("❌ No indicators calculated")
            return False
        
        print(f"✅ Indicators calculated: {len(data_with_indicators)} valid bars")
        
        # แสดง indicators ล่าสุด
        print(f"\n📊 Latest Indicators for {symbol} {timeframe}:")
        print("-" * 50)
        
        latest = data_with_indicators.iloc[-1]
        
        print(f"📈 Price Info:")
        print(f"   Close: {latest['close']:.5f}")
        print(f"   Price Change: {latest['price_change']*100:.3f}%")
        
        print(f"\n🎯 RSI: {latest['rsi']:.2f}")
        if latest['rsi'] > 70:
            print("   → Overbought (แพงเกินไป)")
        elif latest['rsi'] < 30:
            print("   → Oversold (ถูกเกินไป)")
        else:
            print("   → Neutral")
        
        print(f"\n📊 MACD:")
        print(f"   MACD: {latest['macd']:.6f}")
        print(f"   Signal: {latest['macd_signal']:.6f}")
        print(f"   Histogram: {latest['macd_histogram']:.6f}")
        if latest['macd'] > latest['macd_signal']:
            print("   → Bullish (ขาขึ้น)")
        else:
            print("   → Bearish (ขาลง)")
        
        print(f"\n📏 ATR: {latest['atr']:.6f}")
        print("   → Volatility measure")
        
        print(f"\n🎈 Bollinger Bands:")
        print(f"   Upper: {latest['bb_upper']:.5f}")
        print(f"   Middle: {latest['bb_middle']:.5f}")
        print(f"   Lower: {latest['bb_lower']:.5f}")
        print(f"   %B: {latest['bb_percent']:.3f}")
        if latest['bb_percent'] > 1:
            print("   → Above upper band")
        elif latest['bb_percent'] < 0:
            print("   → Below lower band")
        else:
            print("   → Within bands")
        
        # ทดสอบ normalization
        print(f"\n🔧 Testing normalization...")
        normalized_data = calculator.normalize_features(data_with_indicators)
        print("✅ Features normalized for RL")
        
        # แสดง normalized values
        print(f"\n📊 Normalized Features (latest):")
        feature_cols = calculator.get_feature_columns()
        available_features = [col for col in feature_cols if col in normalized_data.columns]
        
        for feature in available_features[:5]:  # แสดง 5 ตัวแรก
            original = data_with_indicators[feature].iloc[-1]
            normalized = normalized_data[feature].iloc[-1]
            print(f"   {feature}: {original:.6f} → {normalized:.6f}")
        
        print(f"\n🎉 Technical Indicators test completed successfully!")
        print(f"📊 Ready for RL training with {len(available_features)} features")
        
        # ปิด database
        db.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_indicators_with_real_data()
    
    if success:
        print("\n🚀 Next step: Create RL Environment with these indicators")
    else:
        print("\n🔧 Please fix the issues above")
    
    input("\nPress Enter to exit...")