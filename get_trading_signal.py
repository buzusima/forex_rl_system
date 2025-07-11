# get_trading_signal.py - Get Real-time Trading Signal
"""
สคริปต์สำหรับดู trading signal จาก trained model
รันก่อนเทรดทุกครั้ง
"""

import sys
import os
from datetime import datetime
import argparse
import numpy as np

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.rl.ppo_agent import create_ppo_agent
from src.rl.trading_environment import create_trading_environment
from config.mt5_connector import create_mt5_connector
import MetaTrader5 as mt5

def get_current_market_data(symbol, timeframe, window_size=100):
    """ดึงข้อมูลตลาดปัจจุบัน"""
    try:
        # Connect to MT5
        connector = create_mt5_connector()
        if not connector.connect():
            print("❌ ไม่สามารถเชื่อมต่อ MT5 ได้")
            return None
        
        # Get more data for indicators calculation
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Get last month
        
        print(f"🔄 Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        data = connector.get_historical_data(symbol, timeframe, start_date, end_date)
        
        if data is None:
            print("❌ ไม่สามารถดึงข้อมูลได้")
            connector.disconnect()
            return None
            
        print(f"✅ ได้ข้อมูล {len(data)} bars")
        
        if len(data) < window_size:
            print(f"❌ ข้อมูลไม่เพียงพอสำหรับ indicators: {len(data)} < {window_size}")
            connector.disconnect()
            return None
        
        # Get current tick
        current_tick = connector.get_current_tick(symbol)
        if current_tick is None:
            print("❌ ไม่สามารถดึง current tick ได้")
        
        connector.disconnect()
        
        return data, current_tick
        
    except Exception as e:
        print(f"❌ Error getting market data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_trading_signal(symbol="GBPUSD", timeframe="M15", model_path="models/ppo_forex/ppo_model_final.pt"):
    """ได้ trading signal จาก model"""
    
    print(f"🔍 Getting trading signal for {symbol} {timeframe}")
    print("="*50)
    
    try:
        # Get current market data
        market_data = get_current_market_data(symbol, timeframe)
        if market_data is None:
            return None
        
        data, current_tick = market_data
        
        # Create environment with current data
        from src.features.technical_indicators import create_indicators_calculator
        
        calculator = create_indicators_calculator()
        
        print(f"🔄 Calculating technical indicators...")
        data_with_indicators = calculator.calculate_all_indicators(data)
        
        print(f"✅ Indicators calculated: {len(data_with_indicators)} valid bars")
        
        if len(data_with_indicators) < 50:
            print(f"❌ ข้อมูล indicators ไม่เพียงพอ: {len(data_with_indicators)} < 50")
            return None
        
        normalized_data = calculator.normalize_features(data_with_indicators)
        
        print(f"✅ Features normalized: {len(normalized_data)} bars ready")
        
        # Load trained model
        obs_dim = 243  # Your model's observation dimension
        action_dim = 3  # Hold, Buy, Sell
        
        agent = create_ppo_agent(obs_dim, action_dim)
        agent.load(model_path)
        
        print(f"✅ Model loaded from {model_path}")
        
        # Prepare observation (last 20 bars + account info)
        feature_cols = calculator.get_feature_columns()
        available_features = [col for col in feature_cols if col in normalized_data.columns]
        
        # Get last 20 bars of features
        recent_features = normalized_data[available_features].tail(20).values.flatten()
        
        # Add dummy account info (since we're just getting signal)
        account_info = [1.0, 0.0, 0.0]  # normalized balance, no position, no PnL
        
        # Combine observation
        observation = list(recent_features) + account_info
        
        # Pad or trim to exact size
        if len(observation) < obs_dim:
            observation.extend([0.0] * (obs_dim - len(observation)))
        elif len(observation) > obs_dim:
            observation = observation[:obs_dim]
        
        observation = np.array(observation, dtype=np.float32)
        
        # Get trading signal
        action, log_prob, value = agent.get_action(observation, deterministic=True)
        
        # Action meanings
        action_names = ["HOLD", "BUY", "SELL"]
        action_colors = ["🟡", "🟢", "🔴"]
        
        # Print results
        print(f"📊 Market Information:")
        print(f"   Symbol: {symbol}")
        print(f"   Current Price: {current_tick['bid']:.5f} / {current_tick['ask']:.5f}")
        print(f"   Spread: {current_tick['spread']*10000:.1f} pips")
        print(f"   Time: {current_tick['time']}")
        
        print(f"\n🤖 Model Prediction:")
        print(f"   Signal: {action_colors[action]} {action_names[action]}")
        print(f"   Confidence: {abs(log_prob):.2f}")
        print(f"   Value Estimate: {value:.4f}")
        
        # Trading recommendation
        print(f"\n💡 Trading Recommendation:")
        if action == 0:  # HOLD
            print("   ⏸️  No action required - Hold current position")
        elif action == 1:  # BUY
            print(f"   📈 Consider BUYING {symbol}")
            print(f"   Entry: ~{current_tick['ask']:.5f}")
            print(f"   Risk: 1-2% of account")
        elif action == 2:  # SELL
            print(f"   📉 Consider SELLING {symbol}")
            print(f"   Entry: ~{current_tick['bid']:.5f}")
            print(f"   Risk: 1-2% of account")
        
        # Technical info
        print(f"\n📋 Technical Details:")
        latest_data = normalized_data.iloc[-1]
        print(f"   RSI: {latest_data.get('rsi', 0)*100:.1f}")
        print(f"   MACD: {latest_data.get('macd', 0):.6f}")
        print(f"   ATR: {latest_data.get('atr', 0):.6f}")
        
        print("="*50)
        
        return {
            'symbol': symbol,
            'action': action,
            'action_name': action_names[action],
            'confidence': abs(log_prob),
            'value': value,
            'current_price': current_tick,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        print(f"❌ Error getting trading signal: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Get Trading Signal')
    parser.add_argument('--symbol', type=str, default='GBPUSD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='M15', help='Timeframe')
    parser.add_argument('--model-path', type=str, default='models/ppo_forex/ppo_model_final.pt', help='Model path')
    
    args = parser.parse_args()
    
    # Get signal
    signal = get_trading_signal(args.symbol, args.timeframe, args.model_path)
    
    if signal:
        print(f"✅ Signal generated successfully!")
        
        # Save to log file
        log_file = f"trading_signals_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(log_file, 'a') as f:
            f.write(f"{signal['timestamp']}: {signal['symbol']} -> {signal['action_name']} (conf: {signal['confidence']:.2f})\n")
        
        print(f"📝 Signal logged to {log_file}")
    else:
        print("❌ Failed to generate signal")

if __name__ == "__main__":
    main()