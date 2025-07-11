# auto_trading_bot.py - Automated Trading Bot
"""
Auto Trading Bot ที่รันทิ้งไว้ได้
เชื่อมต่อ AI Model + MT5 Demo Account
"""

import sys
import os
import time
import schedule
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import logging
from typing import Dict, Optional

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.rl.ppo_agent import create_ppo_agent
from config.mt5_connector import create_mt5_connector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoTradingBot:
    """Automated Trading Bot"""
    
    def __init__(self, 
                 symbol: str = "GBPUSD",
                 timeframe: str = "M15",
                 model_path: str = "models/ppo_forex/ppo_model_final.pt",
                 position_size: float = 0.01,
                 max_positions: int = 1,
                 risk_percent: float = 1.0):
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_path = model_path
        self.position_size = position_size
        self.max_positions = max_positions
        self.risk_percent = risk_percent
        
        # Trading state
        self.current_positions = []
        self.last_signal = None
        self.last_signal_time = None
        self.total_trades = 0
        self.total_pnl = 0.0
        
        # Components
        self.agent = None
        self.mt5_connector = None
        
        # Performance tracking
        self.trading_log = []
        
        self.initialize()
    
    def initialize(self):
        """Initialize bot components"""
        try:
            logger.info("🤖 Initializing Auto Trading Bot...")
            
            # Load AI model
            obs_dim = 243
            action_dim = 3
            self.agent = create_ppo_agent(obs_dim, action_dim)
            self.agent.load(self.model_path)
            logger.info(f"✅ AI Model loaded: {self.model_path}")
            
            # Initialize MT5 connection
            self.mt5_connector = create_mt5_connector()
            if self.mt5_connector.connect():
                logger.info("✅ MT5 connected successfully")
                
                # Check account info
                account_info = mt5.account_info()
                if account_info:
                    logger.info(f"📊 Account: {account_info.login}")
                    logger.info(f"💰 Balance: ${account_info.balance:,.2f}")
                    logger.info(f"🏦 Server: {account_info.server}")
            else:
                raise Exception("Failed to connect to MT5")
            
            logger.info(f"🎯 Bot configured:")
            logger.info(f"   Symbol: {self.symbol}")
            logger.info(f"   Position Size: {self.position_size} lots")
            logger.info(f"   Max Positions: {self.max_positions}")
            logger.info(f"   Risk per Trade: {self.risk_percent}%")
            
        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            raise
    
    def get_market_data(self):
        """Get current market data for signal generation"""
        try:
            # Get historical data for indicators
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            data = self.mt5_connector.get_historical_data(
                self.symbol, self.timeframe, start_date, end_date
            )
            
            if data is None or len(data) < 100:
                logger.warning(f"Insufficient data: {len(data) if data else 0} bars")
                return None
            
            # Calculate indicators
            from src.features.technical_indicators import create_indicators_calculator
            calculator = create_indicators_calculator()
            
            data_with_indicators = calculator.calculate_all_indicators(data)
            if len(data_with_indicators) < 50:
                logger.warning(f"Insufficient indicator data: {len(data_with_indicators)} bars")
                return None
            
            normalized_data = calculator.normalize_features(data_with_indicators)
            
            # Get current tick
            current_tick = self.mt5_connector.get_current_tick(self.symbol)
            
            return {
                'data': normalized_data,
                'features': calculator.get_feature_columns(),
                'tick': current_tick
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def generate_signal(self, market_data):
        """Generate trading signal from AI model"""
        try:
            data = market_data['data']
            available_features = [col for col in market_data['features'] if col in data.columns]
            
            # Prepare observation
            recent_features = data[available_features].tail(20).values.flatten()
            account_info = [1.0, 0.0, 0.0]  # normalized balance, position, pnl
            observation = list(recent_features) + account_info
            
            # Adjust to model size
            obs_dim = 243
            if len(observation) < obs_dim:
                observation.extend([0.0] * (obs_dim - len(observation)))
            elif len(observation) > obs_dim:
                observation = observation[:obs_dim]
            
            observation = np.array(observation, dtype=np.float32)
            
            # Get AI prediction
            action, log_prob, value = self.agent.get_action(observation, deterministic=True)
            
            signal_info = {
                'action': action,
                'confidence': abs(log_prob),
                'value': value,
                'timestamp': datetime.now(),
                'price': market_data['tick']
            }
            
            return signal_info
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def get_current_positions(self):
        """Get current open positions"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            return list(positions) if positions else []
        except:
            return []
    
    def execute_trade(self, signal_info):
        """Execute trade based on signal"""
        try:
            action = signal_info['action']
            price = signal_info['price']
            
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Check position limits
            if len(current_positions) >= self.max_positions and action != 0:
                logger.info(f"⏸️ Position limit reached ({len(current_positions)}/{self.max_positions})")
                return False
            
            action_names = ["HOLD", "BUY", "SELL"]
            logger.info(f"🎯 Signal: {action_names[action]} | Confidence: {signal_info['confidence']:.2f}")
            
            if action == 0:  # HOLD
                return True
            
            # Prepare trade request
            lot = self.position_size
            deviation = 20
            
            if action == 1:  # BUY
                trade_type = mt5.ORDER_TYPE_BUY
                price_value = price['ask']
                sl = price_value - (50 * self.mt5_connector.symbols_info.get(self.symbol, {}).get('point', 0.00001))
                tp = price_value + (100 * self.mt5_connector.symbols_info.get(self.symbol, {}).get('point', 0.00001))
                
            elif action == 2:  # SELL
                trade_type = mt5.ORDER_TYPE_SELL
                price_value = price['bid']
                sl = price_value + (50 * self.mt5_connector.symbols_info.get(self.symbol, {}).get('point', 0.00001))
                tp = price_value - (100 * self.mt5_connector.symbols_info.get(self.symbol, {}).get('point', 0.00001))
            
            # Create trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot,
                "type": trade_type,
                "price": price_value,
                "sl": sl,
                "tp": tp,
                "deviation": deviation,
                "magic": 234000,
                "comment": f"AI_Bot_{action_names[action]}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send trade request
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Trade failed: {result.retcode} - {result.comment}")
                return False
            
            # Log successful trade
            self.total_trades += 1
            trade_info = {
                'timestamp': datetime.now(),
                'action': action_names[action],
                'symbol': self.symbol,
                'volume': lot,
                'price': price_value,
                'sl': sl,
                'tp': tp,
                'ticket': result.order,
                'confidence': signal_info['confidence']
            }
            
            self.trading_log.append(trade_info)
            
            logger.info(f"✅ Trade executed: {action_names[action]} {lot} {self.symbol} @ {price_value:.5f}")
            logger.info(f"   Ticket: {result.order} | SL: {sl:.5f} | TP: {tp:.5f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return False
    
    def check_and_trade(self):
        """Main trading logic - check signal and execute"""
        try:
            logger.info(f"🔄 Checking market for {self.symbol}...")
            
            # Get market data
            market_data = self.get_market_data()
            if market_data is None:
                logger.warning("⚠️ Failed to get market data")
                return
            
            # Generate signal
            signal_info = self.generate_signal(market_data)
            if signal_info is None:
                logger.warning("⚠️ Failed to generate signal")
                return
            
            # Store signal
            self.last_signal = signal_info
            self.last_signal_time = datetime.now()
            
            # Execute trade
            success = self.execute_trade(signal_info)
            
            # Update performance
            if success:
                self.update_performance()
            
        except Exception as e:
            logger.error(f"❌ Trading cycle error: {e}")
    
    def update_performance(self):
        """Update and log performance"""
        try:
            account_info = mt5.account_info()
            if account_info:
                current_balance = account_info.balance
                current_equity = account_info.equity
                
                logger.info(f"📊 Account Status:")
                logger.info(f"   Balance: ${current_balance:,.2f}")
                logger.info(f"   Equity: ${current_equity:,.2f}")
                logger.info(f"   Total Trades: {self.total_trades}")
                
                # Check positions
                positions = self.get_current_positions()
                if positions:
                    logger.info(f"   Open Positions: {len(positions)}")
                    for pos in positions:
                        pnl = pos.profit
                        logger.info(f"     {pos.type_str} {pos.volume} @ {pos.price_open:.5f} | P&L: ${pnl:.2f}")
        
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def emergency_close_all(self):
        """Emergency function to close all positions"""
        try:
            positions = self.get_current_positions()
            if not positions:
                logger.info("No positions to close")
                return
            
            logger.warning(f"🚨 EMERGENCY: Closing {len(positions)} positions")
            
            for position in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "Emergency_Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(close_request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ Position {position.ticket} closed")
                else:
                    logger.error(f"❌ Failed to close {position.ticket}: {result.comment}")
        
        except Exception as e:
            logger.error(f"Emergency close error: {e}")
    
    def run_bot(self):
        """Run the bot with scheduling"""
        logger.info("🚀 Starting Auto Trading Bot...")
        logger.info("📅 Schedule: Every 15 minutes during market hours")
        
        # Schedule trading checks every 15 minutes
        schedule.every(15).minutes.do(self.check_and_trade)
        
        # Schedule performance update every hour
        schedule.every().hour.do(self.update_performance)
        
        # Initial check
        self.check_and_trade()
        
        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user")
            self.emergency_close_all()
        except Exception as e:
            logger.error(f"💥 Bot crashed: {e}")
            self.emergency_close_all()

def main():
    """Main function"""
    print("🤖 FOREX AI AUTO TRADING BOT")
    print("="*50)
    print("⚠️  This will trade automatically!")
    print("⚠️  Make sure you're using DEMO account!")
    print("="*50)
    
    # Confirmation
    confirm = input("Continue with AUTO TRADING? (yes/no): ").lower()
    if confirm != 'yes':
        print("❌ Aborted")
        return
    
    try:
        # Create and run bot
        bot = AutoTradingBot(
            symbol="GBPUSD",
            timeframe="M15",
            position_size=0.01,  # 0.01 lots = $1 per pip
            max_positions=1,
            risk_percent=1.0
        )
        
        print("\n🚀 Bot is running!")
        print("📊 Check 'trading_bot.log' for detailed logs")
        print("🛑 Press Ctrl+C to stop")
        
        bot.run_bot()
        
    except Exception as e:
        print(f"💥 Bot failed to start: {e}")

if __name__ == "__main__":
    main()