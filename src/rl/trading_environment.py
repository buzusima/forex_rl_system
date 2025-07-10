# src/rl/trading_environment.py - Basic Trading Environment for Single Asset
"""
ไฟล์นี้สร้าง RL Environment สำหรับเทรดคู่เงินเดียว (เริ่มจาก EURUSD)
แก้ไขไฟล์นี้เมื่อ: ต้องการปรับ action space, reward function หรือ state representation
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleAssetTradingEnv(gym.Env):
    """Basic RL Environment สำหรับเทรดคู่เงินเดียว"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 symbol: str = "EURUSD",
                 timeframe: str = "M15",
                 window_size: int = 20,
                 initial_balance: float = 10000.0,
                 max_position_size: float = 1.0,
                 spread: float = 0.00015,
                 commission: float = 0.0):
        """
        Initialize Trading Environment
        
        Args:
            data: DataFrame with OHLC and technical indicators
            symbol: Trading symbol (e.g., EURUSD)  
            timeframe: Trading timeframe (e.g., M15)
            window_size: Number of bars to use as observation
            initial_balance: Starting account balance
            max_position_size: Maximum position size (in lots)
            spread: Bid-Ask spread (in price units)
            commission: Commission per trade (in price units)
        """
        super().__init__()
        
        self.data = data.copy()
        self.symbol = symbol
        self.timeframe = timeframe
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.spread = spread
        self.commission = commission
        
        # ตรวจสอบ required columns
        required_columns = ['open', 'high', 'low', 'close', 'rsi', 'macd', 'atr', 'bb_percent']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Feature columns สำหรับ observation
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            'price_change', 'price_change_abs'
        ]
        
        # เลือกเฉพาะ feature columns ที่มีใน data
        self.feature_columns = [col for col in self.feature_columns if col in data.columns]
        self.n_features = len(self.feature_columns)
        
        logger.info(f"Environment created for {symbol} {timeframe}")
        logger.info(f"Data: {len(data)} bars, Features: {self.n_features}")
        logger.info(f"Window size: {window_size}, Max position: {max_position_size}")
        
        # Define action space
        # 0: Hold, 1: Buy, 2: Sell  
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        # [price_features (window_size x n_features) + account_info (3)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(window_size * self.n_features + 3,),  # +3 for balance, position, unrealized_pnl
            dtype=np.float32
        )
        
        # Initialize environment state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment และ return initial observation"""
        
        # เลือก starting point แบบสุ่ม (แต่ไม่ใกล้จุดสิ้นสุด)
        max_start = len(self.data) - self.window_size - 100  # เหลือ 100 bars สำหรับเทรด
        self.current_step = np.random.randint(self.window_size, max_start)
        
        # Reset account state
        self.balance = self.initial_balance
        self.position = 0.0  # 0 = no position, >0 = long, <0 = short
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Trade history
        self.trade_history = []
        self.equity_curve = [self.balance]
        
        logger.debug(f"Environment reset at step {self.current_step}")
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action และ return (observation, reward, done, info)"""
        
        # Get current price
        current_price = self.data['close'].iloc[self.current_step]
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Calculate current equity
        current_equity = self.balance + self.unrealized_pnl
        self.equity_curve.append(current_equity)
        
        # Get new observation
        observation = self._get_observation()
        
        # Create info dictionary
        info = self._get_info()
        
        return observation, reward, done, info
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute trading action และ return reward"""
        
        reward = 0.0
        
        # Calculate bid and ask prices
        ask_price = current_price + self.spread / 2
        bid_price = current_price - self.spread / 2
        
        if action == 1:  # Buy
            if self.position <= 0:  # ไม่มี position หรือมี short position
                if self.position < 0:  # Close short position first
                    reward += self._close_position(bid_price)
                
                # Open long position
                self.position = self.max_position_size
                self.entry_price = ask_price
                self.total_trades += 1
                
                logger.debug(f"BUY at {ask_price:.5f}, Position: {self.position}")
        
        elif action == 2:  # Sell
            if self.position >= 0:  # ไม่มี position หรือมี long position
                if self.position > 0:  # Close long position first
                    reward += self._close_position(bid_price)
                
                # Open short position
                self.position = -self.max_position_size
                self.entry_price = bid_price
                self.total_trades += 1
                
                logger.debug(f"SELL at {bid_price:.5f}, Position: {self.position}")
        
        # action == 0 (Hold) - do nothing
        
        # Update unrealized PnL
        if self.position != 0:
            if self.position > 0:  # Long position
                self.unrealized_pnl = (bid_price - self.entry_price) * self.position * 10000  # Convert to pips
            else:  # Short position
                self.unrealized_pnl = (self.entry_price - ask_price) * abs(self.position) * 10000
        else:
            self.unrealized_pnl = 0.0
        
        return reward
    
    def _close_position(self, close_price: float) -> float:
        """Close current position และ return realized PnL"""
        
        if self.position == 0:
            return 0.0
        
        # Calculate realized PnL
        if self.position > 0:  # Close long position
            pnl = (close_price - self.entry_price) * self.position * 10000  # Convert to pips
        else:  # Close short position
            pnl = (self.entry_price - close_price) * abs(self.position) * 10000
        
        # Apply commission
        pnl -= self.commission
        
        # Update balance
        self.balance += pnl
        self.realized_pnl += pnl
        
        # Track winning trades
        if pnl > 0:
            self.winning_trades += 1
        
        # Record trade
        self.trade_history.append({
            'step': self.current_step,
            'action': 'CLOSE_LONG' if self.position > 0 else 'CLOSE_SHORT',
            'price': close_price,
            'pnl': pnl,
            'balance': self.balance
        })
        
        # Reset position
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        
        logger.debug(f"Position closed, PnL: {pnl:.2f}, Balance: {self.balance:.2f}")
        
        return pnl / 100.0  # Normalize reward
    
    def _get_observation(self) -> np.ndarray:
        """สร้าง observation สำหรับ RL agent"""
        
        # Get price features window
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        price_features = self.data[self.feature_columns].iloc[start_idx:end_idx].values
        price_features = price_features.flatten()  # Flatten to 1D
        
        # Account information
        current_equity = self.balance + self.unrealized_pnl
        account_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position / self.max_position_size,  # Normalized position
            self.unrealized_pnl / self.initial_balance  # Normalized unrealized PnL
        ])
        
        # Combine all features
        observation = np.concatenate([price_features, account_features]).astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict:
        """สร้าง info dictionary สำหรับ monitoring"""
        
        current_equity = self.balance + self.unrealized_pnl
        
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'current_equity': current_equity,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'current_price': self.data['close'].iloc[self.current_step],
            'drawdown': (self.initial_balance - current_equity) / self.initial_balance
        }
        
        return info
    
    def _calculate_reward(self) -> float:
        """คำนวณ reward สำหรับ current step"""
        
        current_equity = self.balance + self.unrealized_pnl
        
        # Base reward: equity change
        if len(self.equity_curve) > 1:
            equity_change = current_equity - self.equity_curve[-2]
            reward = equity_change / self.initial_balance
        else:
            reward = 0.0
        
        # Penalty for large drawdown
        drawdown = (self.initial_balance - current_equity) / self.initial_balance
        if drawdown > 0.1:  # 10% drawdown penalty
            reward -= drawdown * 2
        
        # Bonus for consistent performance
        if len(self.equity_curve) > 10:
            recent_equity = self.equity_curve[-10:]
            equity_std = np.std(recent_equity)
            if equity_std < self.initial_balance * 0.02:  # Low volatility bonus
                reward += 0.01
        
        return reward
    
    def render(self, mode='human'):
        """แสดงสถานะปัจจุบัน"""
        
        current_equity = self.balance + self.unrealized_pnl
        current_price = self.data['close'].iloc[self.current_step]
        
        print(f"\n=== {self.symbol} {self.timeframe} Trading Environment ===")
        print(f"Step: {self.current_step}/{len(self.data)}")
        print(f"Current Price: {current_price:.5f}")
        print(f"Position: {self.position:.3f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Unrealized PnL: ${self.unrealized_pnl:.2f}")
        print(f"Current Equity: ${current_equity:.2f}")
        print(f"Return: {((current_equity/self.initial_balance)-1)*100:.2f}%")
        print(f"Trades: {self.total_trades} (Win Rate: {(self.winning_trades/max(1,self.total_trades))*100:.1f}%)")
        
        if self.position != 0:
            print(f"Entry Price: {self.entry_price:.5f}")
    
    def get_performance_metrics(self) -> Dict:
        """คำนวณ performance metrics"""
        
        if len(self.equity_curve) < 2:
            return {}
        
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        max_equity = equity_series.max()
        current_equity = equity_series.iloc[-1]
        max_drawdown = (max_equity - equity_series.min()) / max_equity
        
        # Risk metrics
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            volatility = returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
            volatility = 0
        
        metrics = {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'realized_pnl': self.realized_pnl,
            'final_equity': current_equity
        }
        
        return metrics

def create_trading_environment(symbol: str = "EURUSD",
                             timeframe: str = "M15", 
                             window_size: int = 20,
                             **kwargs) -> SingleAssetTradingEnv:
    """สร้าง Trading Environment พร้อมข้อมูล"""
    
    # Import required modules
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data.database_manager import create_database_manager
    from src.features.technical_indicators import create_indicators_calculator
    
    # Load data from database
    db = create_database_manager()
    data = db.get_price_data(symbol, timeframe)
    
    if data is None or len(data) == 0:
        raise ValueError(f"No data found for {symbol} {timeframe}")
    
    # Calculate technical indicators
    calculator = create_indicators_calculator()
    data_with_indicators = calculator.calculate_all_indicators(data)
    
    # Normalize features for RL
    normalized_data = calculator.normalize_features(data_with_indicators)
    
    # Close database
    db.close()
    
    logger.info(f"Creating environment with {len(normalized_data)} bars of data")
    
    # Create environment
    env = SingleAssetTradingEnv(
        data=normalized_data,
        symbol=symbol,
        timeframe=timeframe,
        window_size=window_size,
        **kwargs
    )
    
    return env

if __name__ == "__main__":
    # Test environment creation
    try:
        print("🧪 Testing Trading Environment...")
        
        # Create environment
        env = create_trading_environment(
            symbol="EURUSD",
            timeframe="M15",
            window_size=20,
            initial_balance=10000.0
        )
        
        print(f"✅ Environment created successfully")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.n}")
        print(f"   Data length: {len(env.data)} bars")
        
        # Test environment
        print(f"\n🔄 Testing environment...")
        obs = env.reset()
        print(f"✅ Initial observation shape: {obs.shape}")
        
        # Take a few random actions
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Equity=${info['current_equity']:.2f}")
            
            if done:
                break
        
        # Show final performance
        metrics = env.get_performance_metrics()
        print(f"\n📊 Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n🎉 Environment test completed successfully!")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()