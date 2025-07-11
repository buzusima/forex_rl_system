# src/rl/training_pipeline.py - Complete Training Pipeline
"""
ไฟล์นี้สร้าง Training Pipeline สำหรับ PPO Forex Trading
ใช้ประสบการณ์จากการเทรดจริงมากกว่า 15 ปี
แก้ไขไฟล์นี้เมื่อ: ต้องการปรับ training schedule หรือ evaluation metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Import RL components
from .ppo_agent import PPOAgent, PPOConfig, create_ppo_agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training Configuration"""
    
    # Training Parameters
    total_episodes: int = 1000          # Total training episodes
    max_steps_per_episode: int = 1000   # Max steps per episode
    evaluation_frequency: int = 50      # Evaluate every N episodes
    save_frequency: int = 100           # Save model every N episodes
    
    # Early Stopping
    patience: int = 200                 # Early stopping patience
    min_improvement: float = 0.01       # Minimum improvement threshold
    
    # Market Regime Adaptation
    regime_detection: bool = True       # Enable market regime detection
    volatility_window: int = 50         # Volatility calculation window
    trend_window: int = 20              # Trend detection window
    
    # Risk Management
    max_drawdown_stop: float = 0.2      # Stop training if drawdown > 20%
    min_sharpe_ratio: float = 0.5       # Minimum acceptable Sharpe ratio
    
    # Logging and Visualization
    log_frequency: int = 10             # Log stats every N episodes
    plot_frequency: int = 100           # Update plots every N episodes
    save_plots: bool = True             # Save plots to disk
    
    # Paths
    model_save_path: str = "models/ppo_forex"
    results_save_path: str = "results/training"
    plots_save_path: str = "results/plots"

class MarketRegimeDetector:
    """Market Regime Detection สำหรับปรับ strategy"""
    
    def __init__(self, volatility_window: int = 50, trend_window: int = 20):
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.price_history = []
        
    def update(self, price: float):
        """Update price history"""
        self.price_history.append(price)
        if len(self.price_history) > max(self.volatility_window, self.trend_window) * 2:
            self.price_history = self.price_history[-max(self.volatility_window, self.trend_window) * 2:]
    
    def get_regime(self) -> Dict[str, Any]:
        """Detect current market regime"""
        if len(self.price_history) < self.volatility_window:
            return {'regime': 'insufficient_data', 'volatility': 0.0, 'trend': 'neutral'}
        
        prices = np.array(self.price_history)
        
        # Calculate volatility (rolling std of returns)
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-self.volatility_window:]) * np.sqrt(252 * 24 * 4)  # Annualized for M15
        
        # Calculate trend (price momentum)
        if len(prices) >= self.trend_window:
            short_ma = np.mean(prices[-self.trend_window//2:])
            long_ma = np.mean(prices[-self.trend_window:])
            trend_strength = (short_ma - long_ma) / long_ma
        else:
            trend_strength = 0.0
        
        # Determine regime
        if volatility > 0.002:  # High volatility threshold for EURUSD
            regime = 'high_volatility'
        elif volatility < 0.0005:  # Low volatility threshold
            regime = 'low_volatility'
        else:
            regime = 'normal'
        
        # Determine trend direction
        if abs(trend_strength) < 0.0001:
            trend = 'sideways'
        elif trend_strength > 0:
            trend = 'uptrend'
        else:
            trend = 'downtrend'
        
        return {
            'regime': regime,
            'volatility': volatility,
            'trend': trend,
            'trend_strength': trend_strength
        }

class PerformanceTracker:
    """Performance Tracking และ Analysis"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_sharpe = []
        self.episode_max_drawdown = []
        self.episode_win_rate = []
        self.equity_curves = []
        self.training_losses = []
        
        # Current episode tracking
        self.current_equity_curve = []
        self.current_trades = []
        
    def start_episode(self, initial_balance: float):
        """Start new episode tracking"""
        self.current_equity_curve = [initial_balance]
        self.current_trades = []
    
    def update_episode(self, equity: float, trade_info: Dict = None):
        """Update current episode"""
        self.current_equity_curve.append(equity)
        
        if trade_info:
            self.current_trades.append(trade_info)
    
    def finish_episode(self):
        """Finish episode and calculate metrics"""
        if len(self.current_equity_curve) < 2:
            return
        
        equity_series = pd.Series(self.current_equity_curve)
        
        # Calculate metrics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        episode_length = len(equity_series) - 1
        
        # Sharpe ratio
        returns = equity_series.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        if self.current_trades:
            winning_trades = sum(1 for trade in self.current_trades if trade.get('pnl', 0) > 0)
            win_rate = winning_trades / len(self.current_trades)
        else:
            win_rate = 0.0
        
        # Store metrics
        self.episode_returns.append(total_return)
        self.episode_lengths.append(episode_length)
        self.episode_sharpe.append(sharpe_ratio)
        self.episode_max_drawdown.append(abs(max_drawdown))
        self.episode_win_rate.append(win_rate)
        self.equity_curves.append(self.current_equity_curve.copy())
    
    def get_recent_performance(self, window: int = 10) -> Dict[str, float]:
        """Get recent performance metrics"""
        if len(self.episode_returns) < window:
            window = len(self.episode_returns)
        
        if window == 0:
            return {}
        
        recent_returns = self.episode_returns[-window:]
        recent_sharpe = self.episode_sharpe[-window:]
        recent_drawdown = self.episode_max_drawdown[-window:]
        recent_win_rate = self.episode_win_rate[-window:]
        
        return {
            'avg_return': np.mean(recent_returns),
            'std_return': np.std(recent_returns),
            'avg_sharpe': np.mean(recent_sharpe),
            'avg_drawdown': np.mean(recent_drawdown),
            'avg_win_rate': np.mean(recent_win_rate),
            'total_episodes': len(self.episode_returns)
        }

class TrainingPipeline:
    """Complete Training Pipeline สำหรับ PPO Forex Trading"""
    
    def __init__(self, 
                 env_creator: callable,
                 ppo_config: PPOConfig = None,
                 training_config: TrainingConfig = None):
        
        self.env_creator = env_creator
        self.ppo_config = ppo_config or PPOConfig()
        self.training_config = training_config or TrainingConfig()
        
        # Create directories
        os.makedirs(self.training_config.model_save_path, exist_ok=True)
        os.makedirs(self.training_config.results_save_path, exist_ok=True)
        os.makedirs(self.training_config.plots_save_path, exist_ok=True)
        
        # Initialize components
        self.env = None
        self.agent = None
        self.performance_tracker = PerformanceTracker()
        self.regime_detector = MarketRegimeDetector(
            volatility_window=self.training_config.volatility_window,
            trend_window=self.training_config.trend_window
        )
        
        # Training state
        self.current_episode = 0
        self.best_performance = float('-inf')
        self.episodes_without_improvement = 0
        self.training_start_time = None
        
        # Results storage
        self.training_results = {
            'episodes': [],
            'returns': [],
            'sharpe_ratios': [],
            'drawdowns': [],
            'win_rates': [],
            'training_losses': [],
            'market_regimes': []
        }
        
        logger.info("Training Pipeline initialized")
    
    def setup(self):
        """Setup training environment and agent"""
        # Create environment
        self.env = self.env_creator()
        
        # Create agent
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.agent = create_ppo_agent(obs_dim, action_dim, self.ppo_config)
        
        logger.info(f"Setup completed:")
        logger.info(f"  Environment: {self.env.__class__.__name__}")
        logger.info(f"  Observation dim: {obs_dim}")
        logger.info(f"  Action dim: {action_dim}")
        logger.info(f"  Agent parameters: {sum(p.numel() for p in self.agent.network.parameters()):,}")
    
    def train_episode(self) -> Dict[str, Any]:
        """Train single episode"""
        obs = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Start performance tracking
        initial_balance = self.env.balance
        self.performance_tracker.start_episode(initial_balance)
        
        while not done and episode_length < self.training_config.max_steps_per_episode:
            # Get action from agent
            action, log_prob, value = self.agent.get_action(obs)
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.store_experience(obs, action, reward, value, log_prob, done)
            
            # Update tracking
            self.performance_tracker.update_episode(info['current_equity'], info)
            self.regime_detector.update(info['current_price'])
            
            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        # Finish episode tracking
        self.performance_tracker.finish_episode()
        
        # Update agent if buffer is full
        training_stats = {}
        if self.agent.buffer.ptr >= self.agent.config.buffer_size:
            training_stats = self.agent.update()
            self.agent.training_stats['episodes'] += 1
            self.agent.training_stats['total_steps'] += episode_length
        
        # Get final performance
        final_info = self.env.get_performance_metrics()
        market_regime = self.regime_detector.get_regime()
        
        episode_result = {
            'episode': self.current_episode,
            'reward': episode_reward,
            'length': episode_length,
            'final_equity': info['current_equity'],
            'return': final_info.get('total_return', 0.0),
            'sharpe_ratio': final_info.get('sharpe_ratio', 0.0),
            'max_drawdown': final_info.get('max_drawdown', 0.0),
            'win_rate': final_info.get('win_rate', 0.0),
            'total_trades': final_info.get('total_trades', 0),
            'market_regime': market_regime,
            'training_stats': training_stats
        }
        
        return episode_result
    
    def evaluate_agent(self, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate agent performance"""
        evaluation_results = []
        
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Use deterministic action
                action, _, _ = self.agent.get_action(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
            
            # Get final performance
            performance = self.env.get_performance_metrics()
            evaluation_results.append({
                'return': performance.get('total_return', 0.0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0.0),
                'max_drawdown': performance.get('max_drawdown', 0.0),
                'win_rate': performance.get('win_rate', 0.0)
            })
        
        # Calculate average metrics
        avg_metrics = {}
        for key in evaluation_results[0].keys():
            avg_metrics[f'eval_{key}'] = np.mean([result[key] for result in evaluation_results])
            avg_metrics[f'eval_{key}_std'] = np.std([result[key] for result in evaluation_results])
        
        return avg_metrics
    
    def check_early_stopping(self, current_performance: float) -> bool:
        """Check if training should stop early"""
        if current_performance > self.best_performance + self.training_config.min_improvement:
            self.best_performance = current_performance
            self.episodes_without_improvement = 0
            return False
        else:
            self.episodes_without_improvement += 1
            
        return self.episodes_without_improvement >= self.training_config.patience
    
    def save_results(self):
        """Save training results"""
        # Save training data
        results_path = os.path.join(
            self.training_config.results_save_path,
            f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        # Save configuration
        config_path = os.path.join(
            self.training_config.results_save_path,
            f"training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        config_data = {
            'ppo_config': asdict(self.ppo_config),
            'training_config': asdict(self.training_config)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        if len(self.training_results['episodes']) < 10:
            return
        
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PPO Forex Trading - Training Progress', fontsize=16, fontweight='bold')
        
        episodes = self.training_results['episodes']
        
        # Returns
        axes[0, 0].plot(episodes, self.training_results['returns'], 'b-', alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Episode Returns')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sharpe Ratio
        axes[0, 1].plot(episodes, self.training_results['sharpe_ratios'], 'g-', alpha=0.7)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Sharpe Ratio')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max Drawdown
        axes[0, 2].plot(episodes, [-dd for dd in self.training_results['drawdowns']], 'r-', alpha=0.7)
        axes[0, 2].set_title('Max Drawdown')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Max Drawdown')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Win Rate
        axes[1, 0].plot(episodes, self.training_results['win_rates'], 'm-', alpha=0.7)
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Win Rate')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Moving Average Return (50 episodes)
        if len(episodes) >= 50:
            ma_returns = pd.Series(self.training_results['returns']).rolling(50).mean()
            axes[1, 1].plot(episodes, ma_returns, 'c-', linewidth=2, label='MA(50)')
            axes[1, 1].plot(episodes, self.training_results['returns'], 'b-', alpha=0.3, label='Episode')
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Moving Average Return')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Return')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Training Loss (if available)
        if self.training_results['training_losses']:
            loss_episodes = [i for i, loss in enumerate(self.training_results['training_losses']) if loss]
            loss_values = [loss['actor_loss'] for loss in self.training_results['training_losses'] if loss]
            
            if loss_values:
                axes[1, 2].plot(loss_episodes, loss_values, 'orange', alpha=0.7)
                axes[1, 2].set_title('Actor Loss')
                axes[1, 2].set_xlabel('Episode')
                axes[1, 2].set_ylabel('Loss')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.training_config.save_plots:
            plot_path = os.path.join(
                self.training_config.plots_save_path,
                f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training plot saved to {plot_path}")
        
        plt.show()
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting PPO Forex Training...")
        self.training_start_time = datetime.now()
        
        # Setup
        self.setup()
        
        try:
            for episode in range(self.training_config.total_episodes):
                self.current_episode = episode
                
                # Train episode
                episode_result = self.train_episode()
                
                # Store results
                self.training_results['episodes'].append(episode)
                self.training_results['returns'].append(episode_result['return'])
                self.training_results['sharpe_ratios'].append(episode_result['sharpe_ratio'])
                self.training_results['drawdowns'].append(episode_result['max_drawdown'])
                self.training_results['win_rates'].append(episode_result['win_rate'])
                self.training_results['training_losses'].append(episode_result['training_stats'])
                self.training_results['market_regimes'].append(episode_result['market_regime'])
                
                # Logging
                if episode % self.training_config.log_frequency == 0:
                    regime = episode_result['market_regime']
                    recent_perf = self.performance_tracker.get_recent_performance()
                    
                    logger.info(f"Episode {episode:4d} | "
                              f"Return: {episode_result['return']:+.4f} | "
                              f"Sharpe: {episode_result['sharpe_ratio']:+.2f} | "
                              f"DD: {episode_result['max_drawdown']:.3f} | "
                              f"WR: {episode_result['win_rate']:.2f} | "
                              f"Regime: {regime['regime']} | "
                              f"Vol: {regime['volatility']:.5f}")
                
                # Evaluation
                if episode % self.training_config.evaluation_frequency == 0 and episode > 0:
                    eval_metrics = self.evaluate_agent()
                    eval_return = eval_metrics.get('eval_return', 0.0)
                    
                    logger.info(f"🔍 Evaluation Episode {episode}:")
                    logger.info(f"   Avg Return: {eval_return:.4f} ± {eval_metrics.get('eval_return_std', 0.0):.4f}")
                    logger.info(f"   Avg Sharpe: {eval_metrics.get('eval_sharpe_ratio', 0.0):.2f}")
                    logger.info(f"   Avg Drawdown: {eval_metrics.get('eval_max_drawdown', 0.0):.3f}")
                    
                    # Check early stopping
                    if self.check_early_stopping(eval_return):
                        logger.info(f"⏹️ Early stopping triggered at episode {episode}")
                        break
                
                # Save model
                if episode % self.training_config.save_frequency == 0 and episode > 0:
                    model_path = os.path.join(
                        self.training_config.model_save_path,
                        f"ppo_model_episode_{episode}.pt"
                    )
                    self.agent.save(model_path)
                    logger.info(f"💾 Model saved at episode {episode}")
                
                # Plot progress
                if episode % self.training_config.plot_frequency == 0 and episode > 0:
                    self.plot_training_progress()
                
                # Risk management checks
                if episode_result['max_drawdown'] > self.training_config.max_drawdown_stop:
                    logger.warning(f"⚠️ High drawdown detected: {episode_result['max_drawdown']:.3f}")
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Final save and cleanup
            self.save_final_results()
    
    def save_final_results(self):
        """Save final training results"""
        # Save final model
        final_model_path = os.path.join(
            self.training_config.model_save_path,
            "ppo_model_final.pt"
        )
        self.agent.save(final_model_path)
        
        # Save results
        self.save_results()
        
        # Final plot
        self.plot_training_progress()
        
        # Training summary
        training_time = datetime.now() - self.training_start_time
        recent_perf = self.performance_tracker.get_recent_performance()
        
        logger.info("\n" + "="*80)
        logger.info("🎉 TRAINING COMPLETED")
        logger.info("="*80)
        logger.info(f"⏱️  Training Time: {training_time}")
        logger.info(f"📈 Total Episodes: {len(self.training_results['episodes'])}")
        logger.info(f"🏆 Best Performance: {self.best_performance:.4f}")
        logger.info(f"📊 Recent Performance:")
        logger.info(f"   Avg Return: {recent_perf.get('avg_return', 0.0):.4f}")
        logger.info(f"   Avg Sharpe: {recent_perf.get('avg_sharpe', 0.0):.2f}")
        logger.info(f"   Avg Drawdown: {recent_perf.get('avg_drawdown', 0.0):.3f}")
        logger.info(f"   Avg Win Rate: {recent_perf.get('avg_win_rate', 0.0):.2f}")
        logger.info("="*80)

def create_training_pipeline(env_creator: callable, 
                           ppo_config: PPOConfig = None,
                           training_config: TrainingConfig = None) -> TrainingPipeline:
    """Create Training Pipeline instance"""
    return TrainingPipeline(env_creator, ppo_config, training_config)

if __name__ == "__main__":
    # Test training pipeline
    print("🧪 Testing Training Pipeline...")
    
    # This would be replaced with actual environment creator
    def dummy_env_creator():
        # Import your actual environment here
        from trading_environment import create_trading_environment
        return create_trading_environment(symbol="EURUSD", timeframe="M15")
    
    # Create pipeline
    pipeline = create_training_pipeline(
        env_creator=dummy_env_creator,
        ppo_config=PPOConfig(),
        training_config=TrainingConfig(total_episodes=100)  # Short test
    )
    
    print("✅ Training Pipeline created successfully")
    print("Ready to start training!")