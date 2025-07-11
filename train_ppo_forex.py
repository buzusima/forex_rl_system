# train_ppo_forex.py - Main Training Script
"""
ไฟล์หลักสำหรับเทรน PPO Agent กับระบบ Forex RL ที่สมบูรณ์
รันไฟล์นี้เพื่อเริ่มต้นการเทรนและ evaluation
"""

import sys
import os
from datetime import datetime
import logging
import argparse
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import ระบบทั้งหมด
from config.config import ForexRLConfig
from src.rl.trading_environment import create_trading_environment
from src.rl.ppo_agent import PPOConfig, create_ppo_agent
from src.rl.training_pipeline import TrainingConfig, create_training_pipeline
from src.rl.evaluation_visualization import create_evaluation_system

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PPO Forex Trading Training')
    
    # Training parameters
    parser.add_argument('--symbol', type=str, default='EURUSD', 
                       help='Trading symbol (default: EURUSD)')
    parser.add_argument('--timeframe', type=str, default='M15',
                       help='Trading timeframe (default: M15)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    
    # Model parameters
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                       help='Actor learning rate (default: 3e-4)')
    parser.add_argument('--lr-critic', type=float, default=1e-3,
                       help='Critic learning rate (default: 1e-3)')
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[256, 128, 64],
                       help='Hidden layer sizes (default: 256 128 64)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size (default: 64)')
    
    # Environment parameters
    parser.add_argument('--initial-balance', type=float, default=10000.0,
                       help='Initial trading balance (default: 10000)')
    parser.add_argument('--window-size', type=int, default=20,
                       help='Observation window size (default: 20)')
    parser.add_argument('--max-position', type=float, default=1.0,
                       help='Maximum position size (default: 1.0)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save-freq', type=int, default=100,
                       help='Model save frequency (default: 100)')
    parser.add_argument('--eval-freq', type=int, default=50,
                       help='Evaluation frequency (default: 50)')
    
    # Modes
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'continue'], 
                       default='train', help='Operation mode')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model (for continue/evaluate mode)')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to training config JSON file')
    
    return parser.parse_args()

def load_config_from_file(config_path: str):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        ppo_config = PPOConfig(**config_data.get('ppo_config', {}))
        training_config = TrainingConfig(**config_data.get('training_config', {}))
        
        return ppo_config, training_config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return None, None

def create_configs_from_args(args):
    """Create configuration objects from command line arguments"""
    
    # PPO Configuration
    ppo_config = PPOConfig(
        hidden_sizes=tuple(args.hidden_sizes),
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        batch_size=args.batch_size
    )
    
    # Training Configuration
    training_config = TrainingConfig(
        total_episodes=args.episodes,
        evaluation_frequency=args.eval_freq,
        save_frequency=args.save_freq
    )
    
    return ppo_config, training_config

def setup_directories():
    """Setup required directories"""
    directories = [
        'models',
        'results',
        'results/training',
        'results/evaluation',
        'results/plots',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def check_data_availability(symbol: str, timeframe: str):
    """Check if data is available for training"""
    try:
        # Import database manager to check data
        from src.data.database_manager import create_database_manager
        
        db = create_database_manager()
        summary = db.get_data_summary()
        db.close()
        
        if summary.get('total_records', 0) == 0:
            logger.error("❌ No data found in database")
            logger.info("💡 Please run data collection first:")
            logger.info("   python main.py")
            logger.info("   Choose option 3 or 4 to collect data")
            return False
        
        if symbol not in summary.get('symbols', []):
            logger.error(f"❌ Symbol {symbol} not found in database")
            logger.info(f"Available symbols: {summary.get('symbols', [])}")
            return False
        
        if timeframe not in summary.get('timeframes', []):
            logger.error(f"❌ Timeframe {timeframe} not found in database")
            logger.info(f"Available timeframes: {summary.get('timeframes', [])}")
            return False
        
        logger.info(f"✅ Data available for {symbol} {timeframe}")
        logger.info(f"   Total records: {summary.get('total_records', 0):,}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking data availability: {e}")
        return False

def create_environment_factory(symbol: str, timeframe: str, **env_kwargs):
    """Create environment factory function"""
    def env_creator():
        return create_trading_environment(
            symbol=symbol,
            timeframe=timeframe,
            **env_kwargs
        )
    return env_creator

def train_mode(args):
    """Training mode"""
    logger.info("🚀 Starting PPO Forex Training Mode")
    
    # Check data availability
    if not check_data_availability(args.symbol, args.timeframe):
        return False
    
    # Load or create configurations
    if args.config_path:
        ppo_config, training_config = load_config_from_file(args.config_path)
        if ppo_config is None:
            return False
    else:
        ppo_config, training_config = create_configs_from_args(args)
    
    # Create environment factory
    env_factory = create_environment_factory(
        symbol=args.symbol,
        timeframe=args.timeframe,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        max_position_size=args.max_position
    )
    
    # Test environment creation
    logger.info("🧪 Testing environment creation...")
    try:
        test_env = env_factory()
        logger.info(f"✅ Environment created successfully")
        logger.info(f"   Symbol: {test_env.symbol}")
        logger.info(f"   Timeframe: {test_env.timeframe}")
        logger.info(f"   Data length: {len(test_env.data)} bars")
        logger.info(f"   Observation space: {test_env.observation_space.shape}")
        logger.info(f"   Action space: {test_env.action_space.n}")
        test_env = None  # Clean up
    except Exception as e:
        logger.error(f"❌ Environment creation failed: {e}")
        return False
    
    # Create training pipeline
    logger.info("🔧 Setting up training pipeline...")
    pipeline = create_training_pipeline(
        env_creator=env_factory,
        ppo_config=ppo_config,
        training_config=training_config
    )
    
    # Start training
    logger.info("🎯 Starting training...")
    try:
        pipeline.train()
        logger.info("🎉 Training completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def evaluate_mode(args):
    """Evaluation mode"""
    logger.info("🔍 Starting PPO Forex Evaluation Mode")
    
    if not args.model_path or not os.path.exists(args.model_path):
        logger.error("❌ Model path not provided or doesn't exist")
        return False
    
    # Check data availability
    if not check_data_availability(args.symbol, args.timeframe):
        return False
    
    # Create environment
    logger.info("🔧 Setting up evaluation environment...")
    try:
        env = create_trading_environment(
            symbol=args.symbol,
            timeframe=args.timeframe,
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            max_position_size=args.max_position
        )
        logger.info("✅ Environment created")
    except Exception as e:
        logger.error(f"❌ Environment creation failed: {e}")
        return False
    
    # Load agent
    logger.info(f"📥 Loading model from {args.model_path}")
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = create_ppo_agent(obs_dim, action_dim)
        agent.load(args.model_path)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False
    
    # Create evaluation system
    evaluator = create_evaluation_system()
    
    # Run evaluation
    logger.info(f"🏃 Running evaluation ({args.eval_episodes} episodes)...")
    try:
        results = evaluator.evaluate_agent(
            agent=agent,
            env=env,
            n_episodes=args.eval_episodes,
            deterministic=True,
            save_results=True
        )
        
        # Print summary
        metrics = results['summary_metrics']
        logger.info("\n" + "="*60)
        logger.info("📊 EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Total Return: {metrics.total_return*100:.2f}%")
        logger.info(f"Annualized Return: {metrics.annualized_return*100:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {metrics.max_drawdown*100:.2f}%")
        logger.info(f"Win Rate: {metrics.win_rate*100:.1f}%")
        logger.info(f"Total Trades: {metrics.total_trades}")
        logger.info(f"Profit Factor: {metrics.profit_factor:.2f}")
        logger.info("="*60)
        
        logger.info("🎉 Evaluation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def continue_mode(args):
    """Continue training mode"""
    logger.info("↪️ Starting PPO Forex Continue Training Mode")
    
    if not args.model_path or not os.path.exists(args.model_path):
        logger.error("❌ Model path not provided or doesn't exist")
        return False
    
    # Check data availability
    if not check_data_availability(args.symbol, args.timeframe):
        return False
    
    # Load configurations
    if args.config_path:
        ppo_config, training_config = load_config_from_file(args.config_path)
        if ppo_config is None:
            return False
    else:
        ppo_config, training_config = create_configs_from_args(args)
    
    # Create environment factory
    env_factory = create_environment_factory(
        symbol=args.symbol,
        timeframe=args.timeframe,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        max_position_size=args.max_position
    )
    
    # Create training pipeline
    pipeline = create_training_pipeline(
        env_creator=env_factory,
        ppo_config=ppo_config,
        training_config=training_config
    )
    
    # Setup pipeline
    pipeline.setup()
    
    # Load existing model
    logger.info(f"📥 Loading model from {args.model_path}")
    try:
        pipeline.agent.load(args.model_path)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False
    
    # Continue training
    logger.info("🎯 Continuing training...")
    try:
        pipeline.train()
        logger.info("🎉 Continued training completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Continued training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup directories
    setup_directories()
    
    # Print configuration
    logger.info("\n" + "="*80)
    logger.info("🤖 PPO FOREX TRADING SYSTEM")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Initial Balance: ${args.initial_balance:,.2f}")
    logger.info("="*80)
    
    # Route to appropriate mode
    success = False
    
    if args.mode == 'train':
        success = train_mode(args)
    elif args.mode == 'evaluate':
        success = evaluate_mode(args)
    elif args.mode == 'continue':
        success = continue_mode(args)
    else:
        logger.error(f"❌ Unknown mode: {args.mode}")
    
    # Exit with appropriate code
    if success:
        logger.info("✅ Operation completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)