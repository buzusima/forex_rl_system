# execution/master_controller.py
"""
Master Controller - System Orchestrator
ควบคุมและประสานงานระหว่าง engines ทั้งหมด
รันระบบ correlation trading แบบ automated
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import threading
import time
import schedule
from dataclasses import dataclass, field
from enum import Enum
import MetaTrader5 as mt5
import json
import sqlite3

# Create logs directory if not exists
import os
os.makedirs('logs', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/master_controller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """System status types"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY = "emergency"

class TradingMode(Enum):
    """Trading mode types"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

@dataclass
class SystemConfig:
    """System configuration"""
    # Trading parameters
    trading_mode: TradingMode = TradingMode.BALANCED
    max_daily_trades: int = 10
    max_concurrent_positions: int = 5
    default_risk_percent: float = 1.0
    
    # Engine update intervals (seconds)
    data_update_interval: int = 60
    strength_update_interval: int = 300  # 5 minutes
    correlation_update_interval: int = 600  # 10 minutes
    regime_update_interval: int = 900  # 15 minutes
    position_check_interval: int = 30
    
    # Risk management
    max_account_risk: float = 5.0  # % of account
    emergency_stop_loss: float = 10.0  # % drawdown
    correlation_threshold: float = 0.7
    
    # Trading hours (UTC)
    trading_start_hour: int = 0
    trading_end_hour: int = 23
    
    # Enabled strategies
    enable_trend_following: bool = True
    enable_correlation_trading: bool = True
    enable_arbitrage: bool = True
    enable_breakout_trading: bool = True
    enable_smart_recovery: bool = True

@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    direction: str  # BUY/SELL
    strategy: str
    confidence: float
    timeframe: str
    entry_reason: str
    risk_level: str
    timestamp: datetime
    
    # Additional data
    strength_diff: float = 0.0
    correlation_data: Dict = field(default_factory=dict)
    regime_data: Dict = field(default_factory=dict)
    
class MasterController:
    """
    Master Controller - System Orchestrator
    - ประสานงานระหว่าง engines ทั้งหมด
    - สร้างและจัดการ trading signals
    - ควบคุมการทำงานของระบบทั้งหมด
    - Risk management และ monitoring
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        
        # System status
        self.status = SystemStatus.STOPPED
        self.start_time = None
        self.last_error = None
        
        # Components (will be initialized)
        self.data_manager = None
        self.strength_engine = None
        self.correlation_engine = None
        self.regime_detector = None
        self.position_manager = None
        
        # Trading state
        self.active_signals: List[TradingSignal] = []
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.emergency_stop = False
        
        # Performance tracking
        self.performance_log = []
        self.signal_history = []
        
        # Threading
        self.main_thread = None
        self.scheduler_thread = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Create logs directory
        import os
        os.makedirs('logs', exist_ok=True)
        
        logger.info("Master Controller initialized")
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.status = SystemStatus.STARTING
            logger.info("Initializing system components...")
            
            # Import and initialize components
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            sys.path.insert(0, parent_dir)
            
            from core.data_manager import DataManager
            from core.currency_strength import CurrencyStrengthEngine
            from core.correlation_engine import CorrelationEngine
            from core.market_regime import MarketRegimeDetector
            from strategy.position_manager import PositionManager
            
            logger.info("Initializing Data Manager...")
            self.data_manager = DataManager()
            if not self.data_manager.connect_mt5():
                raise Exception("Failed to connect to MT5")
            
            # Initialize historical data
            logger.info("Loading historical data...")
            self.data_manager.initialize_historical_data(days=7)
            
            # Initialize other engines
            logger.info("Initializing Currency Strength Engine...")
            self.strength_engine = CurrencyStrengthEngine(self.data_manager)
            
            logger.info("Initializing Correlation Engine...")
            self.correlation_engine = CorrelationEngine(self.data_manager, self.strength_engine)
            
            logger.info("Initializing Market Regime Detector...")
            self.regime_detector = MarketRegimeDetector(self.data_manager, self.strength_engine)
            
            logger.info("Initializing Position Manager...")
            self.position_manager = PositionManager(
                self.data_manager, self.strength_engine, 
                self.correlation_engine, self.regime_detector
            )
            
            # Start real-time data updates
            logger.info("Starting real-time data updates...")
            self.data_manager.start_real_time_updates(interval=self.config.data_update_interval)
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            self.status = SystemStatus.ERROR
            self.last_error = str(e)
            return False
    
    def start_system(self):
        """Start the trading system"""
        try:
            logger.info("Starting Master Controller...")
            
            # Initialize components
            if not self.initialize_components():
                return False
            
            # Reset daily counters
            self._reset_daily_counters()
            
            # Start main trading loop
            self.is_running = True
            self.start_time = datetime.now()
            self.status = SystemStatus.RUNNING
            
            # Schedule tasks
            self._setup_scheduler()
            
            # Start main thread
            self.main_thread = threading.Thread(target=self._main_trading_loop, daemon=True)
            self.main_thread.start()
            
            # Start scheduler thread
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("Master Controller started successfully")
            logger.info(f"Trading Mode: {self.config.trading_mode.value}")
            logger.info(f"Max Daily Trades: {self.config.max_daily_trades}")
            logger.info(f"Default Risk: {self.config.default_risk_percent}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.status = SystemStatus.ERROR
            self.last_error = str(e)
            return False
    
    def stop_system(self):
        """Stop the trading system"""
        try:
            logger.info("Stopping Master Controller...")
            
            self.is_running = False
            self.status = SystemStatus.STOPPED
            
            # Close all positions if emergency
            if self.emergency_stop:
                logger.warning("Emergency stop - closing all positions")
                self.position_manager.emergency_close_all()
            
            # Stop data updates
            if self.data_manager:
                self.data_manager.stop_real_time_updates()
            
            # Wait for threads to finish
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=5)
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            # Cleanup
            if self.data_manager:
                self.data_manager.close()
            
            logger.info("Master Controller stopped")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def _setup_scheduler(self):
        """Setup scheduled tasks"""
        schedule.clear()
        
        # Engine updates
        schedule.every(self.config.strength_update_interval).seconds.do(self._update_strength_analysis)
        schedule.every(self.config.correlation_update_interval).seconds.do(self._update_correlation_analysis)
        schedule.every(self.config.regime_update_interval).seconds.do(self._update_regime_analysis)
        
        # Position management
        schedule.every(self.config.position_check_interval).seconds.do(self._check_positions)
        
        # Daily tasks
        schedule.every().day.at("00:01").do(self._reset_daily_counters)
        schedule.every().hour.do(self._log_performance)
        
        # Risk monitoring
        schedule.every(60).seconds.do(self._monitor_risk)
        
        logger.info("Scheduler configured")
    
    def _scheduler_loop(self):
        """Scheduler loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(5)
    
    def _main_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop...")
        
        while self.is_running:
            try:
                # Check if trading is allowed
                if not self._is_trading_allowed():
                    time.sleep(60)
                    continue
                
                # Generate and process signals
                signals = self._generate_trading_signals()
                
                if signals:
                    logger.info(f"Generated {len(signals)} trading signals")
                    self._process_trading_signals(signals)
                
                # Wait before next cycle
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Main loop error: {e}")
                time.sleep(60)
        
        logger.info("Main trading loop stopped")
    
    def _is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed"""
        try:
            # Check system status
            if self.status != SystemStatus.RUNNING:
                return False
            
            # Check emergency stop
            if self.emergency_stop:
                return False
            
            # Check trading hours
            current_hour = datetime.utcnow().hour
            if not (self.config.trading_start_hour <= current_hour <= self.config.trading_end_hour):
                return False
            
            # Check daily trade limit
            if self.daily_trade_count >= self.config.max_daily_trades:
                return False
            
            # Check position limit
            if len(self.position_manager.active_positions) >= self.config.max_concurrent_positions:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trading allowance: {e}")
            return False
    
    def _generate_trading_signals(self) -> List[TradingSignal]:
        """Generate trading signals from all engines"""
        try:
            signals = []
            
            # 1. Currency Strength Signals
            if self.config.enable_trend_following:
                strength_signals = self._generate_strength_signals()
                signals.extend(strength_signals)
            
            # 2. Correlation Signals
            if self.config.enable_correlation_trading:
                correlation_signals = self._generate_correlation_signals()
                signals.extend(correlation_signals)
            
            # 3. Arbitrage Signals
            if self.config.enable_arbitrage:
                arbitrage_signals = self._generate_arbitrage_signals()
                signals.extend(arbitrage_signals)
            
            # 4. Breakout Signals
            if self.config.enable_breakout_trading:
                breakout_signals = self._generate_breakout_signals()
                signals.extend(breakout_signals)
            
            # Filter and rank signals
            filtered_signals = self._filter_and_rank_signals(signals)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return []
    
    def _generate_strength_signals(self) -> List[TradingSignal]:
        """Generate signals based on currency strength"""
        try:
            signals = []
            
            # Get strength opportunities
            opportunities = self.strength_engine.find_best_pairs_to_trade("H1")
            
            for opp in opportunities[:3]:  # Top 3 opportunities
                if opp['confidence'] > 30:
                    
                    # Get regime confirmation
                    regime_data = self.regime_detector.determine_regime(opp['pair'], "H1")
                    
                    if regime_data and regime_data.regime.value in ["strong_trend", "weak_trend"]:
                        signal = TradingSignal(
                            symbol=opp['pair'],
                            direction=opp['direction'],
                            strategy="TREND_FOLLOWING",
                            confidence=opp['confidence'],
                            timeframe="H1",
                            entry_reason=f"Currency strength: {opp['strong_currency']} vs {opp['weak_currency']}",
                            risk_level=regime_data.risk_level if hasattr(regime_data, 'risk_level') else "MEDIUM",
                            timestamp=datetime.now(),
                            strength_diff=opp['strength_diff'],
                            regime_data={'regime': regime_data.regime.value, 'confidence': regime_data.confidence}
                        )
                        
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating strength signals: {e}")
            return []
    
    def _generate_correlation_signals(self) -> List[TradingSignal]:
        """Generate signals based on correlation analysis"""
        try:
            signals = []
            
            # Get correlation breakdowns
            breakdowns = self.correlation_engine.find_correlation_breakdowns("H1")
            
            for breakdown in breakdowns[:2]:  # Top 2 breakdowns
                if abs(breakdown.historical_avg) > 0.3:  # Strong historical correlation
                    
                    # Determine direction based on strength
                    strength_ranking = self.strength_engine.get_strength_ranking("H1")
                    strength_dict = dict(strength_ranking)
                    
                    pair1_strength = self._calculate_pair_strength(breakdown.pair1, strength_dict)
                    pair2_strength = self._calculate_pair_strength(breakdown.pair2, strength_dict)
                    
                    if abs(pair1_strength - pair2_strength) > 0.5:
                        stronger_pair = breakdown.pair1 if pair1_strength > pair2_strength else breakdown.pair2
                        direction = "BUY" if stronger_pair == breakdown.pair1 else "SELL"
                        
                        signal = TradingSignal(
                            symbol=stronger_pair,
                            direction=direction,
                            strategy="CORRELATION_DIVERGENCE",
                            confidence=min(90, abs(breakdown.historical_avg - breakdown.correlation) * 100),
                            timeframe="H1",
                            entry_reason=f"Correlation breakdown: {breakdown.correlation:.3f} vs {breakdown.historical_avg:.3f}",
                            risk_level="MEDIUM",
                            timestamp=datetime.now(),
                            correlation_data={'breakdown': True, 'historical_avg': breakdown.historical_avg}
                        )
                        
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating correlation signals: {e}")
            return []
    
    def _generate_arbitrage_signals(self) -> List[TradingSignal]:
        """Generate arbitrage signals"""
        try:
            signals = []
            
            # Get arbitrage opportunities
            opportunities = self.correlation_engine.detect_arbitrage_opportunities("H1")
            
            for opp in opportunities[:2]:  # Top 2 opportunities
                if opp.confidence > 35:
                    
                    for pair, direction in opp.expected_direction.items():
                        signal = TradingSignal(
                            symbol=pair,
                            direction=direction,
                            strategy="ARBITRAGE",
                            confidence=opp.confidence,
                            timeframe="H1",
                            entry_reason=f"{opp.opportunity_type}: {opp.profit_potential:.2f} profit potential",
                            risk_level=opp.risk_level.upper(),
                            timestamp=datetime.now()
                        )
                        
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating arbitrage signals: {e}")
            return []
    
    def _generate_breakout_signals(self) -> List[TradingSignal]:
        """Generate breakout signals"""
        try:
            signals = []
            
            # Get trading opportunities from regime detector
            opportunities = self.regime_detector.get_trading_opportunities("H1")
            
            for opp in opportunities:
                if opp['strategy'] == 'BREAKOUT_TRADING' and opp['confidence'] > 65:
                    signal = TradingSignal(
                        symbol=opp['pair'],
                        direction=opp['direction'],
                        strategy="BREAKOUT_TRADING",
                        confidence=opp['confidence'],
                        timeframe="H1",
                        entry_reason=opp['entry_reason'],
                        risk_level=opp['risk_level'],
                        timestamp=datetime.now()
                    )
                    
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating breakout signals: {e}")
            return []
    
    def _calculate_pair_strength(self, pair: str, strength_dict: Dict) -> float:
        """Calculate overall pair strength"""
        try:
            base_currency = pair[:3]
            quote_currency = pair[3:]
            
            base_strength = strength_dict.get(base_currency, 0)
            quote_strength = strength_dict.get(quote_currency, 0)
            
            return base_strength - quote_strength
            
        except Exception as e:
            logger.error(f"Error calculating pair strength: {e}")
            return 0
    
    def _filter_and_rank_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter and rank signals by quality"""
        try:
            # Remove duplicate pairs
            seen_pairs = set()
            filtered_signals = []
            
            for signal in signals:
                if signal.symbol not in seen_pairs:
                    seen_pairs.add(signal.symbol)
                    filtered_signals.append(signal)
            
            # Sort by confidence
            filtered_signals.sort(key=lambda x: x.confidence, reverse=True)
            
            # Apply risk filters
            risk_filtered = []
            for signal in filtered_signals:
                if self._passes_risk_filter(signal):
                    risk_filtered.append(signal)
            
            return risk_filtered[:5]  # Top 5 signals
            
        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            return signals
    
    def _passes_risk_filter(self, signal: TradingSignal) -> bool:
        """Check if signal passes risk filters"""
        try:
            # Minimum confidence threshold
            min_confidence = {
                TradingMode.CONSERVATIVE: 80,
                TradingMode.BALANCED: 70,
                TradingMode.AGGRESSIVE: 60
            }.get(self.config.trading_mode, 70)
            
            if signal.confidence < min_confidence:
                return False
            
            # Risk level filter
            if self.config.trading_mode == TradingMode.CONSERVATIVE and signal.risk_level == "HIGH":
                return False
            
            # Check correlation exposure
            existing_pairs = [pos.symbol for pos in self.position_manager.active_positions.values()]
            for existing_pair in existing_pairs:
                correlation = self._get_pair_correlation(signal.symbol, existing_pair)
                if abs(correlation) > self.config.correlation_threshold:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk filter: {e}")
            return False
    
    def _get_pair_correlation(self, pair1: str, pair2: str) -> float:
        """Get correlation between two pairs"""
        try:
            corr_data = self.correlation_engine.calculate_correlation(pair1, pair2, "H1", periods=50)
            return corr_data.correlation if corr_data else 0.0
        except:
            return 0.0
    
    def _process_trading_signals(self, signals: List[TradingSignal]):
        """Process and execute trading signals"""
        try:
            for signal in signals:
                logger.info(f"Processing signal: {signal.direction} {signal.symbol} ({signal.strategy})")
                logger.info(f"   Confidence: {signal.confidence:.1f}% | Risk: {signal.risk_level}")
                logger.info(f"   Reason: {signal.entry_reason}")
                
                # Calculate risk percentage based on signal
                risk_percent = self._calculate_signal_risk(signal)
                
                # Execute trade
                ticket = self.position_manager.open_position(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    strategy_type=signal.strategy,
                    entry_reason=signal.entry_reason,
                    risk_percent=risk_percent
                )
                
                if ticket:
                    self.daily_trade_count += 1
                    self.signal_history.append(signal)
                    logger.info(f"Trade executed: Ticket {ticket}")
                else:
                    logger.warning(f"Failed to execute trade for {signal.symbol}")
                
                # Update active signals
                with self.lock:
                    self.active_signals.append(signal)
                
                # Respect trade frequency limits
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"Error processing trading signals: {e}")
    
    def _calculate_signal_risk(self, signal: TradingSignal) -> float:
        """Calculate risk percentage for signal"""
        base_risk = self.config.default_risk_percent
        
        # Adjust based on confidence
        confidence_multiplier = signal.confidence / 100
        
        # Adjust based on trading mode
        mode_multiplier = {
            TradingMode.CONSERVATIVE: 0.7,
            TradingMode.BALANCED: 1.0,
            TradingMode.AGGRESSIVE: 1.5
        }.get(self.config.trading_mode, 1.0)
        
        # Adjust based on risk level
        risk_multiplier = {
            "LOW": 1.2,
            "MEDIUM": 1.0,
            "HIGH": 0.6
        }.get(signal.risk_level, 1.0)
        
        return base_risk * confidence_multiplier * mode_multiplier * risk_multiplier
    
    def _update_strength_analysis(self):
        """Update currency strength analysis"""
        try:
            logger.debug(" Updating currency strength analysis...")
            self.strength_engine.calculate_all_strengths("H1")
        except Exception as e:
            logger.error(f"Error updating strength analysis: {e}")
    
    def _update_correlation_analysis(self):
        """Update correlation analysis"""
        try:
            logger.debug(" Updating correlation analysis...")
            self.correlation_engine.calculate_all_correlations("H1")
        except Exception as e:
            logger.error(f"Error updating correlation analysis: {e}")
    
    def _update_regime_analysis(self):
        """Update market regime analysis"""
        try:
            logger.debug(" Updating market regime analysis...")
            # Update regime for major pairs
            major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
            for pair in major_pairs:
                self.regime_detector.determine_regime(pair, "H1")
        except Exception as e:
            logger.error(f"Error updating regime analysis: {e}")
    
    def _check_positions(self):
        """Check and manage positions"""
        try:
            # Sync positions with MT5
            self.position_manager.sync_positions()
            
            # Check recovery needs
            if self.config.enable_smart_recovery:
                self.position_manager.check_recovery_needs()
            
            # Monitor recovery plans
            self.position_manager.monitor_recovery_plans()
            
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
    
    def _monitor_risk(self):
        """Monitor overall system risk"""
        try:
            self.position_manager.update_account_info()
            
            # Check emergency stop conditions
            if self.position_manager.account_balance > 0:
                drawdown = (self.position_manager.account_balance - self.position_manager.account_equity) / self.position_manager.account_balance * 100
                
                if drawdown > self.config.emergency_stop_loss:
                    logger.critical(f"EMERGENCY STOP: Drawdown {drawdown:.1f}% exceeds limit {self.config.emergency_stop_loss}%")
                    self.emergency_stop = True
                    self.status = SystemStatus.EMERGENCY
            
        except Exception as e:
            logger.error(f"Error monitoring risk: {e}")
    
    def _reset_daily_counters(self):
        """Reset daily counters"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
            logger.info(f"Daily counters reset for {today}")
    
    def _log_performance(self):
        """Log performance statistics"""
        try:
            summary = self.position_manager.get_portfolio_summary()
            
            performance_entry = {
                'timestamp': datetime.now(),
                'balance': summary['account']['balance'],
                'equity': summary['account']['equity'],
                'total_positions': summary['positions']['total_count'],
                'total_profit': summary['positions']['total_profit'],
                'daily_trades': self.daily_trade_count
            }
            
            self.performance_log.append(performance_entry)
            
            logger.info(f"Performance: Balance: ${summary['account']['balance']:,.2f} | "
                       f"Equity: ${summary['account']['equity']:,.2f} | "
                       f"Positions: {summary['positions']['total_count']} | "
                       f"Daily Trades: {self.daily_trade_count}")
            
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
            
            status = {
                'system': {
                    'status': self.status.value,
                    'uptime_hours': round(uptime, 2),
                    'is_running': self.is_running,
                    'emergency_stop': self.emergency_stop,
                    'last_error': self.last_error
                },
                'trading': {
                    'mode': self.config.trading_mode.value,
                    'daily_trades': self.daily_trade_count,
                    'max_daily_trades': self.config.max_daily_trades,
                    'active_signals': len(self.active_signals)
                },
                'performance': self.performance_log[-1] if self.performance_log else {},
                'positions': self.position_manager.get_portfolio_summary() if self.position_manager else {}
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

# Testing and example usage
if __name__ == "__main__":
    print("Master Controller Test")
    print("=" * 50)
    
    # Test configuration
    config = SystemConfig(
        trading_mode=TradingMode.BALANCED,
        max_daily_trades=5,
        max_concurrent_positions=3,
        default_risk_percent=0.5  # Conservative for testing
    )
    
    # Create master controller
    controller = MasterController(config)
    
    print("Master Controller created")
    print("Configuration:")
    print(f"   Trading Mode: {config.trading_mode.value}")
    print(f"   Max Daily Trades: {config.max_daily_trades}")
    print(f"   Default Risk: {config.default_risk_percent}%")
    
    print("\nTo start the system:")
    print("   controller.start_system()")
    print("\nTo stop the system:")
    print("   controller.stop_system()")
    print("\nTo check status:")
    print("   status = controller.get_system_status()")
    
    print("\nMake sure MT5 is connected before starting!")
    print("The system will run automatically once started")