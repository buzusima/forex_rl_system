# strategy/position_manager.py
"""
Intelligent Position Manager & Smart Recovery System
แทนที่ grid recovery ด้วยระบบ correlation-based recovery
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, field
import threading
from enum import Enum
import MetaTrader5 as mt5

# Setup logging
logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    """Position status types"""
    OPEN = "open"
    CLOSED = "closed"
    HEDGED = "hedged"
    RECOVERING = "recovering"
    EMERGENCY = "emergency"

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    CORRELATION_HEDGE = "correlation_hedge"
    CURRENCY_STRENGTH_REVERSE = "currency_strength_reverse"
    CROSS_PAIR_ARBITRAGE = "cross_pair_arbitrage"
    TIME_BASED_EXIT = "time_based_exit"
    REGIME_CHANGE_EXIT = "regime_change_exit"

@dataclass
class Position:
    """Enhanced position data structure"""
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    open_price: float
    current_price: float
    profit: float
    swap: float
    commission: float
    open_time: datetime
    status: PositionStatus = PositionStatus.OPEN
    
    # Strategy information
    strategy_type: str = ""
    entry_reason: str = ""
    target_profit: float = 0.0
    max_loss: float = 0.0
    
    # Recovery information
    recovery_attempts: int = 0
    hedge_positions: List[int] = field(default_factory=list)
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_start_time: Optional[datetime] = None
    
    # Risk metrics
    initial_risk: float = 0.0
    current_risk: float = 0.0
    correlation_exposure: Dict[str, float] = field(default_factory=dict)

@dataclass
class RecoveryPlan:
    """Recovery plan structure"""
    original_position: Position
    recovery_strategy: RecoveryStrategy
    hedge_pairs: List[Tuple[str, str, float]]  # (pair, direction, volume)
    expected_profit: float
    risk_reduction: float
    time_limit: datetime
    exit_conditions: List[str]
    confidence: float

class PositionManager:
    """
    Intelligent Position Manager & Smart Recovery System
    - จัดการ positions แบบ intelligent
    - Correlation-based recovery แทน grid
    - Multi-pair hedging strategies
    - Risk-based position sizing
    """
    
    def __init__(self, data_manager, strength_engine, correlation_engine, regime_detector):
        self.data_manager = data_manager
        self.strength_engine = strength_engine
        self.correlation_engine = correlation_engine
        self.regime_detector = regime_detector
        
        # Configuration
        self.max_positions_per_pair = 1
        self.max_total_positions = 5
        self.max_correlation_exposure = 0.7  # 70% of account
        self.emergency_drawdown_threshold = 0.15  # 15%
        
        # Position tracking
        self.active_positions: Dict[int, Position] = {}
        self.recovery_plans: Dict[int, RecoveryPlan] = {}
        self.position_history: List[Position] = []
        
        # Risk tracking
        self.account_balance = 0.0
        self.account_equity = 0.0
        self.total_exposure = 0.0
        self.correlation_matrix = {}
        
        # Threading
        self.lock = threading.Lock()
        
        logger.info(" Position Manager initialized")
    
    def update_account_info(self):
        """Update account information"""
        try:
            account_info = mt5.account_info()
            if account_info:
                self.account_balance = account_info.balance
                self.account_equity = account_info.equity
                
                # Calculate total exposure
                self.total_exposure = sum(pos.volume * pos.current_price 
                                        for pos in self.active_positions.values())
                
        except Exception as e:
            logger.error(f"Error updating account info: {e}")
    
    def sync_positions(self):
        """Sync positions with MT5"""
        try:
            # Get all open positions from MT5
            mt5_positions = mt5.positions_get()
            if mt5_positions is None:
                mt5_positions = []
            
            # Update existing positions
            current_tickets = set()
            
            for mt5_pos in mt5_positions:
                ticket = mt5_pos.ticket
                current_tickets.add(ticket)
                
                # Update current price and profit
                current_price = mt5_pos.price_current
                profit = mt5_pos.profit
                
                if ticket in self.active_positions:
                    # Update existing position
                    pos = self.active_positions[ticket]
                    pos.current_price = current_price
                    pos.profit = profit
                    pos.swap = mt5_pos.swap
                    pos.commission = getattr(mt5_pos, 'commission', 0.0)
                else:
                    # Add new position (not tracked before)
                    new_position = Position(
                        ticket=ticket,
                        symbol=mt5_pos.symbol,
                        type=mt5_pos.type,
                        volume=mt5_pos.volume,
                        open_price=mt5_pos.price_open,
                        current_price=current_price,
                        profit=profit,
                        swap=mt5_pos.swap,
                        commission=getattr(mt5_pos, 'commission', 0.0),
                        open_time=datetime.fromtimestamp(mt5_pos.time),
                        entry_reason="External position"
                    )
                    
                    with self.lock:
                        self.active_positions[ticket] = new_position
            
            # Remove positions that are no longer open
            closed_tickets = set(self.active_positions.keys()) - current_tickets
            for ticket in closed_tickets:
                if ticket in self.active_positions:
                    closed_pos = self.active_positions[ticket]
                    closed_pos.status = PositionStatus.CLOSED
                    self.position_history.append(closed_pos)
                    
                    with self.lock:
                        del self.active_positions[ticket]
                        
                        # Remove recovery plan if exists
                        if ticket in self.recovery_plans:
                            del self.recovery_plans[ticket]
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    def calculate_position_size(self, symbol: str, strategy_type: str, 
                              risk_percent: float = 1.0) -> float:
        """Calculate intelligent position size based on risk and correlations"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.01
            
            # Base position size from risk percent
            account_risk = self.account_balance * (risk_percent / 100)
            
            # Get volatility for risk adjustment
            volatility = 0.015  # Default 1.5%
            try:
                regime_data = self.regime_detector.determine_regime(symbol, "H1")
                if regime_data:
                    volatility = regime_data.volatility
            except:
                pass
            
            # Calculate position size based on volatility
            pip_value = symbol_info.trade_tick_value
            risk_pips = 50  # Default risk in pips
            
            # Adjust risk based on strategy
            if strategy_type == "TREND_FOLLOWING":
                risk_pips = 100  # More room for trends
            elif strategy_type == "BREAKOUT_TRADING":
                risk_pips = 75
            elif strategy_type == "RANGE_TRADING":
                risk_pips = 30  # Tight stops for ranging
            
            # Calculate lot size
            lot_size = account_risk / (risk_pips * pip_value)
            
            # Apply correlation adjustment
            correlation_multiplier = self._calculate_correlation_multiplier(symbol)
            lot_size *= correlation_multiplier
            
            # Apply regime adjustment
            regime_multiplier = self._calculate_regime_multiplier(symbol)
            lot_size *= regime_multiplier
            
            # Ensure minimum and maximum limits
            min_lot = symbol_info.volume_min
            max_lot = min(symbol_info.volume_max, self.account_balance / 1000)  # Max 1000:1 leverage
            
            return max(min_lot, min(max_lot, round(lot_size, 2)))
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.01
    
    def _calculate_correlation_multiplier(self, symbol: str) -> float:
        """Calculate position size multiplier based on existing correlations"""
        try:
            if not self.active_positions:
                return 1.0
            
            # Get correlations with existing positions
            total_correlation_exposure = 0.0
            
            for pos in self.active_positions.values():
                if pos.symbol != symbol:
                    # Get correlation
                    corr_data = self.correlation_engine.calculate_correlation(
                        symbol, pos.symbol, "H1", periods=50
                    )
                    
                    if corr_data:
                        correlation = abs(corr_data.correlation)
                        position_exposure = pos.volume * pos.current_price / self.account_balance
                        total_correlation_exposure += correlation * position_exposure
            
            # Reduce position size if high correlation exposure
            if total_correlation_exposure > 0.5:
                return max(0.3, 1.0 - total_correlation_exposure)
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation multiplier: {e}")
            return 1.0
    
    def _calculate_regime_multiplier(self, symbol: str) -> float:
        """Calculate position size multiplier based on market regime"""
        try:
            regime_data = self.regime_detector.determine_regime(symbol, "H1")
            if not regime_data:
                return 1.0
            
            # Adjust based on regime confidence and type
            confidence_multiplier = regime_data.confidence / 100
            
            if regime_data.regime.value == "strong_trend":
                return confidence_multiplier * 1.2  # Increase size for strong trends
            elif regime_data.regime.value == "weak_trend":
                return confidence_multiplier * 1.0
            elif regime_data.regime.value == "ranging":
                return confidence_multiplier * 0.8
            elif regime_data.regime.value == "volatile":
                return confidence_multiplier * 0.5  # Reduce for volatile markets
            else:
                return confidence_multiplier * 0.7
                
        except Exception as e:
            logger.error(f"Error calculating regime multiplier: {e}")
            return 1.0
    
    def open_position(self, symbol: str, direction: str, strategy_type: str, 
                     entry_reason: str, risk_percent: float = 1.0) -> Optional[int]:
        """Open new position with intelligent sizing"""
        try:
            # Check position limits
            if len(self.active_positions) >= self.max_total_positions:
                logger.warning(f"Maximum total positions reached ({self.max_total_positions})")
                return None
            
            # Check pair-specific limits
            pair_positions = sum(1 for pos in self.active_positions.values() 
                               if pos.symbol == symbol)
            if pair_positions >= self.max_positions_per_pair:
                logger.warning(f"Maximum positions for {symbol} reached ({self.max_positions_per_pair})")
                return None
            
            # Calculate position size
            volume = self.calculate_position_size(symbol, strategy_type, risk_percent)
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"Failed to get tick for {symbol}")
                return None
            
            # Prepare order
            order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
            price = tick.ask if direction == "BUY" else tick.bid
            
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"{strategy_type}_{entry_reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to open position: {result.retcode} - {result.comment}")
                return None
            
            # Create position object
            position = Position(
                ticket=result.order,
                symbol=symbol,
                type=0 if direction == "BUY" else 1,
                volume=volume,
                open_price=price,
                current_price=price,
                profit=0.0,
                swap=0.0,
                commission=0.0,
                open_time=datetime.now(),
                strategy_type=strategy_type,
                entry_reason=entry_reason,
            )
            
            # Add to tracking
            with self.lock:
                self.active_positions[result.order] = position
            
            logger.info(f" Position opened: {direction} {volume} {symbol} @ {price:.5f}")
            logger.info(f"   Ticket: {result.order} | Strategy: {strategy_type}")
            
            return result.order
            
        except Exception as e:
            logger.error(f"Error opening position {direction} {symbol}: {e}")
            return None
    
    def check_recovery_needs(self):
        """Check if any positions need recovery action"""
        try:
            recovery_candidates = []
            
            for ticket, position in self.active_positions.items():
                # Skip if already in recovery
                if position.status == PositionStatus.RECOVERING:
                    continue
                
                # Check if position needs recovery
                if self._needs_recovery(position):
                    recovery_candidates.append(position)
            
            # Process recovery candidates
            for position in recovery_candidates:
                recovery_plan = self._create_recovery_plan(position)
                if recovery_plan:
                    self._execute_recovery_plan(recovery_plan)
                    
        except Exception as e:
            logger.error(f"Error checking recovery needs: {e}")
    
    def _needs_recovery(self, position: Position) -> bool:
        """Determine if position needs recovery"""
        try:
            # Time-based check
            position_age = (datetime.now() - position.open_time).total_seconds() / 3600  # hours
            
            # Loss-based check
            loss_percent = abs(position.profit) / self.account_balance * 100
            
            # Recovery criteria
            needs_recovery = False
            
            # 1. Significant loss (more than 2% of account)
            if loss_percent > 2.0:
                needs_recovery = True
            
            # 2. Long time in loss (more than 24 hours)
            if position_age > 24 and position.profit < 0:
                needs_recovery = True
            
            # 3. Market regime change against position
            try:
                regime_data = self.regime_detector.determine_regime(position.symbol, "H1")
                if regime_data:
                    # If we're long but regime is bearish (or vice versa)
                    if ((position.type == 0 and regime_data.trend_direction == -1) or
                        (position.type == 1 and regime_data.trend_direction == 1)) and \
                       regime_data.confidence > 70:
                        needs_recovery = True
            except:
                pass
            
            return needs_recovery
            
        except Exception as e:
            logger.error(f"Error checking if position needs recovery: {e}")
            return False
    
    def _create_recovery_plan(self, position: Position) -> Optional[RecoveryPlan]:
        """Create intelligent recovery plan"""
        try:
            logger.info(f" Creating recovery plan for {position.symbol} ticket {position.ticket}")
            
            # Analyze market situation
            regime_data = self.regime_detector.determine_regime(position.symbol, "H1")
            strength_ranking = self.strength_engine.get_strength_ranking("H1")
            
            # Determine best recovery strategy
            recovery_strategy = self._select_recovery_strategy(position, regime_data)
            
            if recovery_strategy == RecoveryStrategy.CORRELATION_HEDGE:
                return self._create_correlation_hedge_plan(position)
            elif recovery_strategy == RecoveryStrategy.CURRENCY_STRENGTH_REVERSE:
                return self._create_strength_reverse_plan(position, strength_ranking)
            elif recovery_strategy == RecoveryStrategy.TIME_BASED_EXIT:
                return self._create_time_exit_plan(position)
            elif recovery_strategy == RecoveryStrategy.REGIME_CHANGE_EXIT:
                return self._create_regime_exit_plan(position)
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating recovery plan: {e}")
            return None
    
    def _select_recovery_strategy(self, position: Position, regime_data) -> RecoveryStrategy:
        """Select best recovery strategy based on market conditions"""
        
        # If strong trend against us - exit or hedge
        if regime_data and regime_data.regime.value == "strong_trend":
            if ((position.type == 0 and regime_data.trend_direction == -1) or
                (position.type == 1 and regime_data.trend_direction == 1)):
                return RecoveryStrategy.REGIME_CHANGE_EXIT
        
        # If ranging market - try correlation hedge
        if regime_data and regime_data.regime.value in ["ranging", "consolidation"]:
            return RecoveryStrategy.CORRELATION_HEDGE
        
        # If currency strength has shifted significantly
        return RecoveryStrategy.CURRENCY_STRENGTH_REVERSE
    
    def _create_correlation_hedge_plan(self, position: Position) -> Optional[RecoveryPlan]:
        """Create correlation-based hedge plan"""
        try:
            # Find highly correlated pairs
            correlations = self.correlation_engine.get_cached_correlations("H1")
            if not correlations:
                correlations = self.correlation_engine.calculate_all_correlations("H1")
            
            hedge_pairs = []
            
            for (pair1, pair2), corr_data in correlations.items():
                if pair1 == position.symbol and abs(corr_data.correlation) > 0.7:
                    # High correlation - hedge with opposite direction
                    hedge_direction = "SELL" if position.type == 0 else "BUY"
                    hedge_volume = position.volume * 0.7  # Partial hedge
                    
                    hedge_pairs.append((pair2, hedge_direction, hedge_volume))
                    
                    if len(hedge_pairs) >= 2:  # Limit hedge pairs
                        break
            
            if hedge_pairs:
                return RecoveryPlan(
                    original_position=position,
                    recovery_strategy=RecoveryStrategy.CORRELATION_HEDGE,
                    hedge_pairs=hedge_pairs,
                    expected_profit=position.volume * 50 * 0.0001,  # Estimate
                    risk_reduction=0.6,  # 60% risk reduction
                    time_limit=datetime.now() + timedelta(hours=48),
                    exit_conditions=["correlation_restored", "profit_target_reached"],
                    confidence=75.0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating correlation hedge plan: {e}")
            return None
    
    def _create_strength_reverse_plan(self, position: Position, 
                                    strength_ranking: List) -> Optional[RecoveryPlan]:
        """Create currency strength reversal plan"""
        try:
            # Find currency strength opportunities
            base_currency = position.symbol[:3]
            quote_currency = position.symbol[3:]
            
            # Get current strength
            strength_dict = dict(strength_ranking)
            base_strength = strength_dict.get(base_currency, 0)
            quote_strength = strength_dict.get(quote_currency, 0)
            
            # If our position is against current strength, create reversal plan
            if position.type == 0 and base_strength < quote_strength:  # We're long but base is weak
                # Look for strong currency to pair with quote
                for currency, strength in strength_ranking[:3]:
                    if currency != base_currency and currency != quote_currency:
                        hedge_pair = f"{currency}{quote_currency}"
                        if hedge_pair in self.data_manager.currency_pairs:
                            hedge_pairs = [(hedge_pair, "BUY", position.volume * 0.8)]
                            break
                else:
                    return None
                    
            elif position.type == 1 and base_strength > quote_strength:  # We're short but base is strong
                # Look for weak currency to pair with base
                for currency, strength in reversed(strength_ranking[-3:]):
                    if currency != base_currency and currency != quote_currency:
                        hedge_pair = f"{base_currency}{currency}"
                        if hedge_pair in self.data_manager.currency_pairs:
                            hedge_pairs = [(hedge_pair, "BUY", position.volume * 0.8)]
                            break
                else:
                    return None
            else:
                return None
            
            return RecoveryPlan(
                original_position=position,
                recovery_strategy=RecoveryStrategy.CURRENCY_STRENGTH_REVERSE,
                hedge_pairs=hedge_pairs,
                expected_profit=position.volume * 75 * 0.0001,
                risk_reduction=0.7,
                time_limit=datetime.now() + timedelta(hours=24),
                exit_conditions=["strength_alignment", "profit_target_reached"],
                confidence=80.0
            )
            
        except Exception as e:
            logger.error(f"Error creating strength reverse plan: {e}")
            return None
    
    def _create_time_exit_plan(self, position: Position) -> RecoveryPlan:
        """Create time-based exit plan"""
        return RecoveryPlan(
            original_position=position,
            recovery_strategy=RecoveryStrategy.TIME_BASED_EXIT,
            hedge_pairs=[],
            expected_profit=0.0,
            risk_reduction=1.0,  # 100% - position will be closed
            time_limit=datetime.now() + timedelta(hours=1),
            exit_conditions=["time_limit_reached"],
            confidence=100.0
        )
    
    def _create_regime_exit_plan(self, position: Position) -> RecoveryPlan:
        """Create regime change exit plan"""
        return RecoveryPlan(
            original_position=position,
            recovery_strategy=RecoveryStrategy.REGIME_CHANGE_EXIT,
            hedge_pairs=[],
            expected_profit=0.0,
            risk_reduction=1.0,
            time_limit=datetime.now() + timedelta(minutes=30),
            exit_conditions=["immediate_exit"],
            confidence=90.0
        )
    
    def _execute_recovery_plan(self, plan: RecoveryPlan):
        """Execute recovery plan"""
        try:
            logger.info(f"🚀 Executing recovery plan: {plan.recovery_strategy.value}")
            logger.info(f"   Original position: {plan.original_position.symbol} ticket {plan.original_position.ticket}")
            logger.info(f"   Expected risk reduction: {plan.risk_reduction*100:.0f}%")
            
            # Mark original position as recovering
            plan.original_position.status = PositionStatus.RECOVERING
            plan.original_position.recovery_strategy = plan.recovery_strategy
            plan.original_position.recovery_start_time = datetime.now()
            
            # Execute based on strategy type
            if plan.recovery_strategy in [RecoveryStrategy.TIME_BASED_EXIT, RecoveryStrategy.REGIME_CHANGE_EXIT]:
                # Close original position
                self._close_position(plan.original_position.ticket, "Recovery: " + plan.recovery_strategy.value)
                
            else:
                # Open hedge positions
                for hedge_pair, direction, volume in plan.hedge_pairs:
                    hedge_ticket = self.open_position(
                        symbol=hedge_pair,
                        direction=direction,
                        strategy_type="RECOVERY_HEDGE",
                        entry_reason=f"Hedge for {plan.original_position.symbol}",
                        risk_percent=0.5  # Lower risk for hedge
                    )
                    
                    if hedge_ticket:
                        plan.original_position.hedge_positions.append(hedge_ticket)
                        logger.info(f"    Hedge opened: {direction} {volume} {hedge_pair}")
                
                # Store recovery plan
                with self.lock:
                    self.recovery_plans[plan.original_position.ticket] = plan
            
        except Exception as e:
            logger.error(f"Error executing recovery plan: {e}")
    
    def _close_position(self, ticket: int, reason: str = "") -> bool:
        """Close position by ticket"""
        try:
            # Get position info
            position = self.active_positions.get(ticket)
            if not position:
                logger.warning(f"Position {ticket} not found")
                return False
            
            # Get current tick
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                logger.error(f"Failed to get tick for {position.symbol}")
                return False
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if position.type == 0 else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Close: {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Send close request
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f" Position closed: {ticket} - {reason}")
                return True
            else:
                logger.error(f"❌ Failed to close position {ticket}: {result.comment}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def monitor_recovery_plans(self):
        """Monitor and manage active recovery plans"""
        try:
            for ticket, plan in list(self.recovery_plans.items()):
                # Check if plan has expired
                if datetime.now() > plan.time_limit:
                    logger.info(f"⏰ Recovery plan expired for ticket {ticket}")
                    self._cleanup_recovery_plan(ticket)
                    continue
                
                # Check exit conditions
                if self._check_recovery_exit_conditions(plan):
                    logger.info(f" Recovery plan successful for ticket {ticket}")
                    self._cleanup_recovery_plan(ticket)
                    continue
                    
        except Exception as e:
            logger.error(f"Error monitoring recovery plans: {e}")
    
    def _check_recovery_exit_conditions(self, plan: RecoveryPlan) -> bool:
        """Check if recovery plan exit conditions are met"""
        try:
            # Check if original position is profitable
            original_pos = plan.original_position
            if original_pos.ticket in self.active_positions:
                current_pos = self.active_positions[original_pos.ticket]
                if current_pos.profit > 0:
                    return True
            
            # Check correlation restoration for correlation hedge
            if plan.recovery_strategy == RecoveryStrategy.CORRELATION_HEDGE:
                for hedge_pair, _, _ in plan.hedge_pairs:
                    corr_data = self.correlation_engine.calculate_correlation(
                        original_pos.symbol, hedge_pair, "H1", periods=20
                    )
                    if corr_data and abs(corr_data.correlation) > 0.7:
                        return True
            
            # Add more exit condition checks as needed
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking recovery exit conditions: {e}")
            return False
    
    def _cleanup_recovery_plan(self, ticket: int):
        """Clean up completed recovery plan"""
        try:
            if ticket in self.recovery_plans:
                plan = self.recovery_plans[ticket]
                
                # Close hedge positions
                for hedge_ticket in plan.original_position.hedge_positions:
                    if hedge_ticket in self.active_positions:
                        self._close_position(hedge_ticket, "Recovery cleanup")
                
                # Update original position status
                if ticket in self.active_positions:
                    self.active_positions[ticket].status = PositionStatus.OPEN
                    self.active_positions[ticket].recovery_strategy = None
                
                # Remove recovery plan
                with self.lock:
                    del self.recovery_plans[ticket]
                    
        except Exception as e:
            logger.error(f"Error cleaning up recovery plan: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            self.update_account_info()
            
            total_profit = sum(pos.profit for pos in self.active_positions.values())
            total_volume = sum(pos.volume for pos in self.active_positions.values())
            
            # Count positions by status
            status_counts = {}
            for pos in self.active_positions.values():
                status_counts[pos.status.value] = status_counts.get(pos.status.value, 0) + 1
            
            # Calculate risk metrics
            total_risk = sum(pos.current_risk for pos in self.active_positions.values())
            risk_percent = (total_risk / self.account_balance * 100) if self.account_balance > 0 else 0
            
            return {
                'timestamp': datetime.now(),
                'account': {
                    'balance': self.account_balance,
                    'equity': self.account_equity,
                    'margin_used': self.total_exposure,
                    'free_margin': self.account_equity - self.total_exposure
                },
                'positions': {
                    'total_count': len(self.active_positions),
                    'total_profit': total_profit,
                    'total_volume': total_volume,
                    'status_breakdown': status_counts
                },
                'risk': {
                    'total_risk': total_risk,
                    'risk_percent': risk_percent,
                    'max_drawdown_threshold': self.emergency_drawdown_threshold * 100
                },
                'recovery': {
                    'active_plans': len(self.recovery_plans),
                    'recovery_positions': status_counts.get('recovering', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}

# Testing and example usage
if __name__ == "__main__":
    print("💼 Position Manager Test")
    print("=" * 50)
    
    # Note: This would need actual MT5 connection and other engines
    # This is just a structural test
    
    print(" Position Manager module loaded successfully")
    print("📋 Features available:")
    print("   - Intelligent position sizing")
    print("   - Correlation-based recovery")
    print("   - Multi-strategy support")
    print("   - Risk management")
    print("   - Recovery plan monitoring")
    
    print("\n🔧 To use with full system:")
    print("   from strategy.position_manager import PositionManager")
    print("   pm = PositionManager(data_manager, strength_engine, correlation_engine, regime_detector)")
    print("   pm.open_position('EURUSD', 'BUY', 'TREND_FOLLOWING', 'Strong trend signal')")
    print("   pm.check_recovery_needs()")