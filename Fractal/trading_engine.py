import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

# Import our custom modules
from login_manager import LoginManager
from symbol_manager import SymbolManager

class SignalStrength(Enum):
    NONE = 0
    WEAK = 1
    MEDIUM = 2
    STRONG = 3

class OrderType(Enum):
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL

@dataclass
class TradingSignal:
    """Trading signal container"""
    timestamp: datetime
    signal_type: OrderType
    strength: SignalStrength
    timeframe: str
    rsi_value: float
    fractal_type: str
    confirmation_score: float
    price: float
    spread: float
    entry_reason: str

@dataclass
class Position:
    """Position container"""
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    commission: float
    open_time: datetime
    comment: str

class XAUUSDTradingEngine:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        
        # Initialize managers
        self.login_manager = LoginManager(config_file)
        self.symbol_manager = SymbolManager(config_file)
        
        # Trading state
        self.is_running = False
        self.is_connected = False
        self.current_symbol = None
        
        # Multi-timeframe data
        self.tf_data = {}
        self.tf_mapping = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        # Signal tracking
        self.last_signal = None
        self.signal_history = []
        self.positions = []
        
        # Recovery system tracking
        self.recovery_groups = {}  # Group positions by strategy
        self.recovery_levels = {}  # Track recovery level for each group
        self.position_groups = {}  # Map position tickets to groups
        
        # Performance tracking
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0.0,
            'max_drawdown': 0.0,
            'start_balance': 0.0,
            'recovery_trades': 0,
            'successful_recoveries': 0
        }
        
        # Callbacks
        self.on_signal_callback = None
        self.on_trade_callback = None
        self.on_error_callback = None
        self.on_recovery_callback = None
        
        # Setup logging
        self.setup_logging()
        
        # Initialize system
        self.initialize()
        
    def load_config(self) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Config load error: {e}")
            return {}
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('monitoring', {}).get('log_level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_engine.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TradingEngine')
    
    def initialize(self):
        """Initialize trading engine"""
        try:
            # Setup login manager callbacks
            self.login_manager.set_callbacks(
                on_login=self.on_login_event,
                on_logout=self.on_logout_event,
                on_connection_lost=self.on_connection_lost,
                on_connection_restored=self.on_connection_restored
            )
            
            # Start login monitoring
            self.login_manager.start_monitoring()
            
            # Auto-detect symbol
            symbol_name = self.config.get('trading_settings', {}).get('symbol', 'XAUUSD')
            if self.symbol_manager.auto_detect_symbol(symbol_name):
                self.current_symbol = self.symbol_manager.active_symbol.name
                self.logger.info(f"Symbol initialized: {self.current_symbol}")
            else:
                self.logger.error(f"Failed to initialize symbol: {symbol_name}")
            
            # Initialize daily stats
            account_info = mt5.account_info()
            if account_info:
                self.daily_stats['start_balance'] = account_info.balance
            
            self.logger.info("Trading engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
    
    def on_login_event(self, account_info):
        """Handle login event"""
        self.is_connected = True
        self.logger.info(f"Login detected: {account_info.login}")
        
        # Reset daily stats on new login
        self.daily_stats['start_balance'] = account_info.balance
        
        if self.on_signal_callback:
            self.on_signal_callback("LOGIN", f"Account {account_info.login} connected")
    
    def on_logout_event(self):
        """Handle logout event"""
        self.is_connected = False
        self.is_running = False
        self.logger.warning("Logout detected - Trading suspended")
        
        if self.on_signal_callback:
            self.on_signal_callback("LOGOUT", "Trading suspended")
    
    def on_connection_lost(self):
        """Handle connection lost"""
        self.is_connected = False
        self.logger.error("Connection lost")
        
        if self.on_error_callback:
            self.on_error_callback("CONNECTION_LOST", "MT5 connection lost")
    
    def on_connection_restored(self):
        """Handle connection restored"""
        self.is_connected = True
        self.logger.info("Connection restored")
        
        if self.on_signal_callback:
            self.on_signal_callback("CONNECTION_RESTORED", "MT5 connection restored")
    
    def get_timeframe_data(self, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """Get OHLC data for specific timeframe"""
        if timeframe not in self.tf_mapping:
            self.logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        try:
            tf = self.tf_mapping[timeframe]
            rates = mt5.copy_rates_from_pos(self.current_symbol, tf, 0, bars)
            
            if rates is None:
                self.logger.error(f"Failed to get {timeframe} data")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calculate indicators
            df['rsi'] = self.calculate_rsi(df)
            fractal_up, fractal_down = self.find_fractals(df)
            df['fractal_up'] = fractal_up
            df['fractal_down'] = fractal_down
            
            # Store in cache
            self.tf_data[timeframe] = df
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting {timeframe} data: {e}")
            return None
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate RSI"""
        if period is None:
            period = self.config.get('signal_parameters', {}).get('rsi_period', 14)
        
        close = data['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def find_fractals(self, data: pd.DataFrame, period: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Find Fractal Up and Down"""
        if period is None:
            period = self.config.get('signal_parameters', {}).get('fractal_period', 5)
        
        high = data['high'].values
        low = data['low'].values
        
        fractal_up = np.zeros(len(data))
        fractal_down = np.zeros(len(data))
        
        # Check from period to len(data) - 2 for recent confirmation
        end_range = max(period + 1, len(data) - 2)
        
        for i in range(period, end_range):
            # Fractal Up (Peak)
            if i >= 2 and i < len(data) - 2:
                is_peak = True
                center_high = high[i]
                
                # Check 2 bars left and 2 bars right
                for j in [i-2, i-1, i+1, i+2]:
                    if j >= 0 and j < len(data) and high[j] >= center_high:
                        is_peak = False
                        break
                
                if is_peak:
                    fractal_up[i] = center_high
            
            # Fractal Down (Valley)
            if i >= 2 and i < len(data) - 2:
                is_valley = True
                center_low = low[i]
                
                # Check 2 bars left and 2 bars right
                for j in [i-2, i-1, i+1, i+2]:
                    if j >= 0 and j < len(data) and low[j] <= center_low:
                        is_valley = False
                        break
                
                if is_valley:
                    fractal_down[i] = center_low
        
        return fractal_up, fractal_down
    
    def analyze_single_timeframe(self, timeframe: str) -> Dict:
        """Analyze single timeframe for signals"""
        data = self.get_timeframe_data(timeframe)
        if data is None or len(data) < 50:
            return {'signal': 'NO_DATA', 'strength': 0, 'rsi': 0}
        
        # Get current values
        current_idx = len(data) - 1
        current_rsi = data['rsi'].iloc[current_idx]
        current_price = data['close'].iloc[current_idx]
        
        # Get signal parameters
        rsi_up = self.config.get('signal_parameters', {}).get('rsi_up', 55)
        rsi_down = self.config.get('signal_parameters', {}).get('rsi_down', 45)
        
        # Check for recent fractals
        recent_fractal_up = False
        recent_fractal_down = False
        
        check_range = min(5, current_idx)
        for i in range(max(0, current_idx - check_range), current_idx + 1):
            if data['fractal_up'].iloc[i] > 0:
                recent_fractal_up = True
            if data['fractal_down'].iloc[i] > 0:
                recent_fractal_down = True
        
        # Determine signal
        signal = 'NONE'
        strength = 0
        fractal_type = 'NONE'
        
        if recent_fractal_down and current_rsi > rsi_up:
            signal = 'BUY'
            strength = min(3, int((current_rsi - rsi_up) / 5) + 1)
            fractal_type = 'FRACTAL_DOWN'
        elif recent_fractal_up and current_rsi < rsi_down:
            signal = 'SELL'
            strength = min(3, int((rsi_down - current_rsi) / 5) + 1)
            fractal_type = 'FRACTAL_UP'
        
        return {
            'signal': signal,
            'strength': strength,
            'rsi': current_rsi,
            'price': current_price,
            'fractal_type': fractal_type,
            'fractal_up': recent_fractal_up,
            'fractal_down': recent_fractal_down
        }
    
    def analyze_multi_timeframe(self) -> Optional[TradingSignal]:
        """Analyze multiple timeframes for signal confirmation"""
        tf_config = self.config.get('timeframes', {})
        entry_tf = tf_config.get('entry_tf', 'M1')
        trend_tf = tf_config.get('trend_tf', 'M15')
        bias_tf = tf_config.get('bias_tf', 'H1')
        require_alignment = tf_config.get('require_tf_alignment', True)
        
        # Analyze each timeframe
        entry_analysis = self.analyze_single_timeframe(entry_tf)
        trend_analysis = self.analyze_single_timeframe(trend_tf)
        bias_analysis = self.analyze_single_timeframe(bias_tf)
        
        # Log analysis
        self.logger.info(f"TF Analysis - Entry({entry_tf}): {entry_analysis['signal']}, "
                        f"Trend({trend_tf}): {trend_analysis['signal']}, "
                        f"Bias({bias_tf}): {bias_analysis['signal']}")
        
        # Check if entry signal exists
        if entry_analysis['signal'] == 'NONE':
            return None
        
        # Calculate confirmation score
        confirmation_score = 0
        tf_weights = self.config.get('timeframes', {}).get('tf_weights', {
            'entry': 0.4, 'trend': 0.35, 'bias': 0.25
        })
        
        # Entry timeframe score
        confirmation_score += entry_analysis['strength'] * tf_weights['entry']
        
        # Trend confirmation
        if not require_alignment or trend_analysis['signal'] == entry_analysis['signal']:
            confirmation_score += trend_analysis['strength'] * tf_weights['trend']
        elif trend_analysis['signal'] != 'NONE':
            confirmation_score -= 0.5  # Penalty for opposite signal
        
        # Bias confirmation  
        if not require_alignment or bias_analysis['signal'] == entry_analysis['signal']:
            confirmation_score += bias_analysis['strength'] * tf_weights['bias']
        elif bias_analysis['signal'] != 'NONE':
            confirmation_score -= 0.3  # Smaller penalty for bias
        
        # Require minimum confirmation score
        min_score = 1.0 if require_alignment else 0.5
        if confirmation_score < min_score:
            self.logger.info(f"Signal rejected - Low confirmation score: {confirmation_score:.2f}")
            return None
        
        # Get current market data
        tick = mt5.symbol_info_tick(self.current_symbol)
        if not tick:
            return None
        
        # Create trading signal
        signal_type = OrderType.BUY if entry_analysis['signal'] == 'BUY' else OrderType.SELL
        strength = SignalStrength.WEAK
        
        if confirmation_score >= 2.0:
            strength = SignalStrength.STRONG
        elif confirmation_score >= 1.5:
            strength = SignalStrength.MEDIUM
        
        # Calculate spread
        spread_points = (tick.ask - tick.bid) / self.symbol_manager.active_symbol.point
        
        trading_signal = TradingSignal(
            timestamp=datetime.now(),
            signal_type=signal_type,
            strength=strength,
            timeframe=f"{entry_tf}+{trend_tf}+{bias_tf}",
            rsi_value=entry_analysis['rsi'],
            fractal_type=entry_analysis['fractal_type'],
            confirmation_score=confirmation_score,
            price=tick.ask if signal_type == OrderType.BUY else tick.bid,
            spread=spread_points,
            entry_reason=f"Multi-TF: {entry_tf}({entry_analysis['signal']}) + "
                        f"{trend_tf}({trend_analysis['signal']}) + "
                        f"{bias_tf}({bias_analysis['signal']})"
        )
        
        return trading_signal
    
    def check_risk_conditions(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk conditions"""
        # Check connection
        if not self.is_connected:
            return False, "Not connected to MT5"
        
        # Check if trading is enabled
        if not self.config.get('trading_settings', {}).get('trading_enabled', False):
            return False, "Trading is disabled in settings"
        
        # Check daily loss limit
        daily_loss_limit = self.config.get('risk_management', {}).get('daily_loss_limit', 100)
        if abs(self.daily_stats['profit']) >= daily_loss_limit:
            return False, f"Daily loss limit reached: ${abs(self.daily_stats['profit']):.2f}"
        
        # Check max positions
        current_positions = self.get_current_positions()
        max_positions = self.config.get('trading_settings', {}).get('max_positions', 5)
        if len(current_positions) >= max_positions:
            return False, f"Maximum positions reached: {len(current_positions)}"
        
        # Check spread
        tick = mt5.symbol_info_tick(self.current_symbol)
        if tick:
            spread_points = (tick.ask - tick.bid) / self.symbol_manager.active_symbol.point
            max_spread = self.config.get('spread_management', {}).get('max_spread_points', 50)
            if spread_points > max_spread:
                return False, f"Spread too wide: {spread_points:.1f} points"
        
        # Check anti-hedge
        anti_hedge = self.config.get('trading_settings', {}).get('anti_hedge', True)
        if anti_hedge and current_positions:
            return True, "OK"  # Will check specific direction in execute_trade
        
        return True, "OK"
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        """Execute trade based on signal"""
        # Final risk check
        can_trade, risk_reason = self.check_risk_conditions()
        if not can_trade:
            self.logger.warning(f"Trade rejected: {risk_reason}")
            return False
        
        # Check for recovery opportunity first
        recovery_trade = self.check_recovery_opportunity(signal)
        if recovery_trade:
            return self.execute_recovery_trade(recovery_trade, signal)
        
        # Check anti-hedge for new position
        anti_hedge = self.config.get('trading_settings', {}).get('anti_hedge', True)
        if anti_hedge:
            positions = self.get_current_positions()
            for pos in positions:
                if signal.signal_type == OrderType.BUY and pos.type == mt5.POSITION_TYPE_SELL:
                    self.logger.warning("Anti-hedge: Cannot open BUY with SELL position open")
                    return False
                elif signal.signal_type == OrderType.SELL and pos.type == mt5.POSITION_TYPE_BUY:
                    self.logger.warning("Anti-hedge: Cannot open SELL with BUY position open")
                    return False
        
        # Execute new position
        return self.execute_new_position(signal)
    
    def execute_new_position(self, signal: TradingSignal) -> bool:
        """Execute new position (not recovery)"""
        # Get lot size
        lot_size = self.config.get('trading_settings', {}).get('lot_size', 0.01)
        
        # Create new position group
        group_id = f"group_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Execute the trade
        result = self.send_market_order(signal.signal_type, lot_size, signal, group_id, recovery_level=0)
        
        if result:
            # Initialize recovery tracking
            self.recovery_groups[group_id] = {
                'initial_signal': signal,
                'positions': [result['ticket']],
                'recovery_level': 0,
                'total_lots': lot_size,
                'target_profit': self.calculate_target_profit(lot_size),
                'creation_time': datetime.now()
            }
            
            self.position_groups[result['ticket']] = group_id
            self.recovery_levels[group_id] = 0
            
            self.logger.info(f"New position group created: {group_id}")
            
        return result is not None
    
    def check_recovery_opportunity(self, signal: TradingSignal) -> Optional[Dict]:
        """Check if this signal can be used for recovery"""
        recovery_config = self.config.get('recovery_system', {})
        
        if not recovery_config.get('enable_recovery', False):
            return None
        
        recovery_trigger = recovery_config.get('recovery_trigger_points', 100)
        max_recovery = recovery_config.get('max_recovery_levels', 3)
        smart_recovery = recovery_config.get('smart_recovery', True)
        
        # Find positions that need recovery
        for group_id, group_info in self.recovery_groups.items():
            if self.recovery_levels[group_id] >= max_recovery:
                continue
            
            # Calculate group P&L
            group_pnl_points = self.calculate_group_pnl_points(group_id)
            
            if group_pnl_points <= -recovery_trigger:
                # Check if signal direction matches for smart recovery
                initial_signal_type = group_info['initial_signal'].signal_type
                
                if smart_recovery:
                    # Must be same direction for smart recovery
                    if signal.signal_type == initial_signal_type:
                        return {
                            'group_id': group_id,
                            'current_level': self.recovery_levels[group_id],
                            'group_pnl': group_pnl_points,
                            'recovery_type': 'smart'
                        }
                else:
                    # Any direction for aggressive recovery
                    return {
                        'group_id': group_id,
                        'current_level': self.recovery_levels[group_id], 
                        'group_pnl': group_pnl_points,
                        'recovery_type': 'aggressive'
                    }
        
        return None
    
    def execute_recovery_trade(self, recovery_info: Dict, signal: TradingSignal) -> bool:
        """Execute recovery trade"""
        group_id = recovery_info['group_id']
        current_level = recovery_info['current_level']
        group_info = self.recovery_groups[group_id]
        
        recovery_config = self.config.get('recovery_system', {})
        multiplier = recovery_config.get('martingale_multiplier', 2.0)
        
        # Calculate recovery lot size
        base_lot = self.config.get('trading_settings', {}).get('lot_size', 0.01)
        recovery_lot = base_lot * (multiplier ** (current_level + 1))
        
        # Execute recovery trade
        result = self.send_market_order(
            signal.signal_type, 
            recovery_lot, 
            signal, 
            group_id, 
            recovery_level=current_level + 1
        )
        
        if result:
            # Update recovery tracking
            self.recovery_groups[group_id]['positions'].append(result['ticket'])
            self.recovery_groups[group_id]['total_lots'] += recovery_lot
            self.recovery_levels[group_id] += 1
            self.position_groups[result['ticket']] = group_id
            
            # Update target profit for dynamic TP
            if self.config.get('take_profit', {}).get('dynamic_tp', False):
                self.update_group_take_profit(group_id)
            
            # Stats
            self.daily_stats['recovery_trades'] += 1
            
            self.logger.info(f"Recovery trade executed: Group {group_id}, Level {current_level + 1}, "
                           f"Lot: {recovery_lot}, Total lots: {group_info['total_lots']}")
            
            if self.on_recovery_callback:
                self.on_recovery_callback(group_id, current_level + 1, recovery_lot, recovery_info['group_pnl'])
            
            return True
        
        return False
    
    def send_market_order(self, order_type: OrderType, volume: float, signal: TradingSignal, 
                         group_id: str, recovery_level: int = 0) -> Optional[Dict]:
        """Send market order with recovery tracking"""
        
        # Get current price
        symbol_info = mt5.symbol_info(self.current_symbol)
        if symbol_info is None:
            self.logger.error(f"Failed to get symbol info for {self.current_symbol}")
            return None
        
        if order_type == OrderType.BUY:
            price = symbol_info.ask
            action = "BUY"
            mt5_order_type = mt5.ORDER_TYPE_BUY
        else:
            price = symbol_info.bid
            action = "SELL"
            mt5_order_type = mt5.ORDER_TYPE_SELL
        
        # Prepare comment
        comment = f"Auto_{action}_{signal.timeframe}"
        if recovery_level > 0:
            comment += f"_R{recovery_level}"
        
        # Calculate TP and SL if enabled
        tp_price = 0
        sl_price = 0
        
        if self.config.get('take_profit', {}).get('enable_tp', False):
            tp_points = self.config.get('take_profit', {}).get('tp_points', 200)
            point = self.symbol_manager.active_symbol.point
            
            if order_type == OrderType.BUY:
                tp_price = price + (tp_points * point)
            else:
                tp_price = price - (tp_points * point)
        
        # Detect broker filling mode
        filling_mode = self.detect_filling_mode()
        
        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.current_symbol,
            "volume": volume,
            "type": mt5_order_type,
            "price": price,
            "deviation": self.config.get('execution_settings', {}).get('slippage_points', 20),
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        
        # Add TP if calculated
        if tp_price > 0:
            request["tp"] = tp_price
        
        # Execute order with retry logic
        for attempt in range(3):
            try:
                result = mt5.order_send(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Success
                    recovery_text = f" (Recovery L{recovery_level})" if recovery_level > 0 else ""
                    self.logger.info(f"✅ {action} executed{recovery_text}: {volume} lots at {price:.5f} "
                                   f"| RSI: {signal.rsi_value:.2f} | Group: {group_id}")
                    
                    # Update statistics
                    self.daily_stats['trades'] += 1
                    
                    # Store signal in history
                    self.signal_history.append(signal)
                    if len(self.signal_history) > 100:
                        self.signal_history.pop(0)
                    
                    # Call callback if set
                    if self.on_trade_callback:
                        self.on_trade_callback(action, volume, price, signal)
                    
                    return {
                        'ticket': result.order,
                        'price': price,
                        'volume': volume,
                        'type': order_type,
                        'group_id': group_id,
                        'recovery_level': recovery_level
                    }
                
                elif result.retcode == 10030:  # Unsupported filling mode
                    self.logger.warning(f"Filling mode {filling_mode} not supported, trying different mode...")
                    # Try different filling mode
                    if filling_mode == mt5.ORDER_FILLING_IOC:
                        request["type_filling"] = mt5.ORDER_FILLING_FOK
                    elif filling_mode == mt5.ORDER_FILLING_FOK:
                        request["type_filling"] = mt5.ORDER_FILLING_RETURN
                    else:
                        request["type_filling"] = mt5.ORDER_FILLING_IOC
                    continue
                
                else:
                    # Other error
                    self.logger.error(f"Order failed (attempt {attempt + 1}): {result.retcode} - {result.comment}")
                    if attempt < 2:  # Retry
                        time.sleep(1)
                        continue
                    else:
                        return None
                        
            except Exception as e:
                self.logger.error(f"Trade execution error (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(1)
                    continue
                else:
                    return None
        
        return None
    
    def detect_filling_mode(self) -> int:
        """Detect appropriate filling mode for broker"""
        try:
            symbol_info = mt5.symbol_info(self.current_symbol)
            if symbol_info is None:
                return mt5.ORDER_FILLING_FOK
            
            # Check what filling modes are supported
            filling_mode = symbol_info.filling_mode
            
            # ORDER_FILLING_FOK = 1
            # ORDER_FILLING_IOC = 2  
            # ORDER_FILLING_RETURN = 3
            
            if filling_mode & 2:  # IOC supported
                return mt5.ORDER_FILLING_IOC
            elif filling_mode & 1:  # FOK supported
                return mt5.ORDER_FILLING_FOK
            else:  # Default to RETURN
                return mt5.ORDER_FILLING_RETURN
                
        except Exception as e:
            self.logger.error(f"Error detecting filling mode: {e}")
            return mt5.ORDER_FILLING_FOK
    
    def calculate_group_pnl_points(self, group_id: str) -> float:
        """Calculate P&L in points for a position group"""
        if group_id not in self.recovery_groups:
            return 0.0
        
        group_info = self.recovery_groups[group_id]
        total_pnl_points = 0.0
        point = self.symbol_manager.active_symbol.point
        
        for ticket in group_info['positions']:
            position = mt5.positions_get(ticket=ticket)
            if position:
                pos = position[0]
                if pos.type == mt5.POSITION_TYPE_BUY:
                    pnl_points = (pos.price_current - pos.price_open) / point
                else:
                    pnl_points = (pos.price_open - pos.price_current) / point
                
                total_pnl_points += pnl_points * pos.volume
        
        return total_pnl_points
    
    def calculate_target_profit(self, lot_size: float) -> float:
        """Calculate target profit in dollars"""
        tp_points = self.config.get('take_profit', {}).get('tp_points', 200)
        point_value = self.symbol_manager.active_symbol.point
        contract_size = self.symbol_manager.active_symbol.contract_size
        
        return tp_points * point_value * lot_size * contract_size
    
    def update_group_take_profit(self, group_id: str):
        """Update take profit for all positions in group (dynamic TP)"""
        if group_id not in self.recovery_groups:
            return
        
        group_info = self.recovery_groups[group_id]
        target_profit = group_info['target_profit']  # Original target in dollars
        total_lots = group_info['total_lots']
        
        # Calculate new TP points
        point_value = self.symbol_manager.active_symbol.point
        contract_size = self.symbol_manager.active_symbol.contract_size
        
        new_tp_points = target_profit / (total_lots * point_value * contract_size)
        
        self.logger.info(f"Group {group_id}: Updated TP to {new_tp_points:.1f} points "
                        f"(Target: ${target_profit:.2f}, Total lots: {total_lots})")
        
        # Note: MT5 doesn't allow modifying TP of existing positions easily
        # This would require closing and reopening, or manual management
        # For now, we track the target and can use it for manual closure
        
        group_info['current_tp_points'] = new_tp_points
    
    def check_take_profit_conditions(self):
        """Check if any position groups should be closed for profit"""
        if not self.config.get('take_profit', {}).get('enable_tp', False):
            return
        
        for group_id, group_info in self.recovery_groups.items():
            # Calculate current group profit in dollars
            group_profit_usd = 0.0
            
            for ticket in group_info['positions']:
                position = mt5.positions_get(ticket=ticket)
                if position:
                    group_profit_usd += position[0].profit
            
            # Check if profit target reached
            target_profit = group_info['target_profit']
            
            if group_profit_usd >= target_profit:
                self.logger.info(f"Profit target reached for group {group_id}: "
                               f"${group_profit_usd:.2f} >= ${target_profit:.2f}")
                
                # Close all positions in group
                self.close_position_group(group_id, "PROFIT_TARGET")
    
    def close_position_group(self, group_id: str, reason: str = "MANUAL"):
        """Close all positions in a group"""
        if group_id not in self.recovery_groups:
            return
        
        group_info = self.recovery_groups[group_id]
        closed_count = 0
        total_profit = 0.0
        
        for ticket in group_info['positions']:
            position = mt5.positions_get(ticket=ticket)
            if position:
                pos = position[0]
                
                # Create close request
                if pos.type == mt5.POSITION_TYPE_BUY:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = mt5.symbol_info_tick(pos.symbol).bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    price = mt5.symbol_info_tick(pos.symbol).ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": order_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": 123456,
                    "comment": f"CLOSE_{reason}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    closed_count += 1
                    total_profit += pos.profit
                    self.logger.info(f"Closed position {pos.ticket}: ${pos.profit:.2f}")
                else:
                    self.logger.error(f"Failed to close {pos.ticket}: {result.comment}")
        
        # Update statistics
        if total_profit > 0:
            self.daily_stats['wins'] += 1
            if self.recovery_levels[group_id] > 0:
                self.daily_stats['successful_recoveries'] += 1
        else:
            self.daily_stats['losses'] += 1
        
        # Clean up tracking
        for ticket in group_info['positions']:
            if ticket in self.position_groups:
                del self.position_groups[ticket]
        
        del self.recovery_groups[group_id]
        del self.recovery_levels[group_id]
        
        self.logger.info(f"Group {group_id} closed: {closed_count} positions, "
                        f"Total profit: ${total_profit:.2f}, Reason: {reason}")
    
    def monitor_recovery_groups(self):
        """Monitor recovery groups for take profit and risk management"""
        self.check_take_profit_conditions()
        
        # Check for groups that hit maximum loss
        max_group_loss = self.config.get('risk_management', {}).get('max_group_loss', 500)
        
        for group_id, group_info in list(self.recovery_groups.items()):
            group_profit_usd = 0.0
            
            for ticket in group_info['positions']:
                position = mt5.positions_get(ticket=ticket)
                if position:
                    group_profit_usd += position[0].profit
            
            # Emergency close if loss too large
            if abs(group_profit_usd) >= max_group_loss:
                self.logger.warning(f"Emergency closing group {group_id}: "
                                  f"Loss ${abs(group_profit_usd):.2f} >= ${max_group_loss}")
                self.close_position_group(group_id, "EMERGENCY_LOSS")
    
    def get_current_positions(self) -> List[Position]:
        """Get current open positions"""
        try:
            positions = mt5.positions_get(symbol=self.current_symbol)
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                position = Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type=pos.type,
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_current=pos.price_current,
                    profit=pos.profit,
                    swap=pos.swap,
                    commission=pos.commission,
                    open_time=datetime.fromtimestamp(pos.time),
                    comment=pos.comment
                )
                result.append(position)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def update_daily_stats(self):
        """Update daily performance statistics"""
        try:
            positions = self.get_current_positions()
            
            # Calculate total profit
            total_profit = sum(pos.profit for pos in positions)
            self.daily_stats['profit'] = total_profit
            
            # Get account info for drawdown calculation
            account_info = mt5.account_info()
            if account_info:
                current_equity = account_info.equity
                start_balance = self.daily_stats['start_balance']
                if start_balance > 0:
                    drawdown_percent = ((start_balance - current_equity) / start_balance) * 100
                    if drawdown_percent > self.daily_stats['max_drawdown']:
                        self.daily_stats['max_drawdown'] = drawdown_percent
            
        except Exception as e:
            self.logger.error(f"Stats update error: {e}")
    
    def analyze_and_trade(self):
        """Main analysis and trading method"""
        try:
            # Check if system is ready
            if not self.is_connected or not self.current_symbol:
                return
            
            # Update statistics
            self.update_daily_stats()
            
            # Monitor recovery groups and TP conditions
            self.monitor_recovery_groups()
            self.monitor_take_profit_conditions()
            
            # Analyze for signals
            signal = self.analyze_multi_timeframe()
            
            if signal:
                self.logger.info(f"🎯 Signal detected: {signal.signal_type.name} "
                               f"| Strength: {signal.strength.name} "
                               f"| Score: {signal.confirmation_score:.2f}")
                
                # Check signal timing (avoid duplicate signals)
                if self.last_signal and self.last_signal.timestamp:
                    time_diff = (signal.timestamp - self.last_signal.timestamp).total_seconds()
                    min_interval = self.config.get('signal_parameters', {}).get('min_signal_interval', 60)
                    
                    if time_diff < min_interval:
                        self.logger.info(f"Signal skipped - Too soon ({time_diff:.0f}s < {min_interval}s)")
                        return
                
                # Execute trade (includes recovery logic)
                if self.execute_trade(signal):
                    self.last_signal = signal
                    
                    if self.on_signal_callback:
                        self.on_signal_callback("TRADE_EXECUTED", 
                                              f"{signal.signal_type.name} at {signal.price:.5f}")
                else:
                    if self.on_signal_callback:
                        self.on_signal_callback("TRADE_FAILED", 
                                              f"Failed to execute {signal.signal_type.name}")
            
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            if self.on_error_callback:
                self.on_error_callback("ANALYSIS_ERROR", str(e))
    
    def monitor_take_profit_conditions(self):
        """Monitor all positions for take profit conditions"""
        if not self.config.get('take_profit', {}).get('enable_tp', False):
            return
        
        try:
            # Monitor individual positions (non-recovery)
            positions = self.get_current_positions()
            individual_positions = [pos for pos in positions if pos.ticket not in self.position_groups]
            
            for pos in individual_positions:
                self.check_individual_position_tp(pos)
            
            # Monitor recovery groups
            for group_id in list(self.recovery_groups.keys()):
                self.check_group_take_profit(group_id)
                
        except Exception as e:
            self.logger.error(f"TP monitoring error: {e}")
    
    def check_individual_position_tp(self, position):
        """Check TP for individual position"""
        try:
            tp_points = self.config.get('take_profit', {}).get('tp_points', 200)
            point = self.symbol_manager.active_symbol.point
            
            # Calculate current profit in points
            if position.type == mt5.POSITION_TYPE_BUY:
                profit_points = (position.price_current - position.price_open) / point
            else:
                profit_points = (position.price_open - position.price_current) / point
            
            # Check if TP reached
            if profit_points >= tp_points:
                self.logger.info(f"TP reached for position {position.ticket}: "
                               f"{profit_points:.1f} points >= {tp_points} points")
                self.close_individual_position(position.ticket, "TAKE_PROFIT")
                
        except Exception as e:
            self.logger.error(f"Individual TP check error: {e}")
    
    def check_group_take_profit(self, group_id: str):
        """Check TP for position group"""
        if group_id not in self.recovery_groups:
            return
        
        try:
            group_info = self.recovery_groups[group_id]
            
            # Calculate total group profit in USD and points
            total_profit_usd = 0.0
            total_profit_points = 0.0
            
            for ticket in group_info['positions']:
                position = mt5.positions_get(ticket=ticket)
                if position:
                    pos = position[0]
                    total_profit_usd += pos.profit
                    
                    # Calculate profit in points
                    point = self.symbol_manager.active_symbol.point
                    if pos.type == mt5.POSITION_TYPE_BUY:
                        profit_points = (pos.price_current - pos.price_open) / point
                    else:
                        profit_points = (pos.price_open - pos.price_current) / point
                    
                    total_profit_points += profit_points * pos.volume
            
            # Check TP conditions
            target_profit = group_info.get('target_profit', 0)
            dynamic_tp = self.config.get('take_profit', {}).get('dynamic_tp', False)
            
            if dynamic_tp and 'current_tp_points' in group_info:
                # Use dynamic TP
                tp_points = group_info['current_tp_points']
                if total_profit_points >= tp_points:
                    self.logger.info(f"Dynamic TP reached for group {group_id}: "
                                   f"{total_profit_points:.1f} points >= {tp_points:.1f} points")
                    self.close_position_group(group_id, "DYNAMIC_TP")
            else:
                # Use target profit in USD
                if total_profit_usd >= target_profit:
                    self.logger.info(f"Target profit reached for group {group_id}: "
                                   f"${total_profit_usd:.2f} >= ${target_profit:.2f}")
                    self.close_position_group(group_id, "TARGET_PROFIT")
                    
        except Exception as e:
            self.logger.error(f"Group TP check error: {e}")
    
    def close_individual_position(self, ticket: int, reason: str = "MANUAL"):
        """Close individual position"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            pos = position[0]
            
            # Create close request
            if pos.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(pos.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(pos.symbol).ask
            
            # Detect filling mode
            filling_mode = self.detect_filling_mode()
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": f"CLOSE_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"✅ Closed position {pos.ticket}: ${pos.profit:.2f} ({reason})")
                
                # Update stats
                if pos.profit > 0:
                    self.daily_stats['wins'] += 1
                else:
                    self.daily_stats['losses'] += 1
                
                return True
            else:
                self.logger.error(f"❌ Failed to close {pos.ticket}: {result.comment}")
                return False
                
        except Exception as e:
            self.logger.error(f"Close position error: {e}")
            return False
    
    def emergency_close_all(self):
        """Emergency close all positions"""
        # Close all recovery groups
        for group_id in list(self.recovery_groups.keys()):
            self.close_position_group(group_id, "EMERGENCY")
        
        # Close any remaining individual positions
        positions = self.get_current_positions()
        remaining_positions = [pos for pos in positions if pos.ticket not in self.position_groups]
        
        if remaining_positions:
            self.logger.warning(f"Emergency closing {len(remaining_positions)} individual positions...")
            
            for pos in remaining_positions:
                try:
                    # Create close request
                    if pos.type == mt5.POSITION_TYPE_BUY:
                        order_type = mt5.ORDER_TYPE_SELL
                        price = mt5.symbol_info_tick(pos.symbol).bid
                    else:
                        order_type = mt5.ORDER_TYPE_BUY
                        price = mt5.symbol_info_tick(pos.symbol).ask
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": pos.symbol,
                        "volume": pos.volume,
                        "type": order_type,
                        "position": pos.ticket,
                        "price": price,
                        "deviation": 20,
                        "magic": 123456,
                        "comment": "EMERGENCY_CLOSE",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.logger.info(f"✅ Emergency closed position {pos.ticket}")
                    else:
                        self.logger.error(f"❌ Failed to close {pos.ticket}: {result.comment}")
                        
                except Exception as e:
                    self.logger.error(f"Emergency close error for {pos.ticket}: {e}")
    
    def get_recovery_status(self) -> Dict:
        """Get comprehensive recovery system status"""
        status = {
            'total_groups': len(self.recovery_groups),
            'active_recoveries': sum(1 for level in self.recovery_levels.values() if level > 0),
            'groups': []
        }
        
        for group_id, group_info in self.recovery_groups.items():
            group_pnl = 0.0
            group_positions = []
            
            for ticket in group_info['positions']:
                position = mt5.positions_get(ticket=ticket)
                if position:
                    pos = position[0]
                    group_pnl += pos.profit
                    group_positions.append({
                        'ticket': ticket,
                        'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'price_current': pos.price_current,
                        'profit': pos.profit
                    })
            
            status['groups'].append({
                'group_id': group_id,
                'recovery_level': self.recovery_levels[group_id],
                'total_lots': group_info['total_lots'],
                'target_profit': group_info['target_profit'],
                'current_pnl': group_pnl,
                'positions': group_positions,
                'creation_time': group_info['creation_time'].isoformat()
            })
        
        return status
    
    def start_trading(self):
        """Start the trading engine"""
        if self.is_running:
            self.logger.warning("Trading engine already running")
            return
        
        if not self.is_connected:
            self.logger.error("Cannot start - Not connected to MT5")
            return
        
        self.is_running = True
        self.logger.info("🚀 Trading engine started")
        
        if self.on_signal_callback:
            self.on_signal_callback("ENGINE_STARTED", "Trading engine is now active")
    
    def stop_trading(self):
        """Stop the trading engine"""
        self.is_running = False
        self.logger.info("⏹️ Trading engine stopped")
        
        if self.on_signal_callback:
            self.on_signal_callback("ENGINE_STOPPED", "Trading engine stopped")
    
    def update_settings(self, new_config: dict):
        """Update trading settings"""
        try:
            self.config.update(new_config)
            
            # Update symbol if changed
            new_symbol = self.config.get('trading_settings', {}).get('symbol')
            if new_symbol and new_symbol != self.current_symbol:
                if self.symbol_manager.switch_symbol(new_symbol):
                    self.current_symbol = self.symbol_manager.active_symbol.name
                    self.logger.info(f"Symbol updated to: {self.current_symbol}")
            
            self.logger.info("Settings updated successfully")
            
        except Exception as e:
            self.logger.error(f"Settings update error: {e}")
    
    def get_status_report(self) -> Dict:
        """Get comprehensive status report"""
        positions = self.get_current_positions()
        
        # Get current market data
        current_analysis = {}
        if self.current_symbol:
            tick = mt5.symbol_info_tick(self.current_symbol)
            if tick:
                entry_tf = self.config.get('timeframes', {}).get('entry_tf', 'M1')
                current_analysis = self.analyze_single_timeframe(entry_tf)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'engine_status': {
                'running': self.is_running,
                'connected': self.is_connected,
                'symbol': self.current_symbol
            },
            'market_data': {
                'bid': tick.bid if 'tick' in locals() and tick else 0,
                'ask': tick.ask if 'tick' in locals() and tick else 0,
                'spread': ((tick.ask - tick.bid) / self.symbol_manager.active_symbol.point) 
                         if 'tick' in locals() and tick and self.symbol_manager.active_symbol else 0
            },
            'current_analysis': current_analysis,
            'positions': {
                'count': len(positions),
                'total_profit': sum(pos.profit for pos in positions),
                'details': [
                    {
                        'ticket': pos.ticket,
                        'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                        'volume': pos.volume,
                        'profit': pos.profit
                    } for pos in positions
                ]
            },
            'daily_stats': self.daily_stats,
            'last_signal': {
                'timestamp': self.last_signal.timestamp.isoformat() if self.last_signal else None,
                'type': self.last_signal.signal_type.name if self.last_signal else None,
                'strength': self.last_signal.strength.name if self.last_signal else None
            } if self.last_signal else None
        }
    
    def set_callbacks(self, 
                     on_signal: Optional[Callable] = None,
                     on_trade: Optional[Callable] = None,
                     on_error: Optional[Callable] = None,
                     on_recovery: Optional[Callable] = None):
        """Set event callbacks"""
        self.on_signal_callback = on_signal
        self.on_trade_callback = on_trade
        self.on_error_callback = on_error
        self.on_recovery_callback = on_recovery
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_trading()
        self.login_manager.stop_monitoring()
        self.logger.info("Trading engine cleanup completed")

# Example usage and testing
if __name__ == "__main__":
    def on_signal_event(event_type, message):
        print(f"📡 Signal: {event_type} - {message}")
    
    def on_trade_event(action, volume, price, signal):
        print(f"💰 Trade: {action} {volume} lots at {price:.5f}")
    
    def on_error_event(error_type, message):
        print(f"❌ Error: {error_type} - {message}")
    
    # Initialize trading engine
    engine = XAUUSDTradingEngine()
    
    # Set callbacks
    engine.set_callbacks(
        on_signal=on_signal_event,
        on_trade=on_trade_event,
        on_error=on_error_event
    )
    
    try:
        # Start trading
        engine.start_trading()
        
        # Main trading loop
        while True:
            if engine.is_running and engine.is_connected:
                engine.analyze_and_trade()
                
                # Print status every 10 cycles
                if engine.daily_stats['trades'] % 10 == 0:
                    status = engine.get_status_report()
                    print(f"\n📊 Status Report:")
                    print(f"   Trades: {status['daily_stats']['trades']}")
                    print(f"   Profit: ${status['daily_stats']['profit']:.2f}")
                    print(f"   Positions: {status['positions']['count']}")
                    print(f"   Current RSI: {status['current_analysis'].get('rsi', 0):.2f}")
            
            # Check interval
            check_interval = engine.config.get('monitoring', {}).get('check_interval_seconds', 30)
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        engine.cleanup()