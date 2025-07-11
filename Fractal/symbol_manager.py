import MetaTrader5 as mt5
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SymbolInfo:
    """Symbol information container"""
    name: str
    display_name: str
    point: float
    digits: int
    trade_mode: int
    min_lot: float
    max_lot: float
    lot_step: float
    contract_size: float
    currency_base: str
    currency_profit: str
    currency_margin: str
    margin_initial: float
    spread: int
    stops_level: int
    freeze_level: int
    is_available: bool
    is_tradeable: bool
    broker_suffix: str = ""

class SymbolManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        
        # Symbol variations database
        self.symbol_variations = {
            'XAUUSD': {
                'primary': ['XAUUSD', 'GOLD'],
                'suffixes': ['.m', '.raw', '.micro', '.mini', '.', '-'],
                'prefixes': ['#', '_', ''],
                'alternatives': ['GOLD_USD', 'XAU_USD', 'XAUUSD_', 'GLD']
            },
            'EURUSD': {
                'primary': ['EURUSD'],
                'suffixes': ['.m', '.raw', '.micro', '.mini', '.', '-'],
                'prefixes': ['#', '_', ''],
                'alternatives': ['EUR_USD', 'EURUSD_']
            },
            'GBPUSD': {
                'primary': ['GBPUSD'],
                'suffixes': ['.m', '.raw', '.micro', '.mini', '.', '-'],
                'prefixes': ['#', '_', ''],
                'alternatives': ['GBP_USD', 'GBPUSD_']
            }
        }
        
        # Detected symbols cache
        self.detected_symbols = {}
        self.active_symbol = None
        self.symbol_specs = None
        
        # Broker detection
        self.broker_info = None
        self.broker_type = None
        
        # Logging
        self.setup_logging()
        
        # Initialize symbol detection
        self.detect_broker_info()
    
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
        
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger('SymbolManager')
    
    def detect_broker_info(self) -> Dict[str, any]:
        """Detect broker information and type"""
        try:
            account_info = mt5.account_info()
            terminal_info = mt5.terminal_info()
            
            if account_info and terminal_info:
                self.broker_info = {
                    'company': account_info.company,
                    'server': account_info.server,
                    'terminal_company': terminal_info.company,
                    'terminal_build': terminal_info.build,
                    'path': terminal_info.path
                }
                
                # Detect broker type based on patterns
                company_name = account_info.company.lower()
                server_name = account_info.server.lower()
                
                if any(x in company_name for x in ['meta', 'metaquotes']):
                    self.broker_type = 'MetaQuotes'
                elif any(x in company_name for x in ['admiral', 'admirals']):
                    self.broker_type = 'Admirals'
                elif any(x in company_name for x in ['ic markets', 'icmarkets']):
                    self.broker_type = 'ICMarkets'
                elif any(x in company_name for x in ['pepperstone']):
                    self.broker_type = 'Pepperstone'
                elif any(x in company_name for x in ['oanda']):
                    self.broker_type = 'OANDA'
                elif any(x in company_name for x in ['fxpro']):
                    self.broker_type = 'FxPro'
                elif any(x in company_name for x in ['exness']):
                    self.broker_type = 'Exness'
                else:
                    self.broker_type = 'Generic'
                
                self.logger.info(f"Broker detected: {self.broker_type} ({account_info.company})")
                return self.broker_info
                
        except Exception as e:
            self.logger.error(f"Broker detection error: {e}")
            
        return {}
    
    def generate_symbol_variations(self, base_symbol: str) -> List[str]:
        """Generate all possible symbol variations"""
        variations = []
        
        if base_symbol not in self.symbol_variations:
            # If unknown symbol, create basic variations
            variations = [base_symbol]
            for suffix in ['.m', '.raw', '.micro', '.']:
                variations.append(f"{base_symbol}{suffix}")
            for prefix in ['#', '_']:
                variations.append(f"{prefix}{base_symbol}")
            return variations
        
        symbol_config = self.symbol_variations[base_symbol]
        
        # Primary names
        for primary in symbol_config['primary']:
            variations.append(primary)
            
            # Add suffixes
            for suffix in symbol_config['suffixes']:
                if suffix:  # Skip empty suffix
                    variations.append(f"{primary}{suffix}")
            
            # Add prefixes
            for prefix in symbol_config['prefixes']:
                if prefix:  # Skip empty prefix
                    variations.append(f"{prefix}{primary}")
                    
                    # Prefix + suffix combinations
                    for suffix in symbol_config['suffixes']:
                        if suffix:
                            variations.append(f"{prefix}{primary}{suffix}")
        
        # Add alternatives
        variations.extend(symbol_config['alternatives'])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(variations))
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get detailed symbol information"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            # Check if symbol is available for trading
            if not symbol_info.visible:
                # Try to enable symbol
                if not mt5.symbol_select(symbol, True):
                    self.logger.warning(f"Could not enable symbol: {symbol}")
                    return None
                # Refresh symbol info
                symbol_info = mt5.symbol_info(symbol)
            
            # Detect broker suffix
            broker_suffix = ""
            base_name = symbol
            for base in ['XAUUSD', 'GOLD', 'EURUSD', 'GBPUSD']:
                if symbol.startswith(base):
                    broker_suffix = symbol[len(base):]
                    base_name = base
                    break
                elif symbol.startswith('#' + base) or symbol.startswith('_' + base):
                    broker_suffix = symbol[len(base) + 1:]
                    base_name = base
                    break
            
            return SymbolInfo(
                name=symbol,
                display_name=base_name,
                point=symbol_info.point,
                digits=symbol_info.digits,
                trade_mode=symbol_info.trade_mode,
                min_lot=symbol_info.volume_min,
                max_lot=symbol_info.volume_max,
                lot_step=symbol_info.volume_step,
                contract_size=symbol_info.trade_contract_size,
                currency_base=symbol_info.currency_base,
                currency_profit=symbol_info.currency_profit,
                currency_margin=symbol_info.currency_margin,
                margin_initial=symbol_info.margin_initial,
                spread=symbol_info.spread,
                stops_level=getattr(symbol_info, 'stops_level', 0),
                freeze_level=getattr(symbol_info, 'freeze_level', 0),
                is_available=symbol_info.visible,
                is_tradeable=symbol_info.trade_mode in [mt5.SYMBOL_TRADE_MODE_FULL, mt5.SYMBOL_TRADE_MODE_LONGONLY, mt5.SYMBOL_TRADE_MODE_SHORTONLY],
                broker_suffix=broker_suffix
            )
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def find_working_symbol(self, base_symbol: str) -> Optional[SymbolInfo]:
        """Find the first working variation of a symbol"""
        variations = self.generate_symbol_variations(base_symbol)
        
        self.logger.info(f"Searching for {base_symbol} variations: {variations}")
        
        for variation in variations:
            symbol_info = self.get_symbol_info(variation)
            if symbol_info and symbol_info.is_tradeable:
                self.logger.info(f"Found working symbol: {variation}")
                self.detected_symbols[base_symbol] = symbol_info
                return symbol_info
        
        self.logger.error(f"No working symbol found for {base_symbol}")
        return None
    
    def auto_detect_symbol(self, preferred_symbol: str = None) -> bool:
        """Auto-detect and set active trading symbol"""
        target_symbol = preferred_symbol or self.config.get('trading_settings', {}).get('symbol', 'XAUUSD')
        
        self.logger.info(f"Auto-detecting symbol for: {target_symbol}")
        
        # First try exact match
        symbol_info = self.get_symbol_info(target_symbol)
        if symbol_info and symbol_info.is_tradeable:
            self.active_symbol = symbol_info
            self.symbol_specs = symbol_info
            self.logger.info(f" Direct match found: {target_symbol}")
            return True
        
        # Try variations
        symbol_info = self.find_working_symbol(target_symbol)
        if symbol_info:
            self.active_symbol = symbol_info
            self.symbol_specs = symbol_info
            return True
        
        return False
    
    def get_all_available_symbols(self, pattern: str = None) -> List[str]:
        """Get all available symbols matching pattern"""
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                return []
            
            symbol_names = [s.name for s in symbols]
            
            if pattern:
                pattern_re = re.compile(pattern, re.IGNORECASE)
                symbol_names = [s for s in symbol_names if pattern_re.search(s)]
            
            return sorted(symbol_names)
            
        except Exception as e:
            self.logger.error(f"Error getting symbols: {e}")
            return []
    
    def search_gold_symbols(self) -> List[SymbolInfo]:
        """Search for all gold-related symbols"""
        gold_patterns = ['XAU', 'GOLD', 'GLD']
        found_symbols = []
        
        for pattern in gold_patterns:
            symbols = self.get_all_available_symbols(pattern)
            for symbol in symbols:
                symbol_info = self.get_symbol_info(symbol)
                if symbol_info and symbol_info.is_tradeable:
                    found_symbols.append(symbol_info)
        
        return found_symbols
    
    def validate_symbol_for_trading(self, symbol_name: str = None) -> Tuple[bool, str]:
        """Validate if symbol is ready for trading"""
        if symbol_name:
            symbol_info = self.get_symbol_info(symbol_name)
        else:
            symbol_info = self.active_symbol
        
        if not symbol_info:
            return False, "Symbol not found"
        
        if not symbol_info.is_available:
            return False, "Symbol not available"
        
        if not symbol_info.is_tradeable:
            return False, f"Symbol not tradeable (mode: {symbol_info.trade_mode})"
        
        if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return False, "Trading disabled for symbol"
        
        if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
            return False, "Close only mode"
        
        # Check market hours
        try:
            tick = mt5.symbol_info_tick(symbol_info.name)
            if tick is None:
                return False, "No price quotes available"
            
            if tick.bid == 0 or tick.ask == 0:
                return False, "Invalid price quotes"
                
        except Exception as e:
            return False, f"Price check error: {e}"
        
        return True, "Symbol ready for trading"
    
    def get_symbol_specs_report(self) -> Dict[str, any]:
        """Get comprehensive symbol specifications report"""
        if not self.active_symbol:
            return {"error": "No active symbol"}
        
        symbol = self.active_symbol
        
        try:
            tick = mt5.symbol_info_tick(symbol.name)
            
            return {
                'symbol_info': {
                    'name': symbol.name,
                    'display_name': symbol.display_name,
                    'broker_suffix': symbol.broker_suffix,
                    'digits': symbol.digits,
                    'point': symbol.point
                },
                'trading_specs': {
                    'min_lot': symbol.min_lot,
                    'max_lot': symbol.max_lot,
                    'lot_step': symbol.lot_step,
                    'contract_size': symbol.contract_size,
                    'margin_initial': symbol.margin_initial
                },
                'market_data': {
                    'bid': tick.bid if tick else 0,
                    'ask': tick.ask if tick else 0,
                    'spread': symbol.spread,
                    'spread_points': (tick.ask - tick.bid) / symbol.point if tick and tick.bid > 0 else 0
                },
                'trading_conditions': {
                    'stops_level': symbol.stops_level,
                    'freeze_level': symbol.freeze_level,
                    'trade_mode': symbol.trade_mode,
                    'is_tradeable': symbol.is_tradeable
                },
                'currencies': {
                    'base': symbol.currency_base,
                    'profit': symbol.currency_profit,
                    'margin': symbol.currency_margin
                },
                'broker_info': self.broker_info,
                'validation': self.validate_symbol_for_trading()
            }
            
        except Exception as e:
            return {"error": f"Report generation failed: {e}"}
    
    def switch_symbol(self, new_symbol: str) -> bool:
        """Switch to a new trading symbol"""
        self.logger.info(f"Switching symbol to: {new_symbol}")
        
        if self.auto_detect_symbol(new_symbol):
            # Update config
            if 'trading_settings' not in self.config:
                self.config['trading_settings'] = {}
            self.config['trading_settings']['symbol'] = self.active_symbol.name
            
            # Save updated config
            try:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to save config: {e}")
            
            self.logger.info(f" Successfully switched to: {self.active_symbol.name}")
            return True
        
        return False
    
    def get_point_value_calculator(self) -> callable:
        """Get point value calculator for current symbol"""
        if not self.active_symbol:
            return lambda points: 0
        
        def calculate_point_value(points: float, lot_size: float = 0.01) -> float:
            """Calculate monetary value of points"""
            return points * self.active_symbol.point * lot_size * self.active_symbol.contract_size
        
        return calculate_point_value

# Example usage and testing
if __name__ == "__main__":
    # Initialize symbol manager
    symbol_mgr = SymbolManager()
    
    # Auto-detect XAUUSD
    if symbol_mgr.auto_detect_symbol('XAUUSD'):
        print(" Symbol auto-detection successful")
        
        # Get specs report
        report = symbol_mgr.get_symbol_specs_report()
        print(f"\n📊 Symbol Report:")
        print(f"   Name: {report['symbol_info']['name']}")
        print(f"   Broker: {report['broker_info']['company'] if 'broker_info' in report else 'Unknown'}")
        print(f"   Spread: {report['market_data']['spread_points']:.1f} points")
        print(f"   Min Lot: {report['trading_specs']['min_lot']}")
        print(f"   Tradeable: {report['validation'][0]} - {report['validation'][1]}")
        
        # Test point value calculator
        calc = symbol_mgr.get_point_value_calculator()
        value = calc(100, 0.01)  # 100 points, 0.01 lot
        print(f"   100 points value: ${value:.2f}")
        
    else:
        print("❌ Symbol auto-detection failed")
        
        # Search for gold symbols
        gold_symbols = symbol_mgr.search_gold_symbols()
        if gold_symbols:
            print(f"\n🔍 Found {len(gold_symbols)} gold symbols:")
            for sym in gold_symbols[:5]:  # Show first 5
                print(f"   {sym.name} - Spread: {sym.spread}")
        else:
            print("❌ No gold symbols found")