# core/currency_strength.py
"""
Multi-Timeframe Currency Strength Engine
คำนวณความแข็งแกร่งของสกุลเงินแต่ละตัวจาก 21 คู่เงิน
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import threading

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class CurrencyStrength:
    """Currency strength data structure"""
    currency: str
    timeframe: str
    strength: float
    momentum: float
    volatility: float
    trend_direction: int  # 1=bullish, 0=neutral, -1=bearish
    timestamp: datetime

class CurrencyStrengthEngine:
    """
    Multi-Timeframe Currency Strength Calculator
    - วิเคราะห์ความแข็งแกร่งของ 8 สกุลเงินหลัก
    - คำนวณจาก 21 คู่เงิน x 4 timeframes
    - Real-time strength monitoring
    - Momentum และ trend analysis
    """
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        
        # 8 major currencies
        self.currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
        
        # Timeframes
        self.timeframes = ["M5", "M15", "H1", "H4"]
        
        # Currency pair mappings
        self.currency_pairs = self.data_manager.currency_pairs
        
        # Build currency-pair mapping
        self.currency_pair_map = self._build_currency_pair_map()
        
        # Strength cache
        self.strength_cache = {}
        self.last_calculation = {}
        
        # Threading
        self.lock = threading.Lock()
        
        logger.info(" Currency Strength Engine initialized")
    
    def _build_currency_pair_map(self) -> Dict[str, List[str]]:
        """Build mapping of currency to its related pairs"""
        currency_map = {currency: [] for currency in self.currencies}
        
        for pair in self.currency_pairs:
            if len(pair) == 6:  # Standard format like EURUSD
                base = pair[:3]
                quote = pair[3:]
                
                if base in self.currencies:
                    currency_map[base].append(pair)
                if quote in self.currencies:
                    currency_map[quote].append(pair)
        
        # Log mapping for verification
        for currency, pairs in currency_map.items():
            logger.info(f"{currency}: {len(pairs)} pairs - {pairs}")
        
        return currency_map
    
    def calculate_single_currency_strength(self, currency: str, timeframe: str, 
                                         bars: int = 100) -> Optional[CurrencyStrength]:
        """Calculate strength for single currency on single timeframe"""
        try:
            related_pairs = self.currency_pair_map.get(currency, [])
            if not related_pairs:
                logger.warning(f"No pairs found for {currency}")
                return None
            
            strength_values = []
            price_changes = []
            
            for pair in related_pairs:
                # Get data
                data = self.data_manager.get_data(pair, timeframe, bars)
                if data is None or len(data) < 20:
                    continue
                
                # Calculate price change
                current_price = data['close'].iloc[-1]
                past_price = data['close'].iloc[-20]  # 20 bars ago
                
                price_change = (current_price - past_price) / past_price * 100
                
                # Determine if currency is base or quote
                if pair[:3] == currency:
                    # Currency is base - positive change = stronger
                    strength_values.append(price_change)
                else:
                    # Currency is quote - negative change = stronger
                    strength_values.append(-price_change)
                
                price_changes.append(abs(price_change))
            
            if not strength_values:
                return None
            
            # Calculate metrics
            strength = np.mean(strength_values)
            momentum = self._calculate_momentum(currency, timeframe, bars)
            volatility = np.mean(price_changes) if price_changes else 0
            trend_direction = self._determine_trend(currency, timeframe, bars)
            
            return CurrencyStrength(
                currency=currency,
                timeframe=timeframe,
                strength=strength,
                momentum=momentum,
                volatility=volatility,
                trend_direction=trend_direction,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating strength for {currency} {timeframe}: {e}")
            return None
    
    def _calculate_momentum(self, currency: str, timeframe: str, bars: int) -> float:
        """Calculate momentum using RSI-like approach"""
        try:
            related_pairs = self.currency_pair_map.get(currency, [])
            momentum_values = []
            
            for pair in related_pairs:
                data = self.data_manager.get_data(pair, timeframe, bars)
                if data is None or len(data) < 14:
                    continue
                
                # Calculate RSI-like momentum
                closes = data['close'].values
                deltas = np.diff(closes)
                
                gains = deltas.copy()
                losses = deltas.copy()
                gains[gains < 0] = 0
                losses[losses > 0] = 0
                losses = np.abs(losses)
                
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                
                if avg_loss == 0:
                    momentum = 100
                else:
                    rs = avg_gain / avg_loss
                    momentum = 100 - (100 / (1 + rs))
                
                # Adjust for currency position
                if pair[:3] == currency:
                    momentum_values.append(momentum)
                else:
                    momentum_values.append(100 - momentum)
            
            return np.mean(momentum_values) if momentum_values else 50.0
            
        except Exception as e:
            logger.error(f"Error calculating momentum for {currency}: {e}")
            return 50.0
    
    def _determine_trend(self, currency: str, timeframe: str, bars: int) -> int:
        """Determine trend direction using multiple moving averages"""
        try:
            related_pairs = self.currency_pair_map.get(currency, [])
            trend_votes = []
            
            for pair in related_pairs:
                data = self.data_manager.get_data(pair, timeframe, bars)
                if data is None or len(data) < 50:
                    continue
                
                # Calculate moving averages
                ma_fast = data['close'].rolling(10).mean().iloc[-1]
                ma_slow = data['close'].rolling(21).mean().iloc[-1]
                ma_trend = data['close'].rolling(50).mean().iloc[-1]
                
                current_price = data['close'].iloc[-1]
                
                # Determine trend
                if ma_fast > ma_slow > ma_trend and current_price > ma_fast:
                    trend = 1  # Bullish
                elif ma_fast < ma_slow < ma_trend and current_price < ma_fast:
                    trend = -1  # Bearish
                else:
                    trend = 0  # Neutral
                
                # Adjust for currency position
                if pair[:3] == currency:
                    trend_votes.append(trend)
                else:
                    trend_votes.append(-trend)
            
            if not trend_votes:
                return 0
            
            # Return majority vote
            avg_trend = np.mean(trend_votes)
            if avg_trend > 0.3:
                return 1
            elif avg_trend < -0.3:
                return -1
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error determining trend for {currency}: {e}")
            return 0
    
    def calculate_all_strengths(self, timeframe: str, bars: int = 100) -> Dict[str, CurrencyStrength]:
        """Calculate strength for all currencies on specific timeframe"""
        logger.info(f" Calculating currency strengths for {timeframe}...")
        
        strengths = {}
        
        for currency in self.currencies:
            strength = self.calculate_single_currency_strength(currency, timeframe, bars)
            if strength:
                strengths[currency] = strength
        
        # Cache results
        with self.lock:
            cache_key = f"{timeframe}_{bars}"
            self.strength_cache[cache_key] = strengths.copy()
            self.last_calculation[cache_key] = datetime.now()
        
        logger.info(f" Calculated strengths for {len(strengths)}/{len(self.currencies)} currencies")
        return strengths
    
    def get_cached_strengths(self, timeframe: str, bars: int = 100, 
                           max_age_minutes: int = 5) -> Optional[Dict[str, CurrencyStrength]]:
        """Get cached strength data if fresh enough"""
        cache_key = f"{timeframe}_{bars}"
        
        with self.lock:
            if (cache_key in self.strength_cache and 
                cache_key in self.last_calculation):
                
                age = (datetime.now() - self.last_calculation[cache_key]).total_seconds() / 60
                if age <= max_age_minutes:
                    return self.strength_cache[cache_key].copy()
        
        return None
    
    def get_strength_ranking(self, timeframe: str, bars: int = 100) -> List[Tuple[str, float]]:
        """Get currencies ranked by strength (strongest first)"""
        # Try cache first
        strengths = self.get_cached_strengths(timeframe, bars)
        
        # Calculate if not cached
        if strengths is None:
            strengths = self.calculate_all_strengths(timeframe, bars)
        
        # Create ranking
        ranking = [(currency, data.strength) for currency, data in strengths.items()]
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def get_strength_pairs(self, timeframe: str, min_strength_diff: float = 0.5) -> List[Tuple[str, str, float]]:
        """Get currency pairs with significant strength difference"""
        ranking = self.get_strength_ranking(timeframe)
        
        if len(ranking) < 2:
            return []
        
        pairs = []
        
        # Find pairs with significant strength difference
        for i, (strong_currency, strong_value) in enumerate(ranking):
            for j, (weak_currency, weak_value) in enumerate(ranking[i+1:], i+1):
                strength_diff = strong_value - weak_value
                
                if strength_diff >= min_strength_diff:
                    pairs.append((strong_currency, weak_currency, strength_diff))
        
        return pairs
    
    def get_multi_timeframe_strength(self, currency: str) -> Dict[str, CurrencyStrength]:
        """Get strength data for single currency across all timeframes"""
        multi_tf_strength = {}
        
        for timeframe in self.timeframes:
            strength = self.calculate_single_currency_strength(currency, timeframe)
            if strength:
                multi_tf_strength[timeframe] = strength
        
        return multi_tf_strength
    
    def get_strength_matrix(self, timeframe: str) -> pd.DataFrame:
        """Get strength matrix as DataFrame for easy analysis"""
        strengths = self.get_cached_strengths(timeframe)
        if strengths is None:
            strengths = self.calculate_all_strengths(timeframe)
        
        data = []
        for currency, strength_data in strengths.items():
            data.append({
                'Currency': currency,
                'Strength': round(strength_data.strength, 3),
                'Momentum': round(strength_data.momentum, 1),
                'Volatility': round(strength_data.volatility, 3),
                'Trend': strength_data.trend_direction,
                'Timestamp': strength_data.timestamp
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('Strength', ascending=False).reset_index(drop=True)
    
    def find_best_pairs_to_trade(self, timeframe: str) -> List[Dict]:
        """Find best currency pairs to trade based on strength analysis"""
        ranking = self.get_strength_ranking(timeframe)
        
        if len(ranking) < 2:
            return []
        
        trade_opportunities = []
        
        # Get top 3 strongest and weakest
        strongest = ranking[:3]
        weakest = ranking[-3:]
        
        for strong_currency, strong_value in strongest:
            for weak_currency, weak_value in weakest:
                strength_diff = strong_value - weak_value
                
                if strength_diff >= 1.0:  # Minimum difference threshold
                    # Find actual trading pair
                    trading_pair = self._find_trading_pair(strong_currency, weak_currency)
                    
                    if trading_pair:
                        trade_opportunities.append({
                            'pair': trading_pair['pair'],
                            'direction': trading_pair['direction'],
                            'strong_currency': strong_currency,
                            'weak_currency': weak_currency,
                            'strength_diff': round(strength_diff, 3),
                            'confidence': min(100, strength_diff * 20)  # Scale to 0-100
                        })
        
        # Sort by strength difference
        trade_opportunities.sort(key=lambda x: x['strength_diff'], reverse=True)
        
        return trade_opportunities[:5]  # Top 5 opportunities
    
    def _find_trading_pair(self, strong_currency: str, weak_currency: str) -> Optional[Dict]:
        """Find actual tradeable pair for two currencies"""
        # Direct pair (strong/weak)
        direct_pair = f"{strong_currency}{weak_currency}"
        if direct_pair in self.currency_pairs:
            return {'pair': direct_pair, 'direction': 'BUY'}
        
        # Reverse pair (weak/strong)
        reverse_pair = f"{weak_currency}{strong_currency}"
        if reverse_pair in self.currency_pairs:
            return {'pair': reverse_pair, 'direction': 'SELL'}
        
        return None
    
    def get_currency_correlation_strength(self, currency1: str, currency2: str, 
                                        timeframe: str) -> float:
        """Calculate correlation between two currencies"""
        try:
            pairs1 = self.currency_pair_map.get(currency1, [])
            pairs2 = self.currency_pair_map.get(currency2, [])
            
            # Find common pairs or related pairs
            correlations = []
            
            for pair in pairs1:
                if currency2 in pair:
                    # Direct correlation through shared pair
                    data = self.data_manager.get_data(pair, timeframe, 100)
                    if data is not None and len(data) > 20:
                        returns = data['close'].pct_change().dropna()
                        if len(returns) > 10:
                            # Calculate correlation strength
                            correlation = abs(returns.rolling(20).corr(returns.shift(1)).iloc[-1])
                            if not np.isnan(correlation):
                                correlations.append(correlation)
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation {currency1}-{currency2}: {e}")
            return 0.0

# Testing and example usage
if __name__ == "__main__":
    print("💪 Currency Strength Engine Test")
    print("=" * 50)
    
    # Import data manager
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core.data_manager import DataManager
    
    # Initialize
    dm = DataManager()
    
    if dm.connect_mt5():
        print(" MT5 connected")
        
        # Create strength engine
        strength_engine = CurrencyStrengthEngine(dm)
        
        # Test H1 timeframe
        print("\n🔍 Testing H1 Currency Strength...")
        
        # Calculate all strengths
        strengths_h1 = strength_engine.calculate_all_strengths("H1")
        
        # Display results
        print(f"\n Currency Strength Ranking (H1):")
        ranking = strength_engine.get_strength_ranking("H1")
        
        for i, (currency, strength) in enumerate(ranking, 1):
            strength_data = strengths_h1[currency]
            trend_symbol = "🔼" if strength_data.trend_direction == 1 else "🔽" if strength_data.trend_direction == -1 else "▶️"
            print(f"{i:2d}. {currency}: {strength:+6.3f} | Momentum: {strength_data.momentum:5.1f} | {trend_symbol}")
        
        # Find trading opportunities
        print(f"\n🎯 Best Trading Opportunities:")
        opportunities = strength_engine.find_best_pairs_to_trade("H1")
        
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp['direction']} {opp['pair']} | Strength Diff: {opp['strength_diff']:.3f} | Confidence: {opp['confidence']:.0f}%")
        
        # Get strength matrix
        print(f"\n📋 Strength Matrix:")
        matrix = strength_engine.get_strength_matrix("H1")
        print(matrix.to_string(index=False))
        
    else:
        print("❌ MT5 connection failed")
    
    # Cleanup
    dm.close()
    print("\n✅ Test completed")