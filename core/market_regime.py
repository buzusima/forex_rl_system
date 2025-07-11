# core/market_regime.py
"""
Market Regime Detection Engine
ตรวจจับ market regime: Trending vs Ranging vs Breakout
สำคัญมากสำหรับการเลือก strategy ที่เหมาะสม
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import threading
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types"""
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    VOLATILE = "volatile"

@dataclass
class RegimeData:
    """Market regime data structure"""
    symbol: str
    timeframe: str
    regime: MarketRegime
    confidence: float
    trend_strength: float
    trend_direction: int  # 1=bullish, -1=bearish, 0=neutral
    volatility: float
    momentum: float
    adx_value: float
    price_location: float  # Where price is relative to range (0-1)
    support_resistance: Dict[str, float]
    volume_trend: str
    timestamp: datetime

@dataclass
class MultiTimeframeRegime:
    """Multi-timeframe regime analysis"""
    symbol: str
    regimes: Dict[str, RegimeData]
    dominant_regime: MarketRegime
    regime_alignment: float  # How aligned the timeframes are (0-1)
    trading_recommendation: str
    risk_level: str
    timestamp: datetime

class MarketRegimeDetector:
    """
    Market Regime Detection Engine
    - ตรวจจับ market regime แบบ multi-timeframe
    - วิเคราะห์ trend strength และ direction
    - ให้คำแนะนำ strategy ที่เหมาะสม
    - Integration กับ currency strength และ correlation
    """
    
    def __init__(self, data_manager, currency_strength_engine=None):
        self.data_manager = data_manager
        self.strength_engine = currency_strength_engine
        
        # Configuration
        self.timeframes = ["M5", "M15", "H1", "H4"]
        self.currency_pairs = self.data_manager.currency_pairs
        
        # Regime detection parameters
        self.adx_threshold_strong = 25
        self.adx_threshold_weak = 20
        self.volatility_threshold = 0.015  # 1.5%
        self.breakout_threshold = 0.02     # 2%
        
        # Cache
        self.regime_cache = {}
        self.multi_tf_cache = {}
        self.last_calculation = {}
        
        # Threading
        self.lock = threading.Lock()
        
        logger.info(" Market Regime Detector initialized")
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calculate True Range (TR)
            hl = high - low
            hc = np.abs(high - np.roll(close, 1))
            lc = np.abs(low - np.roll(close, 1))
            tr = np.maximum(hl, np.maximum(hc, lc))
            
            # Calculate Directional Movement (DM)
            dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                              np.maximum(high - np.roll(high, 1), 0), 0)
            dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                               np.maximum(np.roll(low, 1) - low, 0), 0)
            
            # Smooth TR and DM
            tr_smooth = pd.Series(tr).rolling(window=period).mean().values
            dm_plus_smooth = pd.Series(dm_plus).rolling(window=period).mean().values
            dm_minus_smooth = pd.Series(dm_minus).rolling(window=period).mean().values
            
            # Calculate Directional Indicators (DI)
            di_plus = 100 * dm_plus_smooth / tr_smooth
            di_minus = 100 * dm_minus_smooth / tr_smooth
            
            # Calculate Directional Index (DX)
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
            
            # Calculate ADX
            adx = pd.Series(dx).rolling(window=period).mean().iloc[-1]
            
            return adx if not np.isnan(adx) else 0
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return 0
    
    def calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate price volatility"""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < period:
                return 0
            
            volatility = returns.rolling(window=period).std().iloc[-1]
            return volatility if not np.isnan(volatility) else 0
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0
    
    def calculate_momentum(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate momentum indicator"""
        try:
            if len(data) < period + 1:
                return 0
            
            current_price = data['close'].iloc[-1]
            past_price = data['close'].iloc[-(period + 1)]
            
            momentum = (current_price - past_price) / past_price * 100
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0
    
    def detect_support_resistance(self, data: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """Detect key support and resistance levels"""
        try:
            if len(data) < period * 2:
                return {'support': 0, 'resistance': 0}
            
            # Get recent data
            recent_data = data.tail(period * 2)
            
            # Find pivot highs and lows
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Simple support/resistance using percentiles
            resistance = np.percentile(highs, 80)
            support = np.percentile(lows, 20)
            
            return {'support': support, 'resistance': resistance}
            
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
            return {'support': 0, 'resistance': 0}
    
    def calculate_price_location(self, data: pd.DataFrame, support: float, resistance: float) -> float:
        """Calculate where current price is relative to support/resistance range"""
        try:
            if resistance <= support:
                return 0.5
            
            current_price = data['close'].iloc[-1]
            location = (current_price - support) / (resistance - support)
            
            return max(0, min(1, location))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating price location: {e}")
            return 0.5
    
    def analyze_volume_trend(self, data: pd.DataFrame, period: int = 10) -> str:
        """Analyze volume trend"""
        try:
            if 'volume' not in data.columns or len(data) < period * 2:
                return "neutral"
            
            recent_volume = data['volume'].tail(period).mean()
            past_volume = data['volume'].iloc[-(period * 2):-period].mean()
            
            if recent_volume > past_volume * 1.2:
                return "increasing"
            elif recent_volume < past_volume * 0.8:
                return "decreasing"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error analyzing volume trend: {e}")
            return "neutral"
    
    def determine_regime(self, symbol: str, timeframe: str, periods: int = 100) -> Optional[RegimeData]:
        """Determine market regime for a single symbol and timeframe"""
        try:
            # Get data
            data = self.data_manager.get_data(symbol, timeframe, periods)
            if data is None or len(data) < 50:
                return None
            
            # Calculate indicators
            adx_value = self.calculate_adx(data)
            volatility = self.calculate_volatility(data)
            momentum = self.calculate_momentum(data)
            support_resistance = self.detect_support_resistance(data)
            price_location = self.calculate_price_location(data, 
                                                         support_resistance['support'],
                                                         support_resistance['resistance'])
            volume_trend = self.analyze_volume_trend(data)
            
            # Calculate moving averages for trend detection
            ma_fast = data['close'].rolling(10).mean().iloc[-1]
            ma_slow = data['close'].rolling(21).mean().iloc[-1]
            ma_trend = data['close'].rolling(50).mean().iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # Determine trend direction and strength
            if ma_fast > ma_slow > ma_trend and current_price > ma_fast:
                trend_direction = 1
                trend_strength = (ma_fast - ma_trend) / ma_trend * 100
            elif ma_fast < ma_slow < ma_trend and current_price < ma_fast:
                trend_direction = -1
                trend_strength = (ma_trend - ma_fast) / ma_trend * 100
            else:
                trend_direction = 0
                trend_strength = abs(ma_fast - ma_slow) / ma_slow * 100
            
            # Determine regime based on indicators
            regime, confidence = self._classify_regime(
                adx_value, volatility, momentum, trend_strength, 
                trend_direction, price_location
            )
            
            return RegimeData(
                symbol=symbol,
                timeframe=timeframe,
                regime=regime,
                confidence=confidence,
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                volatility=volatility,
                momentum=momentum,
                adx_value=adx_value,
                price_location=price_location,
                support_resistance=support_resistance,
                volume_trend=volume_trend,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error determining regime for {symbol} {timeframe}: {e}")
            return None
    
    def _classify_regime(self, adx: float, volatility: float, momentum: float,
                        trend_strength: float, trend_direction: int, price_location: float) -> Tuple[MarketRegime, float]:
        """Classify market regime based on indicators"""
        
        # Strong trend conditions
        if adx > self.adx_threshold_strong and trend_strength > 1.0:
            confidence = min(95, adx * 2)
            return MarketRegime.STRONG_TREND, confidence
        
        # Weak trend conditions
        elif adx > self.adx_threshold_weak and trend_strength > 0.5:
            confidence = min(80, adx * 1.5)
            return MarketRegime.WEAK_TREND, confidence
        
        # Breakout conditions
        elif volatility > self.volatility_threshold and abs(momentum) > 2.0:
            if price_location > 0.8 or price_location < 0.2:
                confidence = min(85, volatility * 1000 + abs(momentum) * 10)
                return MarketRegime.BREAKOUT, confidence
        
        # High volatility without clear direction
        elif volatility > self.volatility_threshold * 1.5:
            confidence = min(75, volatility * 800)
            return MarketRegime.VOLATILE, confidence
        
        # Consolidation (low volatility, low ADX)
        elif volatility < self.volatility_threshold * 0.5 and adx < 15:
            confidence = min(70, (self.volatility_threshold * 0.5 - volatility) * 2000)
            return MarketRegime.CONSOLIDATION, confidence
        
        # Default to ranging
        else:
            confidence = 60 - adx  # Lower confidence for unclear regimes
            return MarketRegime.RANGING, max(30, confidence)
    
    def analyze_multi_timeframe_regime(self, symbol: str) -> Optional[MultiTimeframeRegime]:
        """Analyze regime across multiple timeframes"""
        try:
            regimes = {}
            regime_scores = {regime: 0 for regime in MarketRegime}
            
            # Analyze each timeframe
            for timeframe in self.timeframes:
                regime_data = self.determine_regime(symbol, timeframe)
                if regime_data:
                    regimes[timeframe] = regime_data
                    # Weight by timeframe importance (longer timeframes have more weight)
                    weight = {"M5": 1, "M15": 2, "H1": 3, "H4": 4}[timeframe]
                    regime_scores[regime_data.regime] += weight * (regime_data.confidence / 100)
            
            if not regimes:
                return None
            
            # Determine dominant regime
            dominant_regime = max(regime_scores, key=regime_scores.get)
            
            # Calculate regime alignment
            dominant_count = sum(1 for r in regimes.values() if r.regime == dominant_regime)
            regime_alignment = dominant_count / len(regimes)
            
            # Generate trading recommendation
            trading_recommendation = self._generate_trading_recommendation(
                dominant_regime, regime_alignment, regimes
            )
            
            # Assess risk level
            risk_level = self._assess_risk_level(dominant_regime, regime_alignment, regimes)
            
            return MultiTimeframeRegime(
                symbol=symbol,
                regimes=regimes,
                dominant_regime=dominant_regime,
                regime_alignment=regime_alignment,
                trading_recommendation=trading_recommendation,
                risk_level=risk_level,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing multi-timeframe regime for {symbol}: {e}")
            return None
    
    def _generate_trading_recommendation(self, dominant_regime: MarketRegime, 
                                       alignment: float, regimes: Dict) -> str:
        """Generate trading recommendation based on regime analysis"""
        
        if alignment < 0.5:
            return "WAIT - Mixed signals across timeframes"
        
        if dominant_regime == MarketRegime.STRONG_TREND:
            if alignment > 0.75:
                return "TREND_FOLLOWING - High confidence trend trading"
            else:
                return "TREND_FOLLOWING - Moderate confidence trend trading"
        
        elif dominant_regime == MarketRegime.WEAK_TREND:
            return "SCALPING - Short-term trend following with quick exits"
        
        elif dominant_regime == MarketRegime.RANGING:
            return "RANGE_TRADING - Buy support, sell resistance"
        
        elif dominant_regime == MarketRegime.BREAKOUT:
            return "BREAKOUT_TRADING - Wait for breakout confirmation"
        
        elif dominant_regime == MarketRegime.CONSOLIDATION:
            return "WAIT - Market consolidating, wait for clear direction"
        
        elif dominant_regime == MarketRegime.VOLATILE:
            return "REDUCE_EXPOSURE - High volatility, reduce position sizes"
        
        else:
            return "NEUTRAL - No clear trading strategy"
    
    def _assess_risk_level(self, dominant_regime: MarketRegime, 
                          alignment: float, regimes: Dict) -> str:
        """Assess risk level based on regime analysis"""
        
        # Calculate average volatility across timeframes
        avg_volatility = np.mean([r.volatility for r in regimes.values()])
        
        if dominant_regime in [MarketRegime.VOLATILE, MarketRegime.BREAKOUT]:
            return "HIGH"
        elif dominant_regime == MarketRegime.STRONG_TREND and alignment > 0.75:
            return "LOW"
        elif dominant_regime in [MarketRegime.RANGING, MarketRegime.CONSOLIDATION]:
            return "MEDIUM" if avg_volatility > 0.01 else "LOW"
        elif alignment < 0.5:
            return "HIGH"
        else:
            return "MEDIUM"
    
    def get_market_overview(self, major_pairs: List[str] = None) -> Dict:
        """Get market overview for major currency pairs"""
        if major_pairs is None:
            major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
        
        overview = {
            'timestamp': datetime.now(),
            'pairs_analysis': {},
            'market_summary': {
                'trending_pairs': 0,
                'ranging_pairs': 0,
                'volatile_pairs': 0,
                'breakout_pairs': 0,
                'average_risk': 'MEDIUM'
            }
        }
        
        risk_scores = []
        
        for pair in major_pairs:
            if pair in self.currency_pairs:
                multi_tf_analysis = self.analyze_multi_timeframe_regime(pair)
                if multi_tf_analysis:
                    overview['pairs_analysis'][pair] = multi_tf_analysis
                    
                    # Update summary
                    if multi_tf_analysis.dominant_regime in [MarketRegime.STRONG_TREND, MarketRegime.WEAK_TREND]:
                        overview['market_summary']['trending_pairs'] += 1
                    elif multi_tf_analysis.dominant_regime == MarketRegime.RANGING:
                        overview['market_summary']['ranging_pairs'] += 1
                    elif multi_tf_analysis.dominant_regime == MarketRegime.VOLATILE:
                        overview['market_summary']['volatile_pairs'] += 1
                    elif multi_tf_analysis.dominant_regime == MarketRegime.BREAKOUT:
                        overview['market_summary']['breakout_pairs'] += 1
                    
                    # Risk scoring
                    risk_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
                    risk_scores.append(risk_map.get(multi_tf_analysis.risk_level, 2))
        
        # Calculate average risk
        if risk_scores:
            avg_risk_score = np.mean(risk_scores)
            if avg_risk_score <= 1.3:
                overview['market_summary']['average_risk'] = 'LOW'
            elif avg_risk_score >= 2.7:
                overview['market_summary']['average_risk'] = 'HIGH'
            else:
                overview['market_summary']['average_risk'] = 'MEDIUM'
        
        return overview
    
    def get_trading_opportunities(self, timeframe: str = "H1") -> List[Dict]:
        """Get trading opportunities based on regime analysis"""
        opportunities = []
        
        for pair in self.currency_pairs:
            regime_data = self.determine_regime(pair, timeframe)
            if regime_data:
                
                # Strong trend opportunities
                if (regime_data.regime == MarketRegime.STRONG_TREND and 
                    regime_data.confidence > 70):
                    
                    direction = "BUY" if regime_data.trend_direction == 1 else "SELL"
                    opportunities.append({
                        'pair': pair,
                        'strategy': 'TREND_FOLLOWING',
                        'direction': direction,
                        'regime': regime_data.regime.value,
                        'confidence': regime_data.confidence,
                        'risk_level': 'LOW' if regime_data.confidence > 80 else 'MEDIUM',
                        'entry_reason': f"Strong {direction.lower()} trend with ADX {regime_data.adx_value:.1f}"
                    })
                
                # Breakout opportunities
                elif (regime_data.regime == MarketRegime.BREAKOUT and 
                      regime_data.confidence > 60):
                    
                    if regime_data.price_location > 0.8:
                        direction = "BUY"
                        entry_reason = "Price breaking above resistance"
                    elif regime_data.price_location < 0.2:
                        direction = "SELL"
                        entry_reason = "Price breaking below support"
                    else:
                        continue
                    
                    opportunities.append({
                        'pair': pair,
                        'strategy': 'BREAKOUT_TRADING',
                        'direction': direction,
                        'regime': regime_data.regime.value,
                        'confidence': regime_data.confidence,
                        'risk_level': 'HIGH',
                        'entry_reason': entry_reason
                    })
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities[:10]  # Top 10 opportunities

# Testing and example usage
if __name__ == "__main__":
    print(" Market Regime Detector Test")
    print("=" * 50)
    
    # Import required modules
    try:
        from data_manager import DataManager
        from currency_strength import CurrencyStrengthEngine
    except ImportError:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.insert(0, parent_dir)
        from core.data_manager import DataManager
        from core.currency_strength import CurrencyStrengthEngine
    
    # Initialize
    dm = DataManager()
    
    if dm.connect_mt5():
        print(" MT5 connected")
        
        # Create engines
        strength_engine = CurrencyStrengthEngine(dm)
        regime_detector = MarketRegimeDetector(dm, strength_engine)
        
        # Test single pair regime detection
        print("\n🔍 Testing EURUSD Regime Detection...")
        eurusd_regime = regime_detector.determine_regime("EURUSD", "H1")
        
        if eurusd_regime:
            print(f" EURUSD H1 Regime Analysis:")
            print(f"   Regime: {eurusd_regime.regime.value}")
            print(f"   Confidence: {eurusd_regime.confidence:.1f}%")
            print(f"   Trend Direction: {eurusd_regime.trend_direction}")
            print(f"   Trend Strength: {eurusd_regime.trend_strength:.2f}")
            print(f"   ADX: {eurusd_regime.adx_value:.1f}")
            print(f"   Volatility: {eurusd_regime.volatility:.4f}")
            print(f"   Price Location: {eurusd_regime.price_location:.2f}")
            print(f"   Volume Trend: {eurusd_regime.volume_trend}")
        
        # Test multi-timeframe analysis
        print("\n🔍 Testing Multi-Timeframe Analysis...")
        multi_tf = regime_detector.analyze_multi_timeframe_regime("EURUSD")
        
        if multi_tf:
            print(f" EURUSD Multi-Timeframe Analysis:")
            print(f"   Dominant Regime: {multi_tf.dominant_regime.value}")
            print(f"   Regime Alignment: {multi_tf.regime_alignment:.2f}")
            print(f"   Trading Recommendation: {multi_tf.trading_recommendation}")
            print(f"   Risk Level: {multi_tf.risk_level}")
            
            print(f"\n   Timeframe Breakdown:")
            for tf, regime in multi_tf.regimes.items():
                print(f"   {tf}: {regime.regime.value} ({regime.confidence:.0f}%)")
        
        # Test market overview
        print("\n🔍 Testing Market Overview...")
        overview = regime_detector.get_market_overview()
        
        print(f" Market Overview:")
        summary = overview['market_summary']
        print(f"   Trending Pairs: {summary['trending_pairs']}")
        print(f"   Ranging Pairs: {summary['ranging_pairs']}")
        print(f"   Volatile Pairs: {summary['volatile_pairs']}")
        print(f"   Breakout Pairs: {summary['breakout_pairs']}")
        print(f"   Average Risk: {summary['average_risk']}")
        
        # Test trading opportunities
        print("\n🔍 Finding Trading Opportunities...")
        opportunities = regime_detector.get_trading_opportunities("H1")
        
        if opportunities:
            print(f"🎯 Trading Opportunities (H1):")
            for i, opp in enumerate(opportunities[:5], 1):
                print(f"{i}. {opp['direction']} {opp['pair']} - {opp['strategy']}")
                print(f"   Confidence: {opp['confidence']:.0f}% | Risk: {opp['risk_level']}")
                print(f"   Reason: {opp['entry_reason']}")
        else:
            print("📈 No clear trading opportunities at this time")
        
    else:
        print("❌ MT5 connection failed")
    
    # Cleanup
    dm.close()
    print("\n✅ Test completed")