# core/correlation_engine.py
"""
Multi-Pair Correlation Analysis Engine
วิเคราะห์ correlation ระหว่างคู่เงิน และ detect breakdown patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass
import threading
import itertools

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class CorrelationData:
    """Correlation data structure"""
    pair1: str
    pair2: str
    timeframe: str
    correlation: float
    rolling_correlation: List[float]
    correlation_strength: str  # 'strong', 'moderate', 'weak'
    is_breakdown: bool
    breakdown_threshold: float
    historical_avg: float
    volatility: float
    timestamp: datetime

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity structure"""
    pair1: str
    pair2: str
    pair3: str  # For triangular arbitrage
    opportunity_type: str  # 'correlation_divergence', 'triangular_arbitrage'
    expected_direction: Dict[str, str]  # pair -> BUY/SELL
    profit_potential: float
    confidence: float
    risk_level: str
    timestamp: datetime

class CorrelationEngine:
    """
    Multi-Pair Correlation Analysis Engine
    - วิเคราะห์ correlation ระหว่างคู่เงินทั้งหมด
    - Detect correlation breakdown
    - หา arbitrage opportunities
    - Multi-timeframe correlation analysis
    """
    
    def __init__(self, data_manager, currency_strength_engine):
        self.data_manager = data_manager
        self.strength_engine = currency_strength_engine
        
        # Configuration
        self.currency_pairs = self.data_manager.currency_pairs
        self.timeframes = ["M5", "M15", "H1", "H4"]
        
        # Correlation thresholds
        self.strong_correlation_threshold = 0.7
        self.moderate_correlation_threshold = 0.4
        self.breakdown_threshold = 0.3
        
        # Cache
        self.correlation_cache = {}
        self.arbitrage_cache = {}
        self.last_calculation = {}
        
        # Threading
        self.lock = threading.Lock()
        
        # Pre-calculate important pair relationships
        self.important_pairs = self._identify_important_pairs()
        
        logger.info(" Correlation Engine initialized")
        logger.info(f" Monitoring {len(self.important_pairs)} important pair relationships")
    
    def _identify_important_pairs(self) -> List[Tuple[str, str]]:
        """Identify important currency pair relationships for monitoring"""
        important_pairs = []
        
        # 1. Same base currency pairs
        base_groups = {}
        for pair in self.currency_pairs:
            base = pair[:3]
            if base not in base_groups:
                base_groups[base] = []
            base_groups[base].append(pair)
        
        for base, pairs in base_groups.items():
            if len(pairs) > 1:
                for pair1, pair2 in itertools.combinations(pairs, 2):
                    important_pairs.append((pair1, pair2))
        
        # 2. Same quote currency pairs
        quote_groups = {}
        for pair in self.currency_pairs:
            quote = pair[3:]
            if quote not in quote_groups:
                quote_groups[quote] = []
            quote_groups[quote].append(pair)
        
        for quote, pairs in quote_groups.items():
            if len(pairs) > 1:
                for pair1, pair2 in itertools.combinations(pairs, 2):
                    if (pair1, pair2) not in important_pairs and (pair2, pair1) not in important_pairs:
                        important_pairs.append((pair1, pair2))
        
        # 3. Cross-correlation pairs (different base/quote but related)
        cross_pairs = [
            ("EURUSD", "GBPUSD"),  # EUR-GBP relationship
            ("AUDUSD", "NZDUSD"),  # AUD-NZD relationship
            ("USDCHF", "EURUSD"),  # USD strength pairs
            ("USDJPY", "EURJPY"),  # JPY pairs
            ("GBPJPY", "EURJPY"),  # JPY cross pairs
        ]
        
        for pair1, pair2 in cross_pairs:
            if pair1 in self.currency_pairs and pair2 in self.currency_pairs:
                if (pair1, pair2) not in important_pairs and (pair2, pair1) not in important_pairs:
                    important_pairs.append((pair1, pair2))
        
        return important_pairs
    
    def calculate_correlation(self, pair1: str, pair2: str, timeframe: str, 
                            periods: int = 100, rolling_window: int = 20) -> Optional[CorrelationData]:
        """Calculate correlation between two currency pairs"""
        try:
            # Get data for both pairs
            data1 = self.data_manager.get_data(pair1, timeframe, periods)
            data2 = self.data_manager.get_data(pair2, timeframe, periods)
            
            if data1 is None or data2 is None:
                return None
            
            if len(data1) < rolling_window or len(data2) < rolling_window:
                return None
            
            # Align timestamps
            data1['timestamp'] = pd.to_datetime(data1['timestamp'])
            data2['timestamp'] = pd.to_datetime(data2['timestamp'])
            
            # Merge on timestamp
            merged = pd.merge(data1[['timestamp', 'close']], 
                            data2[['timestamp', 'close']], 
                            on='timestamp', suffixes=('_1', '_2'))
            
            if len(merged) < rolling_window:
                return None
            
            # Calculate returns
            merged['return_1'] = merged['close_1'].pct_change()
            merged['return_2'] = merged['close_2'].pct_change()
            
            # Remove NaN values
            merged = merged.dropna()
            
            if len(merged) < rolling_window:
                return None
            
            # Calculate correlations
            overall_correlation = merged['return_1'].corr(merged['return_2'])
            
            # Rolling correlation
            rolling_corr = merged['return_1'].rolling(window=rolling_window).corr(merged['return_2'])
            rolling_corr_list = rolling_corr.dropna().tolist()
            
            # Calculate metrics
            historical_avg = np.mean(rolling_corr_list) if rolling_corr_list else overall_correlation
            current_correlation = rolling_corr_list[-1] if rolling_corr_list else overall_correlation
            volatility = np.std(rolling_corr_list) if len(rolling_corr_list) > 1 else 0
            
            # Determine correlation strength
            abs_corr = abs(current_correlation)
            if abs_corr >= self.strong_correlation_threshold:
                strength = "strong"
            elif abs_corr >= self.moderate_correlation_threshold:
                strength = "moderate"
            else:
                strength = "weak"
            
            # Check for breakdown
            is_breakdown = False
            if abs(historical_avg) > self.strong_correlation_threshold:
                if abs(current_correlation) < self.breakdown_threshold:
                    is_breakdown = True
            
            return CorrelationData(
                pair1=pair1,
                pair2=pair2,
                timeframe=timeframe,
                correlation=current_correlation,
                rolling_correlation=rolling_corr_list[-10:],  # Last 10 values
                correlation_strength=strength,
                is_breakdown=is_breakdown,
                breakdown_threshold=self.breakdown_threshold,
                historical_avg=historical_avg,
                volatility=volatility,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating correlation {pair1}-{pair2} {timeframe}: {e}")
            return None
    
    def calculate_all_correlations(self, timeframe: str, periods: int = 100) -> Dict[Tuple[str, str], CorrelationData]:
        """Calculate correlations for all important pairs"""
        logger.info(f" Calculating correlations for {timeframe}...")
        
        correlations = {}
        total_pairs = len(self.important_pairs)
        
        for i, (pair1, pair2) in enumerate(self.important_pairs, 1):
            correlation_data = self.calculate_correlation(pair1, pair2, timeframe, periods)
            
            if correlation_data:
                correlations[(pair1, pair2)] = correlation_data
                
                if i % 10 == 0:  # Log progress every 10 pairs
                    logger.info(f" Progress: {i}/{total_pairs} pairs calculated")
        
        # Cache results
        with self.lock:
            cache_key = f"{timeframe}_{periods}"
            self.correlation_cache[cache_key] = correlations.copy()
            self.last_calculation[cache_key] = datetime.now()
        
        logger.info(f" Calculated {len(correlations)} correlations for {timeframe}")
        return correlations
    
    def get_cached_correlations(self, timeframe: str, periods: int = 100, 
                              max_age_minutes: int = 5) -> Optional[Dict[Tuple[str, str], CorrelationData]]:
        """Get cached correlation data if fresh enough"""
        cache_key = f"{timeframe}_{periods}"
        
        with self.lock:
            if (cache_key in self.correlation_cache and 
                cache_key in self.last_calculation):
                
                age = (datetime.now() - self.last_calculation[cache_key]).total_seconds() / 60
                if age <= max_age_minutes:
                    return self.correlation_cache[cache_key].copy()
        
        return None
    
    def find_correlation_breakdowns(self, timeframe: str) -> List[CorrelationData]:
        """Find pairs with correlation breakdown"""
        correlations = self.get_cached_correlations(timeframe)
        if correlations is None:
            correlations = self.calculate_all_correlations(timeframe)
        
        breakdowns = []
        for correlation_data in correlations.values():
            if correlation_data.is_breakdown:
                breakdowns.append(correlation_data)
        
        # Sort by breakdown severity (difference from historical average)
        breakdowns.sort(key=lambda x: abs(x.historical_avg - x.correlation), reverse=True)
        
        return breakdowns
    
    def find_strong_correlations(self, timeframe: str, min_correlation: float = 0.7) -> List[CorrelationData]:
        """Find pairs with strong correlations"""
        correlations = self.get_cached_correlations(timeframe)
        if correlations is None:
            correlations = self.calculate_all_correlations(timeframe)
        
        strong_correlations = []
        for correlation_data in correlations.values():
            if abs(correlation_data.correlation) >= min_correlation:
                strong_correlations.append(correlation_data)
        
        # Sort by correlation strength
        strong_correlations.sort(key=lambda x: abs(x.correlation), reverse=True)
        
        return strong_correlations
    
    def detect_arbitrage_opportunities(self, timeframe: str) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities based on correlation analysis"""
        logger.info(f" Detecting arbitrage opportunities for {timeframe}...")
        
        opportunities = []
        
        # 1. Correlation divergence opportunities
        breakdowns = self.find_correlation_breakdowns(timeframe)
        
        for breakdown in breakdowns:
            if abs(breakdown.historical_avg) > 0.7:  # Only strong historical correlations
                # Get currency strength for direction
                strength_ranking = self.strength_engine.get_strength_ranking(timeframe)
                strength_dict = {currency: strength for currency, strength in strength_ranking}
                
                # Determine trade direction based on strength
                pair1_base, pair1_quote = breakdown.pair1[:3], breakdown.pair1[3:]
                pair2_base, pair2_quote = breakdown.pair2[:3], breakdown.pair2[3:]
                
                # Calculate expected directions based on strength and correlation
                expected_direction = self._calculate_divergence_direction(
                    breakdown, strength_dict
                )
                
                if expected_direction:
                    profit_potential = abs(breakdown.historical_avg - breakdown.correlation) * 100
                    confidence = min(95, profit_potential * 20)
                    
                    opportunity = ArbitrageOpportunity(
                        pair1=breakdown.pair1,
                        pair2=breakdown.pair2,
                        pair3="",  # Not used for divergence
                        opportunity_type="correlation_divergence",
                        expected_direction=expected_direction,
                        profit_potential=profit_potential,
                        confidence=confidence,
                        risk_level="medium",
                        timestamp=datetime.now()
                    )
                    
                    opportunities.append(opportunity)
        
        # 2. Triangular arbitrage opportunities
        triangular_opportunities = self._find_triangular_arbitrage(timeframe)
        opportunities.extend(triangular_opportunities)
        
        # Sort by profit potential
        opportunities.sort(key=lambda x: x.profit_potential, reverse=True)
        
        # Cache results
        with self.lock:
            self.arbitrage_cache[timeframe] = opportunities.copy()
        
        logger.info(f" Found {len(opportunities)} arbitrage opportunities")
        return opportunities[:10]  # Top 10 opportunities
    
    def _calculate_divergence_direction(self, breakdown: CorrelationData, 
                                      strength_dict: Dict[str, float]) -> Optional[Dict[str, str]]:
        """Calculate trade directions for correlation divergence"""
        try:
            pair1_base, pair1_quote = breakdown.pair1[:3], breakdown.pair1[3:]
            pair2_base, pair2_quote = breakdown.pair2[:3], breakdown.pair2[3:]
            
            # Get strength values
            base1_strength = strength_dict.get(pair1_base, 0)
            quote1_strength = strength_dict.get(pair1_quote, 0)
            base2_strength = strength_dict.get(pair2_base, 0)
            quote2_strength = strength_dict.get(pair2_quote, 0)
            
            # Calculate pair strength difference
            pair1_strength = base1_strength - quote1_strength
            pair2_strength = base2_strength - quote2_strength
            
            # Historical correlation direction
            expected_correlation_sign = np.sign(breakdown.historical_avg)
            current_correlation_sign = np.sign(breakdown.correlation)
            
            # If correlation has broken down, trade in opposite directions
            if expected_correlation_sign != current_correlation_sign or abs(breakdown.correlation) < 0.3:
                if pair1_strength > pair2_strength:
                    return {
                        breakdown.pair1: "BUY",
                        breakdown.pair2: "SELL"
                    }
                else:
                    return {
                        breakdown.pair1: "SELL",
                        breakdown.pair2: "BUY"
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating divergence direction: {e}")
            return None
    
    def _find_triangular_arbitrage(self, timeframe: str) -> List[ArbitrageOpportunity]:
        """Find triangular arbitrage opportunities"""
        opportunities = []
        
        # Define triangular combinations
        triangles = [
            ("EURUSD", "GBPUSD", "EURGBP"),
            ("AUDUSD", "NZDUSD", "AUDNZD"),
            ("USDJPY", "EURJPY", "EURUSD"),
            ("USDCHF", "EURCHF", "EURUSD"),
        ]
        
        for triangle in triangles:
            pair1, pair2, pair3 = triangle
            
            # Check if all pairs are available
            if all(pair in self.currency_pairs for pair in triangle):
                # Get current prices
                data1 = self.data_manager.get_data(pair1, timeframe, 10)
                data2 = self.data_manager.get_data(pair2, timeframe, 10)
                data3 = self.data_manager.get_data(pair3, timeframe, 10)
                
                if all(data is not None and len(data) > 0 for data in [data1, data2, data3]):
                    price1 = data1['close'].iloc[-1]
                    price2 = data2['close'].iloc[-1]
                    price3 = data3['close'].iloc[-1]
                    
                    # Calculate arbitrage potential
                    arbitrage_potential = self._calculate_triangular_potential(
                        triangle, price1, price2, price3
                    )
                    
                    if arbitrage_potential['profit'] > 0.01:  # Minimum 1 pip profit
                        opportunity = ArbitrageOpportunity(
                            pair1=pair1,
                            pair2=pair2,
                            pair3=pair3,
                            opportunity_type="triangular_arbitrage",
                            expected_direction=arbitrage_potential['directions'],
                            profit_potential=arbitrage_potential['profit'] * 10000,  # Convert to pips
                            confidence=min(90, arbitrage_potential['profit'] * 1000),
                            risk_level="high",
                            timestamp=datetime.now()
                        )
                        
                        opportunities.append(opportunity)
        
        return opportunities
    
    def _calculate_triangular_potential(self, triangle: Tuple[str, str, str], 
                                      price1: float, price2: float, price3: float) -> Dict:
        """Calculate triangular arbitrage potential"""
        try:
            pair1, pair2, pair3 = triangle
            
            # Calculate cross rate vs actual rate
            if triangle == ("EURUSD", "GBPUSD", "EURGBP"):
                # EUR/USD ÷ GBP/USD should equal EUR/GBP
                calculated_eurgbp = price1 / price2
                actual_eurgbp = price3
                
                profit = abs(calculated_eurgbp - actual_eurgbp)
                
                if calculated_eurgbp > actual_eurgbp:
                    # Buy EURGBP, Sell EURUSD, Buy GBPUSD
                    directions = {pair1: "SELL", pair2: "BUY", pair3: "BUY"}
                else:
                    # Sell EURGBP, Buy EURUSD, Sell GBPUSD
                    directions = {pair1: "BUY", pair2: "SELL", pair3: "SELL"}
                
                return {'profit': profit, 'directions': directions}
            
            # Add more triangle calculations as needed
            return {'profit': 0, 'directions': {}}
            
        except Exception as e:
            logger.error(f"Error calculating triangular potential: {e}")
            return {'profit': 0, 'directions': {}}
    
    def get_correlation_matrix(self, timeframe: str) -> pd.DataFrame:
        """Get correlation matrix for all monitored pairs"""
        correlations = self.get_cached_correlations(timeframe)
        if correlations is None:
            correlations = self.calculate_all_correlations(timeframe)
        
        # Create matrix data
        matrix_data = []
        for (pair1, pair2), corr_data in correlations.items():
            matrix_data.append({
                'Pair1': pair1,
                'Pair2': pair2,
                'Correlation': round(corr_data.correlation, 3),
                'Strength': corr_data.correlation_strength,
                'Historical_Avg': round(corr_data.historical_avg, 3),
                'Breakdown': corr_data.is_breakdown,
                'Volatility': round(corr_data.volatility, 3)
            })
        
        df = pd.DataFrame(matrix_data)
        return df.sort_values('Correlation', key=abs, ascending=False)
    
    def get_correlation_summary(self, timeframe: str) -> Dict:
        """Get correlation analysis summary"""
        correlations = self.get_cached_correlations(timeframe)
        if correlations is None:
            correlations = self.calculate_all_correlations(timeframe)
        
        strong_positive = sum(1 for c in correlations.values() if c.correlation > 0.7)
        strong_negative = sum(1 for c in correlations.values() if c.correlation < -0.7)
        moderate = sum(1 for c in correlations.values() if 0.4 <= abs(c.correlation) < 0.7)
        weak = sum(1 for c in correlations.values() if abs(c.correlation) < 0.4)
        breakdowns = sum(1 for c in correlations.values() if c.is_breakdown)
        
        return {
            'total_pairs': len(correlations),
            'strong_positive': strong_positive,
            'strong_negative': strong_negative,
            'moderate': moderate,
            'weak': weak,
            'breakdowns': breakdowns,
            'breakdown_rate': round(breakdowns / len(correlations) * 100, 1) if correlations else 0
        }

# Testing and example usage
if __name__ == "__main__":
    print("🔗 Correlation Engine Test")
    print("=" * 50)
    
    # Import required modules
    import sys
    import os
    
    # Add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    try:
        from core.data_manager import DataManager
        from core.currency_strength import CurrencyStrengthEngine
    except ImportError:
        # Alternative import if running from different directory
        from data_manager import DataManager
        from currency_strength import CurrencyStrengthEngine
    
    # Initialize
    dm = DataManager()
    
    if dm.connect_mt5():
        print(" MT5 connected")
        
        # Create engines
        strength_engine = CurrencyStrengthEngine(dm)
        correlation_engine = CorrelationEngine(dm, strength_engine)
        
        # Test H1 timeframe
        print("\n Testing H1 Correlation Analysis...")
        
        # Calculate correlations
        correlations = correlation_engine.calculate_all_correlations("H1")
        
        # Get summary
        summary = correlation_engine.get_correlation_summary("H1")
        print(f" Correlation Summary (H1):")
        print(f"Total Pairs: {summary['total_pairs']}")
        print(f"Strong Positive: {summary['strong_positive']}")
        print(f"Strong Negative: {summary['strong_negative']}")
        print(f"Moderate: {summary['moderate']}")
        print(f"Weak: {summary['weak']}")
        print(f"Breakdowns: {summary['breakdowns']} ({summary['breakdown_rate']}%)")
        
        # Find strong correlations
        strong_correlations = correlation_engine.find_strong_correlations("H1")
        print(f"\n💪 Strong Correlations (H1):")
        for i, corr in enumerate(strong_correlations[:5], 1):
            print(f"{i}. {corr.pair1} vs {corr.pair2}: {corr.correlation:+.3f} ({corr.correlation_strength})")
        
        # Find breakdowns
        breakdowns = correlation_engine.find_correlation_breakdowns("H1")
        if breakdowns:
            print(f"\n Correlation Breakdowns (H1):")
            for i, breakdown in enumerate(breakdowns[:3], 1):
                print(f"{i}. {breakdown.pair1} vs {breakdown.pair2}: {breakdown.correlation:.3f} (was {breakdown.historical_avg:.3f})")
        
        # Find arbitrage opportunities
        opportunities = correlation_engine.detect_arbitrage_opportunities("H1")
        if opportunities:
            print(f"\n Arbitrage Opportunities (H1):")
            for i, opp in enumerate(opportunities[:3], 1):
                print(f"{i}. {opp.opportunity_type}: {opp.pair1} & {opp.pair2}")
                print(f"   Directions: {opp.expected_direction}")
                print(f"   Profit Potential: {opp.profit_potential:.2f} | Confidence: {opp.confidence:.0f}%")
        
    else:
        print("❌ MT5 connection failed")
    
    # Cleanup
    dm.close()
    print("\n✅ Test completed")