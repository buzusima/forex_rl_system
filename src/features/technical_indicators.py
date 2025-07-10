# src/features/technical_indicators.py - Basic Technical Indicators for RL
"""
ไฟล์นี้คำนวณ technical indicators พื้นฐาน: RSI, MACD, ATR, Bollinger Bands
สำหรับใช้เป็น features ใน RL model ขั้นแรก
แก้ไขไฟล์นี้เมื่อ: ต้องการปรับ parameters หรือเพิ่ม indicators ใหม่
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Basic Technical Indicators Calculator"""
    
    def __init__(self, config=None):
        from config.config import ForexRLConfig
        self.config = config or ForexRLConfig()
        
        # Default parameters (from config)
        self.rsi_period = self.config.INDICATORS_CONFIG['RSI']['period']
        self.macd_fast = self.config.INDICATORS_CONFIG['MACD']['fast']
        self.macd_slow = self.config.INDICATORS_CONFIG['MACD']['slow'] 
        self.macd_signal = self.config.INDICATORS_CONFIG['MACD']['signal']
        self.atr_period = self.config.INDICATORS_CONFIG['ATR']['period']
        self.bb_period = self.config.INDICATORS_CONFIG['BOLLINGER_BANDS']['period']
        self.bb_std = self.config.INDICATORS_CONFIG['BOLLINGER_BANDS']['std']
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """คำนวณ RSI (Relative Strength Index)"""
        period = period or self.rsi_period
        
        try:
            close = data['close'].copy()
            
            # คำนวณ price changes
            delta = close.diff()
            
            # แยก gains และ losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # คำนวณ average gains และ losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # คำนวณ RS และ RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            logger.debug(f"RSI calculated with period {period}")
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_macd(self, data: pd.DataFrame, 
                      fast: int = None, slow: int = None, signal: int = None) -> Dict[str, pd.Series]:
        """คำนวณ MACD (Moving Average Convergence Divergence)"""
        fast = fast or self.macd_fast
        slow = slow or self.macd_slow
        signal = signal or self.macd_signal
        
        try:
            close = data['close'].copy()
            
            # คำนวณ EMAs
            ema_fast = close.ewm(span=fast).mean()
            ema_slow = close.ewm(span=slow).mean()
            
            # คำนวณ MACD line
            macd_line = ema_fast - ema_slow
            
            # คำนวณ Signal line
            signal_line = macd_line.ewm(span=signal).mean()
            
            # คำนวณ Histogram
            histogram = macd_line - signal_line
            
            logger.debug(f"MACD calculated with {fast}/{slow}/{signal}")
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {
                'macd': pd.Series(index=data.index, dtype=float),
                'signal': pd.Series(index=data.index, dtype=float),
                'histogram': pd.Series(index=data.index, dtype=float)
            }
    
    def calculate_atr(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """คำนวณ ATR (Average True Range)"""
        period = period or self.atr_period
        
        try:
            high = data['high'].copy()
            low = data['low'].copy()
            close = data['close'].copy()
            
            # คำนวณ True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            # เลือก True Range ที่สูงสุด
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # คำนวณ ATR (moving average ของ True Range)
            atr = true_range.rolling(window=period).mean()
            
            logger.debug(f"ATR calculated with period {period}")
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, 
                                 period: int = None, std: float = None) -> Dict[str, pd.Series]:
        """คำนวณ Bollinger Bands"""
        period = period or self.bb_period
        std = std or self.bb_std
        
        try:
            close = data['close'].copy()
            
            # คำนวณ Middle line (SMA)
            middle = close.rolling(window=period).mean()
            
            # คำนวณ Standard deviation
            std_dev = close.rolling(window=period).std()
            
            # คำนวณ Upper และ Lower bands
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            # คำนวณ Band width และ %B
            band_width = (upper - lower) / middle
            percent_b = (close - lower) / (upper - lower)
            
            logger.debug(f"Bollinger Bands calculated with period {period}, std {std}")
            
            return {
                'bb_upper': upper,
                'bb_middle': middle,
                'bb_lower': lower,
                'bb_width': band_width,
                'bb_percent': percent_b
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {
                'bb_upper': pd.Series(index=data.index, dtype=float),
                'bb_middle': pd.Series(index=data.index, dtype=float),
                'bb_lower': pd.Series(index=data.index, dtype=float),
                'bb_width': pd.Series(index=data.index, dtype=float),
                'bb_percent': pd.Series(index=data.index, dtype=float)
            }
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """คำนวณ indicators ทั้งหมดในครั้งเดียว"""
        try:
            result_data = data.copy()
            
            logger.info(f"Calculating all indicators for {len(data)} bars")
            
            # RSI
            result_data['rsi'] = self.calculate_rsi(data)
            
            # MACD
            macd_data = self.calculate_macd(data)
            result_data['macd'] = macd_data['macd']
            result_data['macd_signal'] = macd_data['signal']
            result_data['macd_histogram'] = macd_data['histogram']
            
            # ATR
            result_data['atr'] = self.calculate_atr(data)
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(data)
            result_data['bb_upper'] = bb_data['bb_upper']
            result_data['bb_middle'] = bb_data['bb_middle']
            result_data['bb_lower'] = bb_data['bb_lower']
            result_data['bb_width'] = bb_data['bb_width']
            result_data['bb_percent'] = bb_data['bb_percent']
            
            # เพิ่ม basic price features
            result_data['price_change'] = data['close'].pct_change()
            result_data['price_change_abs'] = abs(result_data['price_change'])
            
            # ลบ NaN values
            result_data = result_data.dropna()
            
            logger.info(f"✅ All indicators calculated: {len(result_data)} valid bars")
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error calculating all indicators: {e}")
            return data.copy()
    
    def get_feature_columns(self) -> List[str]:
        """รายชื่อ feature columns ที่สร้างขึ้น"""
        return [
            'rsi',
            'macd', 'macd_signal', 'macd_histogram',
            'atr',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            'price_change', 'price_change_abs'
        ]
    
    def normalize_features(self, data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """Normalize features สำหรับ RL"""
        try:
            normalized_data = data.copy()
            feature_cols = self.get_feature_columns()
            
            # เลือกเฉพาะ feature columns ที่มีอยู่
            available_features = [col for col in feature_cols if col in data.columns]
            
            if method == 'minmax':
                # MinMax normalization (0-1)
                for col in available_features:
                    if col in ['rsi', 'bb_percent']:
                        # RSI และ %B อยู่ในช่วง 0-100 และ 0-1 อยู่แล้ว
                        if col == 'rsi':
                            normalized_data[col] = data[col] / 100.0
                        # bb_percent อยู่ในช่วง 0-1 อยู่แล้ว
                    else:
                        # Normalize อื่นๆ
                        col_min = data[col].min()
                        col_max = data[col].max()
                        if col_max != col_min:
                            normalized_data[col] = (data[col] - col_min) / (col_max - col_min)
                        else:
                            normalized_data[col] = 0.5  # ค่าคงที่
            
            elif method == 'zscore':
                # Z-score normalization
                for col in available_features:
                    normalized_data[col] = (data[col] - data[col].mean()) / data[col].std()
            
            logger.info(f"Features normalized using {method} method")
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return data.copy()

def create_indicators_calculator(config=None) -> TechnicalIndicators:
    """สร้าง TechnicalIndicators instance"""
    return TechnicalIndicators(config)

def test_indicators_calculation():
    """ทดสอบการคำนวณ indicators"""
    try:
        print("🧪 Testing Technical Indicators Calculation...")
        
        # สร้าง sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15T')
        
        # สร้างข้อมูลราคาจำลอง
        np.random.seed(42)
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.0001, 100)
        close_prices = base_price + np.cumsum(price_changes)
        
        sample_data = pd.DataFrame({
            'open': close_prices + np.random.normal(0, 0.00005, 100),
            'high': close_prices + abs(np.random.normal(0, 0.0001, 100)),
            'low': close_prices - abs(np.random.normal(0, 0.0001, 100)),
            'close': close_prices,
            'tick_volume': np.random.randint(50, 200, 100)
        }, index=dates)
        
        print(f"✅ Sample data created: {len(sample_data)} bars")
        
        # สร้าง indicators calculator
        calculator = create_indicators_calculator()
        
        # คำนวณ indicators
        data_with_indicators = calculator.calculate_all_indicators(sample_data)
        
        print(f"✅ Indicators calculated: {len(data_with_indicators)} valid bars")
        
        # แสดงผลลัพธ์
        feature_cols = calculator.get_feature_columns()
        available_features = [col for col in feature_cols if col in data_with_indicators.columns]
        
        print(f"\n📊 Available Features: {len(available_features)}")
        for feature in available_features:
            value = data_with_indicators[feature].iloc[-1]
            print(f"  {feature}: {value:.6f}")
        
        # ทดสอบ normalization
        normalized_data = calculator.normalize_features(data_with_indicators)
        print(f"\n✅ Features normalized")
        
        print("\n🎉 Technical Indicators test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Technical Indicators test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test
    test_indicators_calculation()