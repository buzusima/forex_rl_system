# src/rl/evaluation_visualization.py - Advanced Evaluation และ Visualization
"""
ไฟล์นี้สร้างระบบ evaluation และ visualization ที่ครอบคลุม
ใช้ประสบการณ์จากการเทรดจริงมากกว่า 15 ปี
แก้ไขไฟล์นี้เมื่อ: ต้องการเพิ่ม metrics หรือ visualization ใหม่
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Setup plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance Metrics Container"""
    
    # Return Metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    excess_return: float = 0.0
    
    # Risk Metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Trading Metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Risk-Adjusted Metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    maximum_consecutive_losses: int = 0
    recovery_factor: float = 0.0
    
    # Market Regime Performance
    bull_market_return: float = 0.0
    bear_market_return: float = 0.0
    sideways_market_return: float = 0.0

class PerformanceAnalyzer:
    """Advanced Performance Analysis"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(self, 
                         equity_curve: List[float],
                         trade_history: List[Dict] = None,
                         price_data: pd.DataFrame = None,
                         timeframe: str = "M15") -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if len(equity_curve) < 2:
            return PerformanceMetrics()
        
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Time scaling factors
        periods_per_year = self._get_periods_per_year(timeframe)
        
        # Return Metrics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (periods_per_year / len(equity_series)) - 1
        excess_return = annualized_return - self.risk_free_rate
        
        # Risk Metrics
        volatility = returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0.0
        
        # Drawdown calculation
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Risk-adjusted ratios
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0.0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0.0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
        cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0.0
        
        # Trading metrics
        metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            excess_return=excess_return,
            volatility=volatility,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95
        )
        
        # Trade-specific metrics
        if trade_history:
            trade_metrics = self._calculate_trade_metrics(trade_history)
            for key, value in trade_metrics.items():
                setattr(metrics, key, value)
        
        # Market regime analysis
        if price_data is not None:
            regime_metrics = self._analyze_market_regimes(equity_curve, price_data)
            for key, value in regime_metrics.items():
                setattr(metrics, key, value)
        
        return metrics
    
    def _get_periods_per_year(self, timeframe: str) -> int:
        """Get number of periods per year based on timeframe"""
        timeframe_multipliers = {
            'M1': 525600,    # 1 minute
            'M5': 105120,    # 5 minutes
            'M15': 35040,    # 15 minutes
            'M30': 17520,    # 30 minutes
            'H1': 8760,      # 1 hour
            'H4': 2190,      # 4 hours
            'D1': 365        # 1 day
        }
        return timeframe_multipliers.get(timeframe, 35040)  # Default to M15
    
    def _calculate_trade_metrics(self, trade_history: List[Dict]) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        if not trade_history:
            return {}
        
        # Extract PnL values
        pnls = [trade.get('pnl', 0) for trade in trade_history if 'pnl' in trade]
        
        if not pnls:
            return {}
        
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        total_trades = len(pnls)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        avg_trade = np.mean(pnls)
        
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1.0  # Avoid division by zero
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for pnl in pnls:
            if pnl < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'maximum_consecutive_losses': max_consecutive_losses
        }
    
    def _analyze_market_regimes(self, equity_curve: List[float], price_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze performance in different market regimes"""
        if len(equity_curve) != len(price_data):
            return {}
        
        # Calculate price trends
        price_returns = price_data['close'].pct_change().rolling(20).mean()
        
        # Define market regimes
        bull_mask = price_returns > 0.001    # Strong uptrend
        bear_mask = price_returns < -0.001   # Strong downtrend
        sideways_mask = ~(bull_mask | bear_mask)  # Sideways market
        
        equity_series = pd.Series(equity_curve)
        equity_returns = equity_series.pct_change()
        
        # Calculate regime-specific returns
        bull_return = equity_returns[bull_mask].sum() if bull_mask.any() else 0.0
        bear_return = equity_returns[bear_mask].sum() if bear_mask.any() else 0.0
        sideways_return = equity_returns[sideways_mask].sum() if sideways_mask.any() else 0.0
        
        return {
            'bull_market_return': bull_return,
            'bear_market_return': bear_return,
            'sideways_market_return': sideways_return
        }

class TradingVisualizer:
    """Advanced Trading Visualization System"""
    
    def __init__(self, style: str = "professional"):
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Setup visualization style"""
        if self.style == "professional":
            plt.rcParams.update({
                'figure.figsize': (12, 8),
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'font.family': 'sans-serif',
                'axes.grid': True,
                'grid.alpha': 0.3
            })
    
    def plot_equity_curve(self, 
                         equity_curve: List[float],
                         timestamps: List[datetime] = None,
                         benchmark: List[float] = None,
                         trades: List[Dict] = None,
                         title: str = "Equity Curve") -> plt.Figure:
        """Plot comprehensive equity curve"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[3, 1, 1])
        
        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=len(equity_curve), freq='15T')
        
        equity_series = pd.Series(equity_curve, index=timestamps)
        
        # Main equity curve
        ax1.plot(timestamps, equity_curve, 'b-', linewidth=2, label='Strategy', alpha=0.8)
        
        # Benchmark comparison
        if benchmark is not None and len(benchmark) == len(equity_curve):
            ax1.plot(timestamps, benchmark, 'g--', linewidth=1.5, label='Benchmark', alpha=0.7)
        
        # Mark trades
        if trades:
            buy_times = []
            sell_times = []
            buy_prices = []
            sell_prices = []
            
            for trade in trades:
                if trade.get('action') == 'BUY':
                    buy_times.append(timestamps[trade.get('step', 0)])
                    buy_prices.append(equity_curve[trade.get('step', 0)])
                elif trade.get('action') == 'SELL':
                    sell_times.append(timestamps[trade.get('step', 0)])
                    sell_prices.append(equity_curve[trade.get('step', 0)])
            
            if buy_times:
                ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=50, alpha=0.7, label='Buy')
            if sell_times:
                ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=50, alpha=0.7, label='Sell')
        
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown plot
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        
        ax2.fill_between(timestamps, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(timestamps, drawdown, 'r-', linewidth=1)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdown')
        ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        returns = equity_series.pct_change().dropna() * 100
        ax3.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.3f}%')
        ax3.axvline(returns.quantile(0.05), color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {returns.quantile(0.05):.3f}%')
        ax3.set_xlabel('Returns (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Returns Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_performance_metrics(self, metrics: PerformanceMetrics) -> plt.Figure:
        """Plot performance metrics dashboard"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # Risk-Return Scatter
        risk_return_data = {
            'Strategy': (metrics.volatility, metrics.annualized_return),
            'Risk-Free': (0, self.risk_free_rate if hasattr(self, 'risk_free_rate') else 0.02)
        }
        
        for label, (risk, ret) in risk_return_data.items():
            color = 'blue' if label == 'Strategy' else 'red'
            axes[0, 0].scatter(risk, ret, s=100, c=color, label=label, alpha=0.7)
        
        axes[0, 0].set_xlabel('Volatility (Annual)')
        axes[0, 0].set_ylabel('Return (Annual)')
        axes[0, 0].set_title('Risk-Return Profile')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Key Ratios Bar Chart
        ratios = {
            'Sharpe': metrics.sharpe_ratio,
            'Sortino': metrics.sortino_ratio,
            'Calmar': metrics.calmar_ratio
        }
        
        bars = axes[0, 1].bar(ratios.keys(), ratios.values(), color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 1].set_title('Risk-Adjusted Ratios')
        axes[0, 1].set_ylabel('Ratio Value')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, ratios.values()):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Trading Statistics
        trading_stats = {
            'Win Rate': metrics.win_rate * 100,
            'Profit Factor': metrics.profit_factor,
            'Avg Trade': metrics.avg_trade
        }
        
        colors = ['green' if v > 0 else 'red' for v in trading_stats.values()]
        bars = axes[0, 2].bar(trading_stats.keys(), trading_stats.values(), color=colors, alpha=0.7)
        axes[0, 2].set_title('Trading Statistics')
        axes[0, 2].set_ylabel('Value')
        
        # Risk Metrics
        risk_metrics = {
            'Max DD': metrics.max_drawdown * 100,
            'VaR 95%': abs(metrics.var_95) * 100,
            'CVaR 95%': abs(metrics.cvar_95) * 100
        }
        
        bars = axes[1, 0].bar(risk_metrics.keys(), risk_metrics.values(), color='red', alpha=0.6)
        axes[1, 0].set_title('Risk Metrics (%)')
        axes[1, 0].set_ylabel('Percentage')
        
        # Market Regime Performance
        regime_performance = {
            'Bull Market': metrics.bull_market_return * 100,
            'Bear Market': metrics.bear_market_return * 100,
            'Sideways': metrics.sideways_market_return * 100
        }
        
        colors = ['green', 'red', 'blue']
        bars = axes[1, 1].bar(regime_performance.keys(), regime_performance.values(), color=colors, alpha=0.7)
        axes[1, 1].set_title('Market Regime Performance (%)')
        axes[1, 1].set_ylabel('Return (%)')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Summary Table
        axes[1, 2].axis('off')
        summary_data = [
            ['Total Return', f'{metrics.total_return*100:.2f}%'],
            ['Annual Return', f'{metrics.annualized_return*100:.2f}%'],
            ['Volatility', f'{metrics.volatility*100:.2f}%'],
            ['Max Drawdown', f'{metrics.max_drawdown*100:.2f}%'],
            ['Sharpe Ratio', f'{metrics.sharpe_ratio:.2f}'],
            ['Total Trades', f'{metrics.total_trades}'],
            ['Win Rate', f'{metrics.win_rate*100:.1f}%']
        ]
        
        table = axes[1, 2].table(cellText=summary_data,
                               colLabels=['Metric', 'Value'],
                               cellLoc='center',
                               loc='center',
                               bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        axes[1, 2].set_title('Performance Summary')
        
        plt.tight_layout()
        return fig
    
    def plot_action_analysis(self, 
                           actions: List[int],
                           prices: List[float],
                           timestamps: List[datetime] = None,
                           action_names: List[str] = ['Hold', 'Buy', 'Sell']) -> plt.Figure:
        """Analyze agent's action patterns"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Agent Action Analysis', fontsize=16, fontweight='bold')
        
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=len(actions), freq='15T')
        
        # Action frequency
        action_counts = pd.Series(actions).value_counts().sort_index()
        colors = ['gray', 'green', 'red']
        
        bars = axes[0, 0].bar([action_names[i] for i in action_counts.index], 
                             action_counts.values, 
                             color=[colors[i] for i in action_counts.index],
                             alpha=0.7)
        axes[0, 0].set_title('Action Frequency')
        axes[0, 0].set_ylabel('Count')
        
        # Add percentage labels
        total_actions = sum(action_counts.values)
        for bar, count in zip(bars, action_counts.values):
            pct = count / total_actions * 100
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_actions*0.01,
                           f'{pct:.1f}%', ha='center', va='bottom')
        
        # Price and actions over time
        axes[0, 1].plot(timestamps, prices, 'b-', alpha=0.7, linewidth=1)
        
        # Mark actions on price chart
        for i, (action, price, time) in enumerate(zip(actions, prices, timestamps)):
            if action == 1:  # Buy
                axes[0, 1].scatter(time, price, color='green', marker='^', s=30, alpha=0.7)
            elif action == 2:  # Sell
                axes[0, 1].scatter(time, price, color='red', marker='v', s=30, alpha=0.7)
        
        axes[0, 1].set_title('Actions on Price Chart')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].legend(['Price', 'Buy', 'Sell'])
        
        # Action timing analysis (hour of day)
        if timestamps:
            hours = [t.hour for t in timestamps]
            action_by_hour = pd.DataFrame({'hour': hours, 'action': actions})
            
            for action_type in [1, 2]:  # Buy and Sell
                action_data = action_by_hour[action_by_hour['action'] == action_type]
                if not action_data.empty:
                    hour_counts = action_data['hour'].value_counts().sort_index()
                    color = 'green' if action_type == 1 else 'red'
                    label = action_names[action_type]
                    axes[1, 0].plot(hour_counts.index, hour_counts.values, 
                                   marker='o', color=color, label=label, alpha=0.7)
            
            axes[1, 0].set_title('Action Timing (Hour of Day)')
            axes[1, 0].set_xlabel('Hour')
            axes[1, 0].set_ylabel('Action Count')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Action transition matrix
        action_transitions = np.zeros((3, 3))
        for i in range(len(actions) - 1):
            current_action = actions[i]
            next_action = actions[i + 1]
            action_transitions[current_action, next_action] += 1
        
        # Normalize to probabilities
        row_sums = action_transitions.sum(axis=1, keepdims=True)
        action_transitions_norm = np.divide(action_transitions, row_sums, 
                                          out=np.zeros_like(action_transitions), 
                                          where=row_sums!=0)
        
        im = axes[1, 1].imshow(action_transitions_norm, cmap='Blues', aspect='auto')
        axes[1, 1].set_title('Action Transition Probabilities')
        axes[1, 1].set_xticks(range(3))
        axes[1, 1].set_yticks(range(3))
        axes[1, 1].set_xticklabels(action_names)
        axes[1, 1].set_yticklabels(action_names)
        axes[1, 1].set_xlabel('Next Action')
        axes[1, 1].set_ylabel('Current Action')
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = axes[1, 1].text(j, i, f'{action_transitions_norm[i, j]:.2f}',
                                     ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=axes[1, 1])
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, 
                                   equity_curve: List[float],
                                   prices: List[float],
                                   actions: List[int],
                                   timestamps: List[datetime] = None) -> go.Figure:
        """Create interactive Plotly dashboard"""
        
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=len(equity_curve), freq='15T')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curve', 'Price Chart with Actions', 
                          'Drawdown', 'Action Distribution',
                          'Returns Distribution', 'Cumulative Returns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}],
                   [{"type": "histogram"}, {"secondary_y": False}]],
            vertical_spacing=0.08
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(x=timestamps, y=equity_curve, mode='lines', name='Equity',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Price chart with actions
        fig.add_trace(
            go.Scatter(x=timestamps, y=prices, mode='lines', name='Price',
                      line=dict(color='black', width=1)),
            row=1, col=2
        )
        
        # Add buy/sell markers
        buy_times = [timestamps[i] for i, action in enumerate(actions) if action == 1]
        buy_prices = [prices[i] for i, action in enumerate(actions) if action == 1]
        sell_times = [timestamps[i] for i, action in enumerate(actions) if action == 2]
        sell_prices = [prices[i] for i, action in enumerate(actions) if action == 2]
        
        if buy_times:
            fig.add_trace(
                go.Scatter(x=buy_times, y=buy_prices, mode='markers', name='Buy',
                          marker=dict(color='green', symbol='triangle-up', size=8)),
                row=1, col=2
            )
        
        if sell_times:
            fig.add_trace(
                go.Scatter(x=sell_times, y=sell_prices, mode='markers', name='Sell',
                          marker=dict(color='red', symbol='triangle-down', size=8)),
                row=1, col=2
            )
        
        # Drawdown
        equity_series = pd.Series(equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=drawdown, mode='lines', name='Drawdown',
                      fill='tonexty', line=dict(color='red')),
            row=2, col=1
        )
        
        # Action distribution
        action_counts = pd.Series(actions).value_counts()
        action_names = ['Hold', 'Buy', 'Sell']
        
        fig.add_trace(
            go.Pie(labels=[action_names[i] for i in action_counts.index],
                   values=action_counts.values,
                   name="Actions"),
            row=2, col=2
        )
        
        # Returns distribution
        returns = equity_series.pct_change().dropna() * 100
        
        fig.add_trace(
            go.Histogram(x=returns, name='Returns', nbinsx=50),
            row=3, col=1
        )
        
        # Cumulative returns
        cumulative_returns = (equity_series / equity_series.iloc[0] - 1) * 100
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=cumulative_returns, mode='lines', 
                      name='Cumulative Returns', line=dict(color='purple', width=2)),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Interactive Trading Dashboard",
            showlegend=True
        )
        
        return fig

class EvaluationSystem:
    """Complete Evaluation System สำหรับ PPO Forex Agent"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.analyzer = PerformanceAnalyzer(risk_free_rate)
        self.visualizer = TradingVisualizer()
        
    def evaluate_agent(self, 
                      agent,
                      env,
                      n_episodes: int = 10,
                      deterministic: bool = True,
                      save_results: bool = True,
                      results_path: str = "results/evaluation") -> Dict[str, Any]:
        """Complete agent evaluation"""
        
        logger.info(f"🔍 Starting agent evaluation ({n_episodes} episodes)")
        
        all_equity_curves = []
        all_metrics = []
        all_actions = []
        all_prices = []
        all_timestamps = []
        all_trades = []
        
        for episode in range(n_episodes):
            logger.info(f"Evaluating episode {episode + 1}/{n_episodes}")
            
            # Run episode
            obs = env.reset()
            done = False
            
            equity_curve = [env.balance]
            actions = []
            prices = []
            trades = []
            
            while not done:
                action, _, _ = agent.get_action(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                
                equity_curve.append(info['current_equity'])
                actions.append(action)
                prices.append(info['current_price'])
                
                # Record trades
                if hasattr(env, 'trade_history') and env.trade_history:
                    if len(env.trade_history) > len(trades):
                        trades.extend(env.trade_history[len(trades):])
            
            # Generate timestamps
            timestamps = pd.date_range(
                start='2024-01-01', 
                periods=len(equity_curve), 
                freq='15T'
            )
            
            # Calculate metrics
            metrics = self.analyzer.calculate_metrics(
                equity_curve=equity_curve,
                trade_history=trades,
                timeframe="M15"
            )
            
            # Store results
            all_equity_curves.append(equity_curve)
            all_metrics.append(metrics)
            all_actions.extend(actions)
            all_prices.extend(prices)
            all_timestamps.extend(timestamps[:len(actions)])
            all_trades.extend(trades)
        
        # Aggregate metrics
        avg_metrics = self._aggregate_metrics(all_metrics)
        
        # Create comprehensive report
        evaluation_results = {
            'summary_metrics': avg_metrics,
            'individual_episodes': all_metrics,
            'equity_curves': all_equity_curves,
            'all_actions': all_actions,
            'all_prices': all_prices,
            'all_timestamps': all_timestamps,
            'all_trades': all_trades,
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Generate visualizations
        if save_results:
            self._save_evaluation_results(evaluation_results, results_path)
        
        logger.info("✅ Agent evaluation completed")
        return evaluation_results
    
    def _aggregate_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Aggregate metrics across episodes"""
        if not metrics_list:
            return PerformanceMetrics()
        
        # Calculate mean values
        avg_metrics = PerformanceMetrics()
        
        for field in avg_metrics.__dataclass_fields__:
            values = [getattr(m, field) for m in metrics_list]
            avg_value = np.mean(values) if values else 0.0
            setattr(avg_metrics, field, avg_value)
        
        return avg_metrics
    
    def _save_evaluation_results(self, results: Dict[str, Any], base_path: str):
        """Save evaluation results and visualizations"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save comprehensive plots
        if results['equity_curves']:
            # Use first episode for detailed analysis
            equity_curve = results['equity_curves'][0]
            timestamps = results['all_timestamps'][:len(equity_curve)]
            
            # Equity curve plot
            fig1 = self.visualizer.plot_equity_curve(
                equity_curve=equity_curve,
                timestamps=timestamps,
                trades=results['all_trades'],
                title="Agent Performance - Detailed Analysis"
            )
            fig1.savefig(f"{base_path}/equity_curve_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # Performance metrics plot
            fig2 = self.visualizer.plot_performance_metrics(results['summary_metrics'])
            fig2.savefig(f"{base_path}/performance_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            # Action analysis plot
            if results['all_actions'] and results['all_prices']:
                sample_size = min(1000, len(results['all_actions']))  # Limit for visualization
                fig3 = self.visualizer.plot_action_analysis(
                    actions=results['all_actions'][:sample_size],
                    prices=results['all_prices'][:sample_size],
                    timestamps=results['all_timestamps'][:sample_size]
                )
                fig3.savefig(f"{base_path}/action_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
                plt.close(fig3)
        
        logger.info(f"📊 Evaluation visualizations saved to {base_path}")

def create_evaluation_system(risk_free_rate: float = 0.02) -> EvaluationSystem:
    """Create evaluation system instance"""
    return EvaluationSystem(risk_free_rate)

if __name__ == "__main__":
    # Test evaluation system
    print("🧪 Testing Evaluation System...")
    
    # Create dummy data for testing
    np.random.seed(42)
    equity_curve = [10000]
    for i in range(1000):
        change = np.random.normal(0.001, 0.02)
        equity_curve.append(equity_curve[-1] * (1 + change))
    
    # Test performance analyzer
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(equity_curve)
    
    print(f"✅ Performance Analyzer test:")
    print(f"   Total Return: {metrics.total_return:.4f}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.4f}")
    
    # Test visualizer
    visualizer = TradingVisualizer()
    fig = visualizer.plot_equity_curve(equity_curve)
    plt.close(fig)
    
    print(f"✅ Visualizer test completed")
    print(f"🎉 Evaluation System ready!")