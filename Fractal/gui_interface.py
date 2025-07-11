import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import json
import threading
import time
from datetime import datetime
from typing import Dict, Callable, Optional
import MetaTrader5 as mt5

class TradingGUI:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.root = tk.Tk()
        
        # External managers (will be injected)
        self.login_manager = None
        self.symbol_manager = None
        self.trading_engine = None
        
        # GUI State
        self.is_trading = False
        self.last_update = None
        self.update_thread = None
        self.update_running = False
        
        # Data storage
        self.current_positions = []
        self.daily_stats = {'pnl': 0, 'trades': 0, 'wins': 0, 'losses': 0}
        self.signal_history = []
        
        # Setup GUI
        self.setup_window()
        self.create_widgets()
        self.load_settings_to_gui()
        self.start_auto_update()
        
    def load_config(self) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            "trading_settings": {
                "symbol": "XAUUSD",
                "lot_size": 0.01,
                "max_positions": 5,
                "anti_hedge": True,
                "trading_enabled": False
            },
            "signal_parameters": {
                "rsi_period": 14,
                "rsi_up": 55,
                "rsi_down": 45,
                "fractal_period": 5
            },
            "timeframes": {
                "entry_tf": "M1",
                "trend_tf": "M15",
                "bias_tf": "H1",
                "require_tf_alignment": True
            },
            "risk_management": {
                "daily_loss_limit": 100
            },
            "monitoring": {
                "check_interval_seconds": 30
            },
            "presets": {
                "scalping": {
                    "entry_tf": "M1",
                    "trend_tf": "M5",
                    "bias_tf": "M15",
                    "rsi_up": 60,
                    "rsi_down": 40,
                    "check_interval_seconds": 15
                },
                "intraday": {
                    "entry_tf": "M15",
                    "trend_tf": "H1", 
                    "bias_tf": "H4",
                    "rsi_up": 55,
                    "rsi_down": 45,
                    "check_interval_seconds": 60
                },
                "swing": {
                    "entry_tf": "H1",
                    "trend_tf": "H4",
                    "bias_tf": "D1", 
                    "rsi_up": 50,
                    "rsi_down": 50,
                    "check_interval_seconds": 300
                }
            }
        }
    
    def setup_window(self):
        """Setup main window"""
        self.root.title("XAUUSD Multi-Timeframe Trading System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # Configure style
        style = ttk.Style()
        
        # Try to set dark theme if available
        try:
            style.theme_use('clam')
        except:
            style.theme_use('default')
        
        # Configure colors
        self.colors = {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'success': '#4caf50',
            'error': '#f44336',
            'warning': '#ff9800',
            'info': '#2196f3'
        }
        
        self.root.configure(bg=self.colors['bg'])
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_control_tab()
        self.create_settings_tab()
        self.create_monitoring_tab()
        self.create_positions_tab()
        self.create_logs_tab()
        
        # Create status bar
        self.create_status_bar()
    
    def create_control_tab(self):
        """Create main control tab"""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="Control Panel")
        
        # Trading Controls Section
        controls_group = ttk.LabelFrame(control_frame, text="Trading Controls", padding=10)
        controls_group.pack(fill='x', padx=5, pady=5)
        
        # Main control buttons
        button_frame = ttk.Frame(controls_group)
        button_frame.pack(fill='x')
        
        self.start_btn = ttk.Button(button_frame, text="🚀 START TRADING", 
                                   command=self.start_trading, width=20, 
                                   style="Accent.TButton")
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹️ STOP TRADING", 
                                  command=self.stop_trading, width=20, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        self.emergency_btn = ttk.Button(button_frame, text="🚨 EMERGENCY STOP", 
                                       command=self.emergency_stop, width=20)
        self.emergency_btn.pack(side='left', padx=5)
        
        # Strategy Quick Selection
        strategy_frame = ttk.LabelFrame(control_frame, text="🎯 Quick Strategy Selection", padding=10)
        strategy_frame.pack(fill='x', padx=5, pady=5)
        
        strategy_desc = ttk.Label(strategy_frame, 
                                 text="Choose your trading style and start immediately - all settings are pre-configured!")
        strategy_desc.pack(anchor='w', pady=(0, 10))
        
        # Strategy buttons
        strategies_grid = ttk.Frame(strategy_frame)
        strategies_grid.pack(fill='x')
        
        # Row 1: Basic Strategies
        basic_label = ttk.Label(strategies_grid, text="💎 Recommended Strategies:", font=('Arial', 10, 'bold'))
        basic_label.grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 5))
        
        self.scalping_btn = ttk.Button(strategies_grid, text="⚡ Scalping (M1)\nFast • High Risk • Quick Profits", 
                                      command=lambda: self.apply_strategy_and_start('scalping'), width=25)
        self.scalping_btn.grid(row=1, column=0, padx=5, pady=2)
        
        self.intraday_btn = ttk.Button(strategies_grid, text="📈 Intraday (M15)\nBalanced • Medium Risk • Steady", 
                                      command=lambda: self.apply_strategy_and_start('intraday'), width=25)
        self.intraday_btn.grid(row=1, column=1, padx=5, pady=2)
        
        self.swing_btn = ttk.Button(strategies_grid, text="🎯 Swing (H1)\nSlow • Low Risk • Patient", 
                                   command=lambda: self.apply_strategy_and_start('swing'), width=25)
        self.swing_btn.grid(row=1, column=2, padx=5, pady=2)
        
        # Row 2: Advanced Strategies
        advanced_label = ttk.Label(strategies_grid, text="🔥 Advanced Strategies:", font=('Arial', 10, 'bold'))
        advanced_label.grid(row=2, column=0, columnspan=3, sticky='w', pady=(15, 5))
        
        self.aggressive_btn = ttk.Button(strategies_grid, text="🚀 Aggressive Scalping\nVery Fast • Very High Risk", 
                                        command=lambda: self.apply_strategy_and_start('aggressive_scalping'), width=25)
        self.aggressive_btn.grid(row=3, column=0, padx=5, pady=2)
        
        self.conservative_btn = ttk.Button(strategies_grid, text="🛡️ Conservative Swing\nVery Safe • Very Low Risk", 
                                          command=lambda: self.apply_strategy_and_start('conservative_swing'), width=25)
        self.conservative_btn.grid(row=3, column=1, padx=5, pady=2)
        
        self.news_btn = ttk.Button(strategies_grid, text="📺 News Trading\nNews Events • Quick Moves", 
                                  command=lambda: self.apply_strategy_and_start('news_trading'), width=25)
        self.news_btn.grid(row=3, column=2, padx=5, pady=2)
        
        # Current Strategy Display
        current_frame = ttk.LabelFrame(control_frame, text="📊 Current Status", padding=10)
        current_frame.pack(fill='x', padx=5, pady=5)
        
        self.current_strategy_label = ttk.Label(current_frame, text="Strategy: Not Selected", 
                                               font=('Arial', 11, 'bold'))
        self.current_strategy_label.pack(anchor='w')
        
        self.current_settings_label = ttk.Label(current_frame, text="Click a strategy button to start trading")
        self.current_settings_label.pack(anchor='w')
        
        # Market Information
        market_frame = ttk.LabelFrame(control_frame, text="📈 Live Market Data", padding=10)
        market_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create market info display
        self.market_info = tk.Text(market_frame, height=6, state='disabled', 
                                  font=('Consolas', 10))
        market_scroll = ttk.Scrollbar(market_frame, orient='vertical', command=self.market_info.yview)
        self.market_info.configure(yscrollcommand=market_scroll.set)
        
        self.market_info.pack(side='left', fill='both', expand=True)
        market_scroll.pack(side='right', fill='y')
        
        # Force Enable Trading Button
        force_enable_frame = ttk.Frame(control_frame)
        force_enable_frame.pack(fill='x', padx=5, pady=5)
        
        self.force_enable_btn = ttk.Button(force_enable_frame, text="🔓 FORCE ENABLE TRADING", 
                                          command=self.force_enable_trading, width=25)
        self.force_enable_btn.pack(side='left', padx=5)
        
        force_enable_label = ttk.Label(force_enable_frame, 
                                      text="Use this if 'Trading is disabled' error persists",
                                      font=('Arial', 9, 'italic'))
        force_enable_label.pack(side='left', padx=10)
        
        # Advanced Settings Link
        advanced_note = ttk.Label(control_frame, 
                                 text="💡 Need custom settings? Go to 'Settings' tab for detailed configuration",
                                 font=('Arial', 9, 'italic'))
        advanced_note.pack(pady=5)
    
    def create_settings_tab(self):
        """Create comprehensive settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Create scrollable frame
        canvas = tk.Canvas(settings_frame)
        scrollbar = ttk.Scrollbar(settings_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Timeframe Settings
        tf_frame = ttk.LabelFrame(scrollable_frame, text="Timeframe Settings", padding=10)
        tf_frame.pack(fill='x', padx=5, pady=5)
        
        tf_options = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        
        row = ttk.Frame(tf_frame)
        row.pack(fill='x', pady=2)
        
        ttk.Label(row, text="Entry TF:").grid(row=0, column=0, sticky='w')
        self.entry_tf_var = tk.StringVar(value=self.config.get('timeframes', {}).get('entry_tf', 'M1'))
        self.entry_tf_combo = ttk.Combobox(row, textvariable=self.entry_tf_var, values=tf_options, width=10, state='readonly')
        self.entry_tf_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(row, text="Trend TF:").grid(row=0, column=2, sticky='w', padx=(20, 0))
        self.trend_tf_var = tk.StringVar(value=self.config.get('timeframes', {}).get('trend_tf', 'M15'))
        self.trend_tf_combo = ttk.Combobox(row, textvariable=self.trend_tf_var, values=tf_options, width=10, state='readonly')
        self.trend_tf_combo.grid(row=0, column=3, padx=5)
        
        ttk.Label(row, text="Bias TF:").grid(row=0, column=4, sticky='w', padx=(20, 0))
        self.bias_tf_var = tk.StringVar(value=self.config.get('timeframes', {}).get('bias_tf', 'H1'))
        self.bias_tf_combo = ttk.Combobox(row, textvariable=self.bias_tf_var, values=tf_options, width=10, state='readonly')
        self.bias_tf_combo.grid(row=0, column=5, padx=5)
        
        # Checkboxes
        self.tf_alignment_var = tk.BooleanVar(value=self.config.get('timeframes', {}).get('require_tf_alignment', True))
        ttk.Checkbutton(tf_frame, text="Require TF Alignment", variable=self.tf_alignment_var).pack(anchor='w', pady=5)
        
        # Risk Management
        risk_frame = ttk.LabelFrame(scrollable_frame, text="Risk Management", padding=10)
        risk_frame.pack(fill='x', padx=5, pady=5)
        
        risk_grid = ttk.Frame(risk_frame)
        risk_grid.pack(fill='x')
        
        # Max Positions
        ttk.Label(risk_grid, text="Max Positions:").grid(row=0, column=0, sticky='w', pady=2)
        self.max_pos_var = tk.IntVar(value=self.config.get('trading_settings', {}).get('max_positions', 5))
        ttk.Spinbox(risk_grid, from_=1, to=20, textvariable=self.max_pos_var, width=8).grid(row=0, column=1, padx=5, pady=2)
        
        # Daily Loss Limit
        ttk.Label(risk_grid, text="Daily Loss Limit ($):").grid(row=0, column=2, sticky='w', padx=(20, 0), pady=2)
        self.daily_loss_var = tk.DoubleVar(value=self.config.get('risk_management', {}).get('daily_loss_limit', 100))
        ttk.Spinbox(risk_grid, from_=10, to=1000, increment=10, textvariable=self.daily_loss_var, width=8).grid(row=0, column=3, padx=5, pady=2)
        
        # Anti-hedge
        self.anti_hedge_var = tk.BooleanVar(value=self.config.get('trading_settings', {}).get('anti_hedge', True))
        ttk.Checkbutton(risk_frame, text="Anti-Hedge (Prevent opposite positions)", variable=self.anti_hedge_var).pack(anchor='w', pady=5)
        
        # Advanced Settings
        adv_frame = ttk.LabelFrame(scrollable_frame, text="Advanced Settings", padding=10)
        adv_frame.pack(fill='x', padx=5, pady=5)
        
        adv_grid = ttk.Frame(adv_frame)
        adv_grid.pack(fill='x')
        
        # Fractal Period
        ttk.Label(adv_grid, text="Fractal Period:").grid(row=0, column=0, sticky='w', pady=2)
        self.fractal_period_var = tk.IntVar(value=self.config.get('signal_parameters', {}).get('fractal_period', 5))
        ttk.Spinbox(adv_grid, from_=3, to=10, textvariable=self.fractal_period_var, width=8).grid(row=0, column=1, padx=5, pady=2)
        
        # RSI Period
        ttk.Label(adv_grid, text="RSI Period:").grid(row=0, column=2, sticky='w', padx=(20, 0), pady=2)
        self.rsi_period_var = tk.IntVar(value=self.config.get('signal_parameters', {}).get('rsi_period', 14))
        ttk.Spinbox(adv_grid, from_=5, to=30, textvariable=self.rsi_period_var, width=8).grid(row=0, column=3, padx=5, pady=2)
        
        # Check Interval
        ttk.Label(adv_grid, text="Check Interval (sec):").grid(row=1, column=0, sticky='w', pady=2)
        self.check_interval_var = tk.IntVar(value=self.config.get('monitoring', {}).get('check_interval_seconds', 30))
        ttk.Spinbox(adv_grid, from_=5, to=300, increment=5, textvariable=self.check_interval_var, width=8).grid(row=1, column=1, padx=5, pady=2)
        
        # Recovery & Take Profit Settings
        recovery_frame = ttk.LabelFrame(scrollable_frame, text="Recovery & Take Profit Settings", padding=10)
        recovery_frame.pack(fill='x', padx=5, pady=5)
        
        # Recovery System
        recovery_section = ttk.LabelFrame(recovery_frame, text="Recovery System", padding=5)
        recovery_section.pack(fill='x', pady=(0, 10))
        
        recovery_grid = ttk.Frame(recovery_section)
        recovery_grid.pack(fill='x')
        
        # Enable Recovery
        self.enable_recovery_var = tk.BooleanVar(value=self.config.get('recovery_system', {}).get('enable_recovery', True))
        ttk.Checkbutton(recovery_grid, text="Enable Recovery System", variable=self.enable_recovery_var).grid(row=0, column=0, columnspan=2, sticky='w', pady=2)
        
        # Recovery Trigger
        ttk.Label(recovery_grid, text="Recovery Trigger (points):").grid(row=1, column=0, sticky='w', pady=2)
        self.recovery_trigger_var = tk.IntVar(value=self.config.get('recovery_system', {}).get('recovery_trigger_points', 100))
        ttk.Spinbox(recovery_grid, from_=50, to=500, increment=10, textvariable=self.recovery_trigger_var, width=8).grid(row=1, column=1, padx=5, pady=2)
        
        # Martingale Multiplier
        ttk.Label(recovery_grid, text="Martingale Multiplier:").grid(row=1, column=2, sticky='w', padx=(20, 0), pady=2)
        self.martingale_multiplier_var = tk.DoubleVar(value=self.config.get('recovery_system', {}).get('martingale_multiplier', 2.0))
        ttk.Spinbox(recovery_grid, from_=1.1, to=5.0, increment=0.1, textvariable=self.martingale_multiplier_var, width=8).grid(row=1, column=3, padx=5, pady=2)
        
        # Max Recovery Levels
        ttk.Label(recovery_grid, text="Max Recovery Levels:").grid(row=2, column=0, sticky='w', pady=2)
        self.max_recovery_var = tk.IntVar(value=self.config.get('recovery_system', {}).get('max_recovery_levels', 3))
        ttk.Spinbox(recovery_grid, from_=1, to=10, textvariable=self.max_recovery_var, width=8).grid(row=2, column=1, padx=5, pady=2)
        
        # Smart Recovery
        self.smart_recovery_var = tk.BooleanVar(value=self.config.get('recovery_system', {}).get('smart_recovery', True))
        ttk.Checkbutton(recovery_grid, text="Smart Recovery (wait for same signal)", variable=self.smart_recovery_var).grid(row=2, column=2, columnspan=2, sticky='w', padx=(20, 0), pady=2)
        
        # Take Profit System
        tp_section = ttk.LabelFrame(recovery_frame, text="Take Profit System", padding=5)
        tp_section.pack(fill='x', pady=(0, 10))
        
        tp_grid = ttk.Frame(tp_section)
        tp_grid.pack(fill='x')
        
        # Enable TP
        self.enable_tp_var = tk.BooleanVar(value=self.config.get('take_profit', {}).get('enable_tp', False))
        ttk.Checkbutton(tp_grid, text="Enable Take Profit", variable=self.enable_tp_var).grid(row=0, column=0, columnspan=2, sticky='w', pady=2)
        
        # TP Points
        ttk.Label(tp_grid, text="TP Points:").grid(row=1, column=0, sticky='w', pady=2)
        self.tp_points_var = tk.IntVar(value=self.config.get('take_profit', {}).get('tp_points', 200))
        ttk.Spinbox(tp_grid, from_=50, to=1000, increment=10, textvariable=self.tp_points_var, width=8).grid(row=1, column=1, padx=5, pady=2)
        
        # Dynamic TP
        self.dynamic_tp_var = tk.BooleanVar(value=self.config.get('take_profit', {}).get('dynamic_tp', False))
        ttk.Checkbutton(tp_grid, text="Dynamic TP (recalculate for recovery)", variable=self.dynamic_tp_var).grid(row=1, column=2, columnspan=2, sticky='w', padx=(20, 0), pady=2)
        
        # TP Multiplier
        ttk.Label(tp_grid, text="TP Multiplier:").grid(row=2, column=0, sticky='w', pady=2)
        self.tp_multiplier_var = tk.DoubleVar(value=self.config.get('take_profit', {}).get('tp_multiplier', 1.5))
        ttk.Spinbox(tp_grid, from_=1.0, to=5.0, increment=0.1, textvariable=self.tp_multiplier_var, width=8).grid(row=2, column=1, padx=5, pady=2)
        
        # Trailing Stop
        self.trailing_stop_var = tk.BooleanVar(value=self.config.get('take_profit', {}).get('trailing_stop', False))
        ttk.Checkbutton(tp_grid, text="Trailing Stop", variable=self.trailing_stop_var).grid(row=2, column=2, sticky='w', padx=(20, 0), pady=2)
        
        # Trailing Points
        ttk.Label(tp_grid, text="Trailing Points:").grid(row=2, column=3, sticky='w', padx=(10, 0), pady=2)
        self.trailing_points_var = tk.IntVar(value=self.config.get('take_profit', {}).get('trailing_points', 50))
        ttk.Spinbox(tp_grid, from_=10, to=200, increment=5, textvariable=self.trailing_points_var, width=8).grid(row=2, column=4, padx=5, pady=2)
        
        # Spread Management
        spread_frame = ttk.LabelFrame(scrollable_frame, text="Spread & Risk Management", padding=10)
        spread_frame.pack(fill='x', padx=5, pady=5)
        
        spread_grid = ttk.Frame(spread_frame)
        spread_grid.pack(fill='x')
        
        # Max Spread
        ttk.Label(spread_grid, text="Max Spread (points):").grid(row=0, column=0, sticky='w', pady=2)
        self.max_spread_var = tk.IntVar(value=self.config.get('spread_management', {}).get('max_spread_points', 50))
        ttk.Spinbox(spread_grid, from_=5, to=200, increment=5, textvariable=self.max_spread_var, width=8).grid(row=0, column=1, padx=5, pady=2)
        
        # Spread Filter
        self.spread_filter_var = tk.BooleanVar(value=self.config.get('spread_management', {}).get('spread_filter_enabled', True))
        ttk.Checkbutton(spread_grid, text="Enable Spread Filter", variable=self.spread_filter_var).grid(row=0, column=2, columnspan=2, sticky='w', padx=(20, 0), pady=2)
        
        # Max Drawdown
        ttk.Label(spread_grid, text="Max Drawdown (%):").grid(row=1, column=0, sticky='w', pady=2)
        self.max_drawdown_var = tk.IntVar(value=self.config.get('risk_management', {}).get('max_drawdown_percent', 15))
        ttk.Spinbox(spread_grid, from_=5, to=50, increment=1, textvariable=self.max_drawdown_var, width=8).grid(row=1, column=1, padx=5, pady=2)
        
        # Max Daily Trades
        ttk.Label(spread_grid, text="Max Daily Trades:").grid(row=1, column=2, sticky='w', padx=(20, 0), pady=2)
        self.max_daily_trades_var = tk.IntVar(value=self.config.get('risk_management', {}).get('max_daily_trades', 20))
        ttk.Spinbox(spread_grid, from_=1, to=100, increment=1, textvariable=self.max_daily_trades_var, width=8).grid(row=1, column=3, padx=5, pady=2)
        
        # Buttons
        btn_frame = ttk.Frame(adv_frame)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="Save All Settings", command=self.save_all_settings).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Load Preset", command=self.load_preset).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Reset to Default", command=self.reset_to_default).pack(side='left', padx=5)
        
        # Presets
        preset_frame = ttk.LabelFrame(scrollable_frame, text="Quick Presets", padding=10)
        preset_frame.pack(fill='x', padx=5, pady=5)
        
        # Basic presets
        basic_preset_buttons = ttk.Frame(preset_frame)
        basic_preset_buttons.pack(fill='x')
        
        ttk.Label(basic_preset_buttons, text="Basic Presets:").pack(anchor='w')
        basic_row = ttk.Frame(basic_preset_buttons)
        basic_row.pack(fill='x', pady=2)
        
        ttk.Button(basic_row, text="Scalping (M1)", command=lambda: self.apply_preset('scalping')).pack(side='left', padx=5)
        ttk.Button(basic_row, text="Intraday (M15)", command=lambda: self.apply_preset('intraday')).pack(side='left', padx=5)
        ttk.Button(basic_row, text="Swing (H1)", command=lambda: self.apply_preset('swing')).pack(side='left', padx=5)
        
        # Advanced presets
        advanced_preset_buttons = ttk.Frame(preset_frame)
        advanced_preset_buttons.pack(fill='x', pady=(10, 0))
        
        ttk.Label(advanced_preset_buttons, text="Advanced Presets:").pack(anchor='w')
        advanced_row1 = ttk.Frame(advanced_preset_buttons)
        advanced_row1.pack(fill='x', pady=2)
        
        ttk.Button(advanced_row1, text="Aggressive Scalping", command=lambda: self.apply_preset('aggressive_scalping')).pack(side='left', padx=5)
        ttk.Button(advanced_row1, text="Conservative Swing", command=lambda: self.apply_preset('conservative_swing')).pack(side='left', padx=5)
        ttk.Button(advanced_row1, text="News Trading", command=lambda: self.apply_preset('news_trading')).pack(side='left', padx=5)
        
        # Custom preset management
        custom_preset_buttons = ttk.Frame(preset_frame)
        custom_preset_buttons.pack(fill='x', pady=(10, 0))
        
        ttk.Label(custom_preset_buttons, text="Custom Presets:").pack(anchor='w')
        custom_row = ttk.Frame(custom_preset_buttons)
        custom_row.pack(fill='x', pady=2)
        
        ttk.Button(custom_row, text="Save Current as Preset", command=self.save_custom_preset).pack(side='left', padx=5)
        ttk.Button(custom_row, text="Manage Presets", command=self.manage_presets).pack(side='left', padx=5)
    
    def create_monitoring_tab(self):
        """Create monitoring and statistics tab"""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="Monitoring")
        
        # Connection Status
        conn_frame = ttk.LabelFrame(monitor_frame, text="Connection Status", padding=10)
        conn_frame.pack(fill='x', padx=5, pady=5)
        
        self.conn_status = tk.Text(conn_frame, height=4, state='disabled')
        self.conn_status.pack(fill='x')
        
        # Signal Status
        signal_frame = ttk.LabelFrame(monitor_frame, text="Signal Analysis", padding=10)
        signal_frame.pack(fill='x', padx=5, pady=5)
        
        self.signal_status = tk.Text(signal_frame, height=6, state='disabled')
        self.signal_status.pack(fill='x')
        
        # Performance Stats
        stats_frame = ttk.LabelFrame(monitor_frame, text="Performance Statistics", padding=10)
        stats_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Stats display using Treeview
        self.stats_tree = ttk.Treeview(stats_frame, columns=('Value',), show='tree headings', height=10)
        self.stats_tree.heading('#0', text='Metric')
        self.stats_tree.heading('Value', text='Value')
        self.stats_tree.column('#0', width=200)
        self.stats_tree.column('Value', width=150)
        
        stats_scroll = ttk.Scrollbar(stats_frame, orient='vertical', command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=stats_scroll.set)
        
        self.stats_tree.pack(side='left', fill='both', expand=True)
        stats_scroll.pack(side='right', fill='y')
    
    def create_positions_tab(self):
        """Create positions monitoring tab"""
        pos_frame = ttk.Frame(self.notebook)
        self.notebook.add(pos_frame, text="Positions")
        
        # Position controls
        ctrl_frame = ttk.Frame(pos_frame)
        ctrl_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(ctrl_frame, text="Refresh Positions", command=self.refresh_positions).pack(side='left', padx=5)
        ttk.Button(ctrl_frame, text="Close All Positions", command=self.close_all_positions).pack(side='left', padx=5)
        
        # Positions table
        pos_table_frame = ttk.LabelFrame(pos_frame, text="Open Positions", padding=5)
        pos_table_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create treeview for positions
        columns = ('Ticket', 'Type', 'Volume', 'Symbol', 'Open Price', 'Current Price', 'P&L', 'Open Time')
        self.pos_tree = ttk.Treeview(pos_table_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        for col in columns:
            self.pos_tree.heading(col, text=col)
            self.pos_tree.column(col, width=100, anchor='center')
        
        # Scrollbars
        pos_v_scroll = ttk.Scrollbar(pos_table_frame, orient='vertical', command=self.pos_tree.yview)
        pos_h_scroll = ttk.Scrollbar(pos_table_frame, orient='horizontal', command=self.pos_tree.xview)
        self.pos_tree.configure(yscrollcommand=pos_v_scroll.set, xscrollcommand=pos_h_scroll.set)
        
        self.pos_tree.grid(row=0, column=0, sticky='nsew')
        pos_v_scroll.grid(row=0, column=1, sticky='ns')
        pos_h_scroll.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        pos_table_frame.grid_rowconfigure(0, weight=1)
        pos_table_frame.grid_columnconfigure(0, weight=1)
        
        # Position details
        details_frame = ttk.LabelFrame(pos_frame, text="Position Details", padding=5)
        details_frame.pack(fill='x', padx=5, pady=5)
        
        self.pos_details = tk.Text(details_frame, height=4, state='disabled')
        self.pos_details.pack(fill='x')
        
        # Bind selection event
        self.pos_tree.bind('<<TreeviewSelect>>', self.on_position_select)
    
    def create_logs_tab(self):
        """Create logs and history tab"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="Logs")
        
        # Log controls
        log_ctrl_frame = ttk.Frame(logs_frame)
        log_ctrl_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(log_ctrl_frame, text="Clear Logs", command=self.clear_logs).pack(side='left', padx=5)
        ttk.Button(log_ctrl_frame, text="Export Logs", command=self.export_logs).pack(side='left', padx=5)
        
        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_ctrl_frame, text="Auto Scroll", variable=self.auto_scroll_var).pack(side='left', padx=20)
        
        # Logs display
        log_display_frame = ttk.LabelFrame(logs_frame, text="Trading Logs", padding=5)
        log_display_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.logs_text = tk.Text(log_display_frame, state='disabled', wrap='word')
        log_scroll = ttk.Scrollbar(log_display_frame, orient='vertical', command=self.logs_text.yview)
        self.logs_text.configure(yscrollcommand=log_scroll.set)
        
        self.logs_text.pack(side='left', fill='both', expand=True)
        log_scroll.pack(side='right', fill='y')
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side='bottom', fill='x')
        
        # Status indicators
        self.status_mt5 = ttk.Label(self.status_frame, text="MT5: ❌", foreground='red')
        self.status_mt5.pack(side='left', padx=5)
        
        self.status_login = ttk.Label(self.status_frame, text="Login: ❌", foreground='red')
        self.status_login.pack(side='left', padx=5)
        
        self.status_trading = ttk.Label(self.status_frame, text="Trading: ⏹️", foreground='orange')
        self.status_trading.pack(side='left', padx=5)
        
        self.status_time = ttk.Label(self.status_frame, text="Last Update: Never")
        self.status_time.pack(side='right', padx=5)
        
    def force_enable_trading(self):
        """Force enable trading in the engine"""
        try:
            # Update config
            self.config['trading_settings']['trading_enabled'] = True
            
            # Force enable TP as well
            if 'take_profit' not in self.config:
                self.config['take_profit'] = {}
            self.config['take_profit']['enable_tp'] = True
            self.config['take_profit']['tp_points'] = 200
            self.config['take_profit']['dynamic_tp'] = True
            
            # Save to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            
            # Force update trading engine
            if self.trading_engine:
                self.trading_engine.config['trading_settings']['trading_enabled'] = True
                self.trading_engine.config['take_profit'] = self.config['take_profit']
                self.log_message("🔓 FORCED trading + TP enabled in engine")
                messagebox.showinfo("Success", "Trading & Take Profit ENABLED!\n\nTP: 200 points\nDynamic TP: ON\nYou can now start trading.")
            else:
                self.log_message("❌ Trading engine not available")
                messagebox.showerror("Error", "Trading engine not initialized")
            
        except Exception as e:
            self.log_message(f"❌ Force enable error: {e}")
            messagebox.showerror("Error", f"Failed to force enable: {e}")
    
    def apply_strategy_and_start(self, strategy_name: str):
        """Apply strategy preset and start trading immediately"""
        try:
            # Apply the preset first
            self.apply_preset(strategy_name)
            
            # Update current strategy display
            strategy_display_names = {
                'scalping': '⚡ Scalping (M1) - Fast & Aggressive',
                'intraday': '📈 Intraday (M15) - Balanced & Steady', 
                'swing': '🎯 Swing (H1) - Patient & Conservative',
                'aggressive_scalping': '🚀 Aggressive Scalping - Very Fast & Risky',
                'conservative_swing': '🛡️ Conservative Swing - Very Safe & Slow',
                'news_trading': '📺 News Trading - Event-Based'
            }
            
            display_name = strategy_display_names.get(strategy_name, strategy_name.title())
            self.current_strategy_label.configure(text=f"Strategy: {display_name}")
            
            # Get strategy details
            presets = self.config.get('presets', {})
            if strategy_name in presets:
                preset = presets[strategy_name]
                settings_text = f"TF: {preset.get('entry_tf', 'M1')}/{preset.get('trend_tf', 'M15')}/{preset.get('bias_tf', 'H1')} | "
                settings_text += f"RSI: {preset.get('rsi_up', 55)}/{preset.get('rsi_down', 45)} | "
                settings_text += f"Lot: {preset.get('lot_size', 0.01)} | "
                settings_text += f"Recovery: {'ON' if preset.get('enable_recovery', True) else 'OFF'}"
                
                self.current_settings_label.configure(text=settings_text)
            
            # Auto-start trading
            self.root.after(1000, self.start_trading)  # Start after 1 second
            
            self.log_message(f"🎯 Applied strategy: {strategy_name} - Auto-starting...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply strategy: {e}")
            self.log_message(f"❌ Strategy error: {e}")
    
    def start_trading(self):
        """Start trading system"""
        if self.trading_engine:
            self.is_trading = True
            self.start_btn.configure(state='disabled')
            self.stop_btn.configure(state='normal')
            self.status_trading.configure(text="Trading: ▶️", foreground='green')
            
            # Disable strategy buttons during trading
            self.scalping_btn.configure(state='disabled')
            self.intraday_btn.configure(state='disabled') 
            self.swing_btn.configure(state='disabled')
            self.aggressive_btn.configure(state='disabled')
            self.conservative_btn.configure(state='disabled')
            self.news_btn.configure(state='disabled')
            
            self.log_message("🚀 Trading system started")
            
            # Start trading engine in separate thread
            threading.Thread(target=self.run_trading_loop, daemon=True).start()
        else:
            messagebox.showerror("Error", "Trading engine not initialized")
    
    def stop_trading(self):
        """Stop trading system"""
        self.is_trading = False
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        self.status_trading.configure(text="Trading: ⏸️", foreground='orange')
        
        # Re-enable strategy buttons
        self.scalping_btn.configure(state='normal')
        self.intraday_btn.configure(state='normal')
        self.swing_btn.configure(state='normal') 
        self.aggressive_btn.configure(state='normal')
        self.conservative_btn.configure(state='normal')
        self.news_btn.configure(state='normal')
        
        self.log_message("⏹️ Trading system stopped")
    
    def emergency_stop(self):
        """Emergency stop - close all positions"""
        result = messagebox.askyesno("Emergency Stop", 
                                   "This will stop trading and close ALL positions!\nAre you sure?")
        if result:
            self.stop_trading()
            self.close_all_positions()
            self.log_message("🚨 EMERGENCY STOP - All positions closed")
    
    def apply_quick_settings(self):
        """Apply quick settings changes"""
        try:
            # Update config
            self.config['trading_settings']['symbol'] = self.symbol_var.get()
            self.config['trading_settings']['lot_size'] = self.lot_var.get()
            self.config['signal_parameters']['rsi_up'] = self.rsi_up_var.get()
            self.config['signal_parameters']['rsi_down'] = self.rsi_down_var.get()
            
            # Force enable trading
            self.config['trading_settings']['trading_enabled'] = True
            
            # Apply to trading engine if available
            if self.trading_engine:
                self.trading_engine.update_settings(self.config)
                self.log_message("✅ Settings updated in trading engine")
            
            # Save to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            
            self.log_message("✅ Quick settings applied + Trading ENABLED")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {e}")
            self.log_message(f"❌ Settings error: {e}")
    
    def save_all_settings(self):
        """Save all settings to config file"""
        try:
            # Collect all settings from GUI
            self.config['trading_settings']['symbol'] = self.symbol_var.get()
            self.config['trading_settings']['lot_size'] = self.lot_var.get()
            self.config['trading_settings']['max_positions'] = self.max_pos_var.get()
            self.config['trading_settings']['anti_hedge'] = self.anti_hedge_var.get()
            
            self.config['signal_parameters']['rsi_up'] = self.rsi_up_var.get()
            self.config['signal_parameters']['rsi_down'] = self.rsi_down_var.get()
            self.config['signal_parameters']['rsi_period'] = self.rsi_period_var.get()
            self.config['signal_parameters']['fractal_period'] = self.fractal_period_var.get()
            
            self.config['timeframes']['entry_tf'] = self.entry_tf_var.get()
            self.config['timeframes']['trend_tf'] = self.trend_tf_var.get()
            self.config['timeframes']['bias_tf'] = self.bias_tf_var.get()
            self.config['timeframes']['require_tf_alignment'] = self.tf_alignment_var.get()
            
            if 'risk_management' not in self.config:
                self.config['risk_management'] = {}
            self.config['risk_management']['daily_loss_limit'] = self.daily_loss_var.get()
            self.config['risk_management']['max_drawdown_percent'] = self.max_drawdown_var.get()
            self.config['risk_management']['max_daily_trades'] = self.max_daily_trades_var.get()
            
            if 'monitoring' not in self.config:
                self.config['monitoring'] = {}
            self.config['monitoring']['check_interval_seconds'] = self.check_interval_var.get()
            
            # Recovery System Settings
            if 'recovery_system' not in self.config:
                self.config['recovery_system'] = {}
            self.config['recovery_system']['enable_recovery'] = self.enable_recovery_var.get()
            self.config['recovery_system']['recovery_trigger_points'] = self.recovery_trigger_var.get()
            self.config['recovery_system']['martingale_multiplier'] = self.martingale_multiplier_var.get()
            self.config['recovery_system']['max_recovery_levels'] = self.max_recovery_var.get()
            self.config['recovery_system']['smart_recovery'] = self.smart_recovery_var.get()
            
            # Take Profit Settings
            if 'take_profit' not in self.config:
                self.config['take_profit'] = {}
            self.config['take_profit']['enable_tp'] = self.enable_tp_var.get()
            self.config['take_profit']['tp_points'] = self.tp_points_var.get()
            self.config['take_profit']['dynamic_tp'] = self.dynamic_tp_var.get()
            self.config['take_profit']['tp_multiplier'] = self.tp_multiplier_var.get()
            self.config['take_profit']['trailing_stop'] = self.trailing_stop_var.get()
            self.config['take_profit']['trailing_points'] = self.trailing_points_var.get()
            
            # Spread Management
            if 'spread_management' not in self.config:
                self.config['spread_management'] = {}
            self.config['spread_management']['max_spread_points'] = self.max_spread_var.get()
            self.config['spread_management']['spread_filter_enabled'] = self.spread_filter_var.get()
            
            # Save to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            
            messagebox.showinfo("Success", "Settings saved successfully!")
            self.log_message("💾 All settings saved")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
            self.log_message(f"❌ Save error: {e}")
    
    def load_settings_to_gui(self):
        """Load settings from config to GUI"""
        try:
            # Initialize all variables first
            ts = self.config.get('trading_settings', {})
            sp = self.config.get('signal_parameters', {})
            tf = self.config.get('timeframes', {})
            rm = self.config.get('risk_management', {})
            mon = self.config.get('monitoring', {})
            
            # Variables that are always available
            self.symbol_var = tk.StringVar(value=ts.get('symbol', 'XAUUSD'))
            self.lot_var = tk.DoubleVar(value=ts.get('lot_size', 0.01))
            self.rsi_up_var = tk.IntVar(value=sp.get('rsi_up', 55))
            self.rsi_down_var = tk.IntVar(value=sp.get('rsi_down', 45))
            
            # Variables for settings tab (will be created when tab is created)
            self.max_pos_var = tk.IntVar(value=ts.get('max_positions', 5))
            self.anti_hedge_var = tk.BooleanVar(value=ts.get('anti_hedge', True))
            self.rsi_period_var = tk.IntVar(value=sp.get('rsi_period', 14))
            self.fractal_period_var = tk.IntVar(value=sp.get('fractal_period', 5))
            
            self.entry_tf_var = tk.StringVar(value=tf.get('entry_tf', 'M1'))
            self.trend_tf_var = tk.StringVar(value=tf.get('trend_tf', 'M15'))
            self.bias_tf_var = tk.StringVar(value=tf.get('bias_tf', 'H1'))
            self.tf_alignment_var = tk.BooleanVar(value=tf.get('require_tf_alignment', True))
            
            self.daily_loss_var = tk.DoubleVar(value=rm.get('daily_loss_limit', 100))
            self.check_interval_var = tk.IntVar(value=mon.get('check_interval_seconds', 30))
            
        except Exception as e:
            self.log_message(f"⚠️ Settings load warning: {e}")
    
    def apply_preset(self, preset_name: str):
        """Apply preset configuration"""
        presets = self.config.get('presets', {})
        if preset_name not in presets:
            messagebox.showerror("Error", f"Preset '{preset_name}' not found")
            return
            
        preset = presets[preset_name]
        
        try:
            # Apply preset values to GUI variables
            entry_tf = preset.get('entry_tf', 'M1')
            trend_tf = preset.get('trend_tf', 'M15')
            bias_tf = preset.get('bias_tf', 'H1')
            rsi_up = preset.get('rsi_up', 55)
            rsi_down = preset.get('rsi_down', 45)
            check_interval = preset.get('check_interval_seconds', 30)
            
            # Update GUI variables
            self.entry_tf_var.set(entry_tf)
            self.trend_tf_var.set(trend_tf)
            self.bias_tf_var.set(bias_tf)
            self.rsi_up_var.set(rsi_up)
            self.rsi_down_var.set(rsi_down)
            self.check_interval_var.set(check_interval)
            
            # Apply additional settings if available
            if 'lot_size' in preset:
                self.lot_var.set(preset['lot_size'])
                self.config['trading_settings']['lot_size'] = preset['lot_size']
            
            if 'max_positions' in preset:
                self.max_pos_var.set(preset['max_positions'])
                self.config['trading_settings']['max_positions'] = preset['max_positions']
            
            if 'daily_loss_limit' in preset:
                self.daily_loss_var.set(preset['daily_loss_limit'])
                self.config['risk_management']['daily_loss_limit'] = preset['daily_loss_limit']
            
            if 'require_tf_alignment' in preset:
                self.tf_alignment_var.set(preset['require_tf_alignment'])
                self.config['timeframes']['require_tf_alignment'] = preset['require_tf_alignment']
            
            # Recovery System settings
            if 'recovery_trigger_points' in preset:
                self.recovery_trigger_var.set(preset['recovery_trigger_points'])
            if 'martingale_multiplier' in preset:
                self.martingale_multiplier_var.set(preset['martingale_multiplier'])
            if 'max_recovery_levels' in preset:
                self.max_recovery_var.set(preset['max_recovery_levels'])
            if 'smart_recovery' in preset:
                self.smart_recovery_var.set(preset['smart_recovery'])
            
            # Take Profit settings
            if 'enable_tp' in preset:
                self.enable_tp_var.set(preset['enable_tp'])
            if 'tp_points' in preset:
                self.tp_points_var.set(preset['tp_points'])
            if 'dynamic_tp' in preset:
                self.dynamic_tp_var.set(preset['dynamic_tp'])
            
            # Force update combobox display values
            self.entry_tf_combo.set(entry_tf)
            self.trend_tf_combo.set(trend_tf) 
            self.bias_tf_combo.set(bias_tf)
            
            # Update config
            self.config['timeframes']['entry_tf'] = entry_tf
            self.config['timeframes']['trend_tf'] = trend_tf
            self.config['timeframes']['bias_tf'] = bias_tf
            self.config['signal_parameters']['rsi_up'] = rsi_up
            self.config['signal_parameters']['rsi_down'] = rsi_down
            self.config['monitoring']['check_interval_seconds'] = check_interval
            
            # Force refresh the GUI
            self.root.update_idletasks()
            
            self.log_message(f"Applied preset: {preset_name}")
            
            # Show comprehensive preset info
            preset_info = f"Preset '{preset_name}' applied successfully!\n\n"
            preset_info += f"Entry TF: {entry_tf}\n"
            preset_info += f"Trend TF: {trend_tf}\n"
            preset_info += f"Bias TF: {bias_tf}\n"
            preset_info += f"RSI UP/DOWN: {rsi_up}/{rsi_down}\n"
            preset_info += f"Check Interval: {check_interval}s\n"
            
            if 'lot_size' in preset:
                preset_info += f"Lot Size: {preset['lot_size']}\n"
            if 'max_positions' in preset:
                preset_info += f"Max Positions: {preset['max_positions']}\n"
            if 'daily_loss_limit' in preset:
                preset_info += f"Daily Loss Limit: ${preset['daily_loss_limit']}\n"
            if 'require_tf_alignment' in preset:
                preset_info += f"TF Alignment: {preset['require_tf_alignment']}\n"
            
            messagebox.showinfo("Success", preset_info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply preset: {e}")
            self.log_message(f"Preset error: {e}")
    
    def save_custom_preset(self):
        """Save current settings as custom preset"""
        preset_name = tk.simpledialog.askstring("Save Preset", "Enter preset name:")
        if not preset_name:
            return
        
        try:
            # Collect current settings including recovery & TP
            custom_preset = {
                "entry_tf": self.entry_tf_var.get(),
                "trend_tf": self.trend_tf_var.get(),
                "bias_tf": self.bias_tf_var.get(),
                "rsi_up": self.rsi_up_var.get(),
                "rsi_down": self.rsi_down_var.get(),
                "check_interval_seconds": self.check_interval_var.get(),
                "lot_size": self.lot_var.get(),
                "max_positions": self.max_pos_var.get(),
                "daily_loss_limit": self.daily_loss_var.get(),
                "require_tf_alignment": self.tf_alignment_var.get(),
                "rsi_period": self.rsi_period_var.get(),
                "fractal_period": self.fractal_period_var.get(),
                "anti_hedge": self.anti_hedge_var.get(),
                
                # Recovery System
                "enable_recovery": self.enable_recovery_var.get(),
                "recovery_trigger_points": self.recovery_trigger_var.get(),
                "martingale_multiplier": self.martingale_multiplier_var.get(),
                "max_recovery_levels": self.max_recovery_var.get(),
                "smart_recovery": self.smart_recovery_var.get(),
                
                # Take Profit
                "enable_tp": self.enable_tp_var.get(),
                "tp_points": self.tp_points_var.get(),
                "dynamic_tp": self.dynamic_tp_var.get(),
                "tp_multiplier": self.tp_multiplier_var.get(),
                "trailing_stop": self.trailing_stop_var.get(),
                "trailing_points": self.trailing_points_var.get(),
                
                # Spread & Risk
                "max_spread_points": self.max_spread_var.get(),
                "spread_filter_enabled": self.spread_filter_var.get(),
                "max_drawdown_percent": self.max_drawdown_var.get(),
                "max_daily_trades": self.max_daily_trades_var.get()
            }
            
            # Add to config
            if 'presets' not in self.config:
                self.config['presets'] = {}
            
            self.config['presets'][preset_name] = custom_preset
            
            # Save to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            
            # Show detailed preset info
            preset_info = f"Preset '{preset_name}' saved successfully!\n\n"
            preset_info += f"Timeframes: {custom_preset['entry_tf']}/{custom_preset['trend_tf']}/{custom_preset['bias_tf']}\n"
            preset_info += f"RSI: {custom_preset['rsi_up']}/{custom_preset['rsi_down']}\n"
            preset_info += f"Lot Size: {custom_preset['lot_size']}\n"
            preset_info += f"Recovery: {'ON' if custom_preset['enable_recovery'] else 'OFF'}\n"
            preset_info += f"Take Profit: {'ON' if custom_preset['enable_tp'] else 'OFF'}\n"
            
            messagebox.showinfo("Success", preset_info)
            self.log_message(f"Custom preset saved: {preset_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset: {e}")
    
    def manage_presets(self):
        """Manage custom presets"""
        preset_window = tk.Toplevel(self.root)
        preset_window.title("Manage Presets")
        preset_window.geometry("600x400")
        preset_window.transient(self.root)
        preset_window.grab_set()
        
        # Preset list
        list_frame = ttk.Frame(preset_window)
        list_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(list_frame, text="Available Presets:").pack(anchor='w')
        
        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill='both', expand=True, pady=(5, 10))
        
        preset_listbox = tk.Listbox(list_container)
        scrollbar = ttk.Scrollbar(list_container, orient='vertical', command=preset_listbox.yview)
        preset_listbox.configure(yscrollcommand=scrollbar.set)
        
        preset_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Load presets
        presets = self.config.get('presets', {})
        for preset_name in presets.keys():
            preset_listbox.insert('end', preset_name)
        
        # Buttons
        button_frame = ttk.Frame(preset_window)
        button_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        def apply_selected():
            selection = preset_listbox.curselection()
            if selection:
                preset_name = preset_listbox.get(selection[0])
                self.apply_preset(preset_name)
                preset_window.destroy()
        
        def delete_selected():
            selection = preset_listbox.curselection()
            if selection:
                preset_name = preset_listbox.get(selection[0])
                
                # Protect default presets
                if preset_name in ['scalping', 'intraday', 'swing']:
                    messagebox.showwarning("Warning", "Cannot delete default presets")
                    return
                
                result = messagebox.askyesno("Confirm Delete", f"Delete preset '{preset_name}'?")
                if result:
                    del self.config['presets'][preset_name]
                    
                    # Save config
                    with open(self.config_file, 'w', encoding='utf-8') as f:
                        json.dump(self.config, f, indent=2)
                    
                    # Refresh list
                    preset_listbox.delete(selection[0])
                    self.log_message(f"Deleted preset: {preset_name}")
        
        def export_preset():
            selection = preset_listbox.curselection()
            if selection:
                preset_name = preset_listbox.get(selection[0])
                preset_data = presets[preset_name]
                
                filename = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    title=f"Export {preset_name} Preset"
                )
                
                if filename:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump({preset_name: preset_data}, f, indent=2)
                    
                    messagebox.showinfo("Success", f"Preset exported to {filename}")
        
        def import_preset():
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Import Preset"
            )
            
            if filename:
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        imported_data = json.load(f)
                    
                    # Add imported presets
                    for name, data in imported_data.items():
                        self.config['presets'][name] = data
                        preset_listbox.insert('end', name)
                    
                    # Save config
                    with open(self.config_file, 'w', encoding='utf-8') as f:
                        json.dump(self.config, f, indent=2)
                    
                    messagebox.showinfo("Success", f"Imported {len(imported_data)} presets")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Import failed: {e}")
        
        ttk.Button(button_frame, text="Apply", command=apply_selected).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Delete", command=delete_selected).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Export", command=export_preset).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Import", command=import_preset).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close", command=preset_window.destroy).pack(side='right', padx=5)
    
    def reset_to_default(self):
        """Reset all settings to default"""
        result = messagebox.askyesno("Reset Settings", 
                                   "This will reset all settings to default values.\nAre you sure?")
        if result:
            self.config = self.get_default_config()
            self.load_settings_to_gui()
            self.log_message("🔄 Settings reset to default")
    
    def load_preset(self):
        """Load preset from file"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Load Preset"
            )
            
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    preset_config = json.load(f)
                
                self.config.update(preset_config)
                self.load_settings_to_gui()
                messagebox.showinfo("Success", f"Preset loaded from {filename}")
                self.log_message(f"📥 Preset loaded: {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {e}")
    
    # Utility Methods
    def log_message(self, message: str):
        """Add message to logs"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Update logs text widget
        self.logs_text.configure(state='normal')
        self.logs_text.insert('end', log_entry)
        
        # Auto-scroll if enabled
        if self.auto_scroll_var.get():
            self.logs_text.see('end')
            
        self.logs_text.configure(state='disabled')
        
        # Keep only last 1000 lines
        lines = self.logs_text.get('1.0', 'end').count('\n')
        if lines > 1000:
            self.logs_text.configure(state='normal')
            self.logs_text.delete('1.0', '100.0')
            self.logs_text.configure(state='disabled')
    
    def update_status_display(self, status_data: Dict):
        """Update all status displays"""
        # Update status bar
        mt5_status = "✅" if status_data.get('mt5_connected') else "❌"
        login_status = "✅" if status_data.get('logged_in') else "❌"
        
        self.status_mt5.configure(text=f"MT5: {mt5_status}", 
                                 foreground='green' if status_data.get('mt5_connected') else 'red')
        self.status_login.configure(text=f"Login: {login_status}",
                                   foreground='green' if status_data.get('logged_in') else 'red')
        
        # Update connection status
        conn_text = f"MT5 Connected: {status_data.get('mt5_connected', False)}\n"
        conn_text += f"Account Login: {status_data.get('logged_in', False)}\n"
        conn_text += f"Account: {status_data.get('account_number', 'N/A')}\n"
        conn_text += f"Balance: ${status_data.get('balance', 0):.2f}"
        
        self.update_text_widget(self.conn_status, conn_text)
        
        # Update market info
        market_text = f"Symbol: {status_data.get('symbol', 'N/A')}\n"
        market_text += f"Bid: {status_data.get('bid', 0):.5f}\n"
        market_text += f"Ask: {status_data.get('ask', 0):.5f}\n"
        market_text += f"Spread: {status_data.get('spread', 0):.1f} points\n"
        market_text += f"RSI: {status_data.get('rsi', 0):.2f}\n"
        market_text += f"Fractal: {status_data.get('fractal_status', 'NONE')}\n"
        market_text += f"Signal: {status_data.get('signal_status', 'NO SIGNAL')}"
        
        self.update_text_widget(self.market_info, market_text)
        
        # Update last update time
        self.status_time.configure(text=f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    def update_text_widget(self, widget: tk.Text, text: str):
        """Update text widget content"""
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)
        widget.configure(state='disabled')
    
    def refresh_positions(self):
        """Refresh positions table"""
        # Clear existing items
        for item in self.pos_tree.get_children():
            self.pos_tree.delete(item)
        
        # Get positions from MT5
        try:
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    pos_type = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                    open_time = datetime.fromtimestamp(pos.time).strftime('%H:%M:%S')
                    
                    self.pos_tree.insert('', 'end', values=(
                        pos.ticket,
                        pos_type,
                        pos.volume,
                        pos.symbol,
                        f"{pos.price_open:.5f}",
                        f"{pos.price_current:.5f}",
                        f"${pos.profit:.2f}",
                        open_time
                    ))
                    
        except Exception as e:
            self.log_message(f"❌ Position refresh error: {e}")
    
    def close_all_positions(self):
        """Close all open positions"""
        if self.trading_engine:
            result = messagebox.askyesno("Close All", "Close all open positions?")
            if result:
                self.trading_engine.emergency_close_all()
                self.log_message("🚨 All positions closed")
                self.refresh_positions()
    
    def on_position_select(self, event):
        """Handle position selection"""
        selection = self.pos_tree.selection()
        if selection:
            item = self.pos_tree.item(selection[0])
            values = item['values']
            
            details = f"Ticket: {values[0]}\n"
            details += f"Type: {values[1]} {values[2]} lots\n"
            details += f"Entry: {values[4]} → Current: {values[5]}\n"
            details += f"P&L: {values[6]} (Opened: {values[7]})"
            
            self.update_text_widget(self.pos_details, details)
    
    def clear_logs(self):
        """Clear all logs"""
        self.logs_text.configure(state='normal')
        self.logs_text.delete('1.0', 'end')
        self.logs_text.configure(state='disabled')
        self.log_message("📝 Logs cleared")
    
    def export_logs(self):
        """Export logs to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Logs"
            )
            
            if filename:
                logs_content = self.logs_text.get('1.0', 'end')
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(logs_content)
                
                messagebox.showinfo("Success", f"Logs exported to {filename}")
                self.log_message(f"📤 Logs exported: {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def run_trading_loop(self):
        """Main trading loop (runs in separate thread)"""
        while self.is_trading:
            try:
                if self.trading_engine:
                    # Run one trading cycle
                    self.trading_engine.analyze_and_trade()
                    
                # Sleep for check interval
                time.sleep(self.check_interval_var.get())
                
            except Exception as e:
                self.log_message(f"❌ Trading loop error: {e}")
                time.sleep(5)  # Short pause on error
    
    def start_auto_update(self):
        """Start automatic status updates"""
        self.update_running = True
        self.update_thread = threading.Thread(target=self.auto_update_loop, daemon=True)
        self.update_thread.start()
    
    def auto_update_loop(self):
        """Auto-update loop for GUI"""
        while self.update_running:
            try:
                # Update positions table
                self.root.after(0, self.refresh_positions)
                
                # Update status if managers available
                if self.login_manager and self.symbol_manager:
                    status_data = self.get_current_status()
                    self.root.after(0, lambda: self.update_status_display(status_data))
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"Auto-update error: {e}")
                time.sleep(5)
    
    def get_current_status(self) -> Dict:
        """Get current system status"""
        status = {
            'mt5_connected': False,
            'logged_in': False,
            'account_number': 'N/A',
            'balance': 0,
            'symbol': 'N/A',
            'bid': 0,
            'ask': 0,
            'spread': 0,
            'rsi': 0,
            'fractal_status': 'NONE',
            'signal_status': 'NO SIGNAL'
        }
        
        try:
            if self.login_manager:
                login_status = self.login_manager.check_connection_status()
                status['mt5_connected'] = login_status['mt5_initialized']
                status['logged_in'] = login_status['logged_in']
                
                if login_status['account_info']:
                    acc_info = login_status['account_info']
                    status['account_number'] = acc_info['login']
                    status['balance'] = acc_info['balance']
            
            if self.symbol_manager and self.symbol_manager.active_symbol:
                symbol = self.symbol_manager.active_symbol.name
                status['symbol'] = symbol
                
                # Get current prices
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    status['bid'] = tick.bid
                    status['ask'] = tick.ask
                    status['spread'] = (tick.ask - tick.bid) / self.symbol_manager.active_symbol.point
            
        except Exception as e:
            print(f"Status update error: {e}")
        
        return status
    
    def set_managers(self, login_manager=None, symbol_manager=None, trading_engine=None):
        """Set external manager references"""
        self.login_manager = login_manager
        self.symbol_manager = symbol_manager
        self.trading_engine = trading_engine
    
    def run(self):
        """Start the GUI"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
    
    def on_closing(self):
        """Handle application closing"""
        self.update_running = False
        self.is_trading = False
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        
        self.root.quit()
        self.root.destroy()

# Example usage
if __name__ == "__main__":
    app = TradingGUI()
    app.run()