import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time
import warnings
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles data loading and preprocessing"""
    
    @staticmethod
    def get_trading_days_from_yfinance(start_date, end_date):
        """Fetch trading days from yfinance using Nifty 50 data"""
        try:
            nifty_data = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
            trading_days = nifty_data.index.date
            print(f"Fetched {len(trading_days)} trading days from yfinance")
            return trading_days
        except Exception as e:
            print(f"Error fetching trading days from yfinance: {e}")
            print("Falling back to weekend filtering only")
            return None
    
    @staticmethod
    def load_and_preprocess(file_path, use_yfinance_calendar=True):
        """Load and preprocess the data using yfinance trading calendar"""
        df = pd.read_parquet(file_path)
        df.index = pd.to_datetime(df.index)
        
        # Filter trading hours (9:15 to 15:30)
        market_open = time(9, 15)
        market_close = time(15, 30)
        df_filtered = df.between_time(market_open, market_close)
        
        # Get trading days from yfinance
        if use_yfinance_calendar:
            start_date = df.index.min().date()
            end_date = df.index.max().date()
            
            trading_days = DataPreprocessor.get_trading_days_from_yfinance(start_date, end_date)
            
            if trading_days is not None:
                trading_days_set = set(trading_days)
                date_mask = [date in trading_days_set for date in df_filtered.index.date]
                df_filtered = df_filtered[date_mask]
                print(f"Filtered data to {len(trading_days)} trading days")
            else:
                df_filtered = df_filtered[df_filtered.index.weekday < 5]
                print("Using weekend filtering as fallback")
        else:
            df_filtered = df_filtered[df_filtered.index.weekday < 5]
        
        # Interpolate missing values within trading hours/days
        df_filtered = df_filtered.interpolate(method='linear', limit_direction='both')
        df_filtered = df_filtered.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Final dataset shape: {df_filtered.shape}")
        return df_filtered

class ZScoreStrategy:
    """Base model using z-score trading system"""
    
    def __init__(self, lookback_window=120, entry_threshold=2.0, exit_threshold=0.5):
        self.lookback_window = lookback_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def backtest(self, df):
        """Run backtest and return results dataset"""
        results = df.copy()
        
        # Calculate spread
        results['spread'] = results['banknifty'] - results['nifty']
        
        # Calculate rolling z-score
        results['spread_mean'] = results['spread'].rolling(self.lookback_window).mean()
        results['spread_std'] = results['spread'].rolling(self.lookback_window).std()
        results['z_score'] = (results['spread'] - results['spread_mean']) / results['spread_std']
        
        # Initialize columns
        results['position'] = 0
        results['trade_pnl'] = 0.0
        results['cumulative_pnl'] = 0.0
        
        # Track trade state
        position = 0
        entry_spread = 0
        entry_tte = 0
        cumulative_pnl = 0
        
        # Loop until len-1 to avoid index out of bounds for i+1
        for i in range(self.lookback_window, len(results) - 1):
            z = results['z_score'].iloc[i]
            current_spread = results['spread'].iloc[i]
            current_tte = results['tte'].iloc[i]
            
            # Next period's spread for realistic execution (avoid look-ahead bias)
            next_spread = results['spread'].iloc[i + 1]
            next_tte = results['tte'].iloc[i + 1]
            
            # Entry signals (detect at i, execute at i+1)
            if position == 0:
                if z > self.entry_threshold:
                    position = -1  # Short spread (expecting reversion)
                    entry_spread = next_spread  # Use next period's spread
                    entry_tte = current_tte     # Use current TTE for P/L calculation
                elif z < -self.entry_threshold:
                    position = 1   # Long spread (expecting reversion)
                    entry_spread = next_spread  # Use next period's spread
                    entry_tte = current_tte     # Use current TTE for P/L calculation
            
            # Exit signals (detect at i, execute at i+1)
            elif abs(z) < self.exit_threshold:
                # Calculate P/L using the formula: P/L = Spread_change × (TTE)^0.7
                exit_spread = next_spread  # Use next period's spread for exit
                spread_change = exit_spread - entry_spread
                trade_pnl = position * spread_change * (entry_tte ** 0.7)
                
                results['trade_pnl'].iloc[i + 1] = trade_pnl  # Record P/L at execution time
                cumulative_pnl += trade_pnl
                
                position = 0
                entry_spread = 0
                entry_tte = 0
            
            results['position'].iloc[i] = position
            results['cumulative_pnl'].iloc[i] = cumulative_pnl
        
        # Handle final position if still open
        if position != 0:
            results['position'].iloc[-1] = position
            results['cumulative_pnl'].iloc[-1] = cumulative_pnl
        
        return results

class PerformanceAnalytics:
    """UNIFIED analytics class for all strategies - NO DUPLICATION"""
    
    def __init__(self, results_df, strategy_name="Strategy"):
        self.results = results_df
        self.strategy_name = strategy_name
        self.metrics = {}
        
    def calculate_metrics(self):
        """Calculate only required performance metrics"""
        trade_pnls = self.results['trade_pnl'][self.results['trade_pnl'] != 0]
        cum_pnl = self.results['cumulative_pnl']
        
        # Basic metrics
        self.metrics['total_profit'] = cum_pnl.iloc[-1]
        self.metrics['total_trades'] = len(trade_pnls)
        
        # Win rate calculation
        if len(trade_pnls) > 0:
            winning_trades = trade_pnls[trade_pnls > 0]
            self.metrics['win_rate'] = len(winning_trades) / len(trade_pnls) * 100
            self.metrics['avg_win'] = winning_trades.mean() if len(winning_trades) > 0 else 0
            
            losing_trades = trade_pnls[trade_pnls < 0]
            self.metrics['avg_loss'] = losing_trades.mean() if len(losing_trades) > 0 else 0
        else:
            self.metrics['win_rate'] = 0
            self.metrics['avg_win'] = 0
            self.metrics['avg_loss'] = 0
        
        # Drawdown analysis
        self._calculate_drawdown_metrics()
        
        # Holding duration analysis (works for all strategies)
        self._calculate_holding_duration()
        
        return self.metrics
    
    def _calculate_drawdown_metrics(self):
        """Calculate drawdown metrics"""
        cum_pnl = self.results['cumulative_pnl']
        peak = cum_pnl.expanding().max()
        drawdown = cum_pnl - peak
        
        self.results['drawdown'] = drawdown
        self.metrics['max_drawdown'] = drawdown.min()
    
    def _calculate_holding_duration(self):
        """Calculate holding duration statistics - works for all strategies"""
        # Try hold_minutes column first (for OU/Kalman), fallback to position analysis
        if 'hold_minutes' in self.results.columns:
            hold_times = self.results['hold_minutes'][self.results['hold_minutes'] > 0]
            if len(hold_times) > 0:
                self.metrics['avg_holding_duration_minutes'] = hold_times.mean()
                self.metrics['median_holding_duration_minutes'] = hold_times.median()
                self.metrics['max_holding_duration_minutes'] = hold_times.max()
                return
        
        # Fallback: calculate from position changes (for ZScore)
        positions = self.results['position']
        position_changes = positions.diff().fillna(0)
        entries = position_changes[position_changes != 0].index
        
        durations = []
        i = 0
        while i < len(entries):
            entry_time = entries[i]
            entry_position = positions[entry_time]
            
            if entry_position != 0:
                future_positions = positions[positions.index > entry_time]
                exit_idx = future_positions[future_positions == 0].index
                if len(exit_idx) > 0:
                    exit_time = exit_idx[0]
                    duration = (exit_time - entry_time).total_seconds() / 60  # minutes
                    durations.append(duration)
            i += 1
        
        if durations:
            self.metrics['avg_holding_duration_minutes'] = np.mean(durations)
            self.metrics['median_holding_duration_minutes'] = np.median(durations)
            self.metrics['max_holding_duration_minutes'] = np.max(durations)
        else:
            self.metrics['avg_holding_duration_minutes'] = 0
            self.metrics['median_holding_duration_minutes'] = 0
            self.metrics['max_holding_duration_minutes'] = 0
    
    def plot_results(self, save_html=True, filename=None):
        """Generate simple P&L and Drawdown plots only"""
        if filename is None:
            filename = f'{self.strategy_name.lower()}_pnl_drawdown.html'
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative P&L', 'Drawdown'),
            vertical_spacing=0.12
        )
        
        # 1. Cumulative P&L 
        fig.add_trace(
            go.Scatter(
                x=self.results.index,
                y=self.results['cumulative_pnl'],
                name='Cumulative P&L',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Time</b>: %{x}<br><b>P&L</b>: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Drawdown
        fig.add_trace(
            go.Scatter(
                x=self.results.index,
                y=self.results['drawdown'],
                fill='tonexty',
                name='Drawdown',
                line=dict(color='red', width=1),
                fillcolor='rgba(255, 0, 0, 0.3)',
                hovertemplate='<b>Time</b>: %{x}<br><b>Drawdown</b>: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
        
        fig.update_layout(
            title=f'{self.strategy_name} Strategy - P&L and Drawdown',
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="P&L", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=1)
        
        if save_html:
            fig.write_html(filename)
            print(f"Plot saved as '{filename}'")
        
        fig.show()
        return fig
    
    def print_summary(self):
        """Print simplified performance summary"""
        print(f"=== {self.strategy_name.upper()} STRATEGY PERFORMANCE ===")
        print(f"Total Profit: {self.metrics['total_profit']:.4f}")
        print(f"Win Rate: {self.metrics['win_rate']:.2f}%")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.4f}")
        print(f"Average Holding Duration (minutes): {self.metrics['avg_holding_duration_minutes']:.2f}")
        print(f"Max Holding Duration (minutes): {self.metrics['max_holding_duration_minutes']:.2f}")
        print(f"Average Win: {self.metrics['avg_win']:.4f}")
        print(f"Average Loss: {self.metrics['avg_loss']:.4f}")

if __name__ == "__main__":
    # Load and preprocess data using yfinance trading calendar
    data = DataPreprocessor.load_and_preprocess('data.parquet')
    
    # Run z-score strategy
    strategy = ZScoreStrategy(lookback_window=120, entry_threshold=2.0, exit_threshold=0.5)
    results = strategy.backtest(data)
    
    # Save results to CSV
    results.to_csv('zscore_strategy_results.csv', index=True)
    print("Results saved to zscore_strategy_results.csv")
    
    # Analyze performance with unified analytics
    analytics = PerformanceAnalytics(results, "Z-Score")
    metrics = analytics.calculate_metrics()
    
    # Display results
    analytics.print_summary()
    
    # Generate simple plots
    analytics.plot_results(save_html=True, filename='zscore_pnl_drawdown.html')