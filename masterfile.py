import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Import existing classes - NO DUPLICATION
from base_strat_bt import DataPreprocessor, PerformanceAnalytics
from kalman_strat_bt import KalmanFilterStrategy
from ou_strat_bt import OUProcessStrategy

class DataSplitter:
    """Split data around major gaps for clean backtesting"""
    
    @staticmethod
    def analyze_large_gaps(df, min_gap_days=7):
        """Find large gaps in the data that should be split points"""
        time_diffs = df.index.to_series().diff().dropna()
        gap_days = time_diffs.dt.total_seconds() / 86400
        
        large_gaps = gap_days[gap_days > min_gap_days]
        
        print("=== LARGE GAP ANALYSIS ===")
        print(f"Total data points: {len(df)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Total duration: {(df.index.max() - df.index.min()).days} days")
        
        if len(large_gaps) == 0:
            print(f"No gaps larger than {min_gap_days} days found.")
            return []
        
        gaps = []
        for gap_idx, gap_duration in large_gaps.items():
            gap_start = df.index[df.index.get_loc(gap_idx) - 1]
            gap_end = gap_idx
            
            gaps.append({
                'gap_start': gap_start,
                'gap_end': gap_end,
                'duration_days': gap_duration,
                'split_before': gap_idx  # Split the data before this index
            })
            
            print(f"Large gap found: {gap_start} to {gap_end} ({gap_duration:.1f} days)")
        
        return gaps
    
    @staticmethod
    def split_data_around_gaps(df, gaps=None, min_gap_days=7):
        """Split data into clean periods around large gaps"""
        
        if gaps is None:
            gaps = DataSplitter.analyze_large_gaps(df, min_gap_days)
        
        if len(gaps) == 0:
            print("No major gaps found. Returning original dataset as single period.")
            return [{'period': 'full', 'data': df, 'start': df.index.min(), 'end': df.index.max()}]
        
        periods = []
        start_idx = 0
        
        for i, gap in enumerate(gaps):
            # Get data before this gap
            end_idx = df.index.get_loc(gap['split_before'])
            period_data = df.iloc[start_idx:end_idx].copy()
            
            if len(period_data) > 1000:  # Only include periods with sufficient data
                periods.append({
                    'period': f'period_{i+1}',
                    'data': period_data,
                    'start': period_data.index.min(),
                    'end': period_data.index.max(),
                    'duration_days': (period_data.index.max() - period_data.index.min()).days,
                    'data_points': len(period_data)
                })
                print(f"Period {i+1}: {period_data.index.min()} to {period_data.index.max()} ({len(period_data)} points, {(period_data.index.max() - period_data.index.min()).days} days)")
            else:
                print(f"Skipping short period {i+1}: only {len(period_data)} data points")
            
            # Set start for next period (after the gap)
            start_idx = df.index.get_loc(gap['split_before'])
        
        # Add final period after last gap
        if start_idx < len(df):
            final_data = df.iloc[start_idx:].copy()
            if len(final_data) > 1000:
                periods.append({
                    'period': f'period_{len(gaps)+1}',
                    'data': final_data,
                    'start': final_data.index.min(),
                    'end': final_data.index.max(),
                    'duration_days': (final_data.index.max() - final_data.index.min()).days,
                    'data_points': len(final_data)
                })
                print(f"Period {len(gaps)+1}: {final_data.index.min()} to {final_data.index.max()} ({len(final_data)} points, {(final_data.index.max() - final_data.index.min()).days} days)")
        
        return periods


class MultiPeriodBacktester:
    """Run strategies across multiple data periods and combine results"""
    
    def __init__(self, periods):
        self.periods = periods
        self.results = {}
    
    def run_zscore_strategy(self, lookback_window=120, entry_threshold=2.0, exit_threshold=0.5):
        """Run Z-Score strategy on all periods"""
        print("\n=== RUNNING Z-SCORE STRATEGY ACROSS PERIODS ===")
        
        from base_strat_bt import ZScoreStrategy
        
        period_results = {}
        combined_metrics = {
            'total_profit': 0,
            'total_trades': 0,
            'all_trade_pnls': []
        }
        
        for period in self.periods:
            print(f"\nRunning Z-Score on {period['period']}...")
            
            strategy = ZScoreStrategy(lookback_window, entry_threshold, exit_threshold)
            results = strategy.backtest(period['data'])
            
            # Calculate metrics for this period using UNIFIED analytics
            analytics = PerformanceAnalytics(results, "Z-Score")
            metrics = analytics.calculate_metrics()
            
            period_results[period['period']] = {
                'results': results,
                'metrics': metrics,
                'period_info': period
            }
            
            # Add to combined metrics
            combined_metrics['total_profit'] += metrics['total_profit']
            combined_metrics['total_trades'] += metrics['total_trades']
            
            trade_pnls = results['trade_pnl'][results['trade_pnl'] != 0]
            combined_metrics['all_trade_pnls'].extend(trade_pnls.tolist())
            
            print(f"  {period['period']}: P&L = {metrics['total_profit']:.4f}, Trades = {metrics['total_trades']}")
        
        # Calculate overall combined metrics
        all_pnls = np.array(combined_metrics['all_trade_pnls'])
        if len(all_pnls) > 0:
            combined_metrics['win_rate'] = (all_pnls > 0).mean() * 100
        else:
            combined_metrics['win_rate'] = 0
        
        self.results['zscore'] = {
            'period_results': period_results,
            'combined_metrics': combined_metrics
        }
        
        return period_results, combined_metrics
    
    def run_kalman_strategy(self, observation_covariance=0.5, state_transition_covariance=1e-4,
                           entry_threshold=1.8, exit_threshold=0.5):
        """Run Kalman Filter strategy on all periods"""
        print("\n=== RUNNING KALMAN FILTER STRATEGY ACROSS PERIODS ===")
        
        period_results = {}
        combined_metrics = {
            'total_profit': 0,
            'total_trades': 0,
            'all_trade_pnls': []
        }
        
        for period in self.periods:
            print(f"\nRunning Kalman Filter on {period['period']}...")
            
            strategy = KalmanFilterStrategy(
                observation_covariance=observation_covariance,
                state_transition_covariance=state_transition_covariance,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold
            )
            results = strategy.backtest(period['data'])
            
            # Calculate metrics for this period using UNIFIED analytics
            analytics = PerformanceAnalytics(results, "Kalman Filter")
            metrics = analytics.calculate_metrics()
            
            period_results[period['period']] = {
                'results': results,
                'metrics': metrics,
                'period_info': period
            }
            
            # Add to combined metrics
            combined_metrics['total_profit'] += metrics['total_profit']
            combined_metrics['total_trades'] += metrics['total_trades']
            
            trade_pnls = results['trade_pnl'][results['trade_pnl'] != 0]
            combined_metrics['all_trade_pnls'].extend(trade_pnls.tolist())
            
            print(f"  {period['period']}: P&L = {metrics['total_profit']:.4f}, Trades = {metrics['total_trades']}")
        
        # Calculate overall combined metrics
        all_pnls = np.array(combined_metrics['all_trade_pnls'])
        if len(all_pnls) > 0:
            combined_metrics['win_rate'] = (all_pnls > 0).mean() * 100
        else:
            combined_metrics['win_rate'] = 0
        
        self.results['kalman'] = {
            'period_results': period_results,
            'combined_metrics': combined_metrics
        }
        
        return period_results, combined_metrics
    
    def run_ou_strategy(self, lookback_window=120, estimation_window=60,
                       min_mean_reversion_speed=0.01, entry_threshold=1.8,
                       exit_threshold=0.5, max_hold_minutes=2000):
        """Run OU Process strategy on all periods"""
        print("\n=== RUNNING OU PROCESS STRATEGY ACROSS PERIODS ===")
        
        period_results = {}
        combined_metrics = {
            'total_profit': 0,
            'total_trades': 0,
            'all_trade_pnls': []
        }
        
        for period in self.periods:
            print(f"\nRunning OU Process on {period['period']}...")
            
            strategy = OUProcessStrategy(
                lookback_window=lookback_window,
                estimation_window=estimation_window,
                min_mean_reversion_speed=min_mean_reversion_speed,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                max_hold_minutes=max_hold_minutes
            )
            results = strategy.backtest(period['data'])
            
            # Calculate metrics for this period using UNIFIED analytics
            analytics = PerformanceAnalytics(results, "OU Process")
            metrics = analytics.calculate_metrics()
            
            period_results[period['period']] = {
                'results': results,
                'metrics': metrics,
                'period_info': period
            }
            
            # Add to combined metrics
            combined_metrics['total_profit'] += metrics['total_profit']
            combined_metrics['total_trades'] += metrics['total_trades']
            
            trade_pnls = results['trade_pnl'][results['trade_pnl'] != 0]
            combined_metrics['all_trade_pnls'].extend(trade_pnls.tolist())
            
            print(f"  {period['period']}: P&L = {metrics['total_profit']:.4f}, Trades = {metrics['total_trades']}")
        
        # Calculate overall combined metrics
        all_pnls = np.array(combined_metrics['all_trade_pnls'])
        if len(all_pnls) > 0:
            combined_metrics['win_rate'] = (all_pnls > 0).mean() * 100
        else:
            combined_metrics['win_rate'] = 0
        
        self.results['ou'] = {
            'period_results': period_results,
            'combined_metrics': combined_metrics
        }
        
        return period_results, combined_metrics
    
    def print_summary(self):
        """Print summary of all strategies across all periods - SIMPLIFIED"""
        print("\n" + "="*80)
        print("MULTI-PERIOD BACKTEST SUMMARY")
        print("="*80)
        
        for strategy_name, strategy_results in self.results.items():
            print(f"\n{strategy_name.upper()} STRATEGY:")
            print("-" * 50)
            
            combined = strategy_results['combined_metrics']
            
            # Calculate additional required metrics from individual period results
            all_avg_wins = []
            all_avg_losses = []
            all_max_drawdowns = []
            all_avg_hold_durations = []
            all_max_hold_durations = []
            
            for period_result in strategy_results['period_results'].values():
                metrics = period_result['metrics']
                if metrics['avg_win'] > 0:
                    all_avg_wins.append(metrics['avg_win'])
                if metrics['avg_loss'] < 0:
                    all_avg_losses.append(metrics['avg_loss'])
                all_max_drawdowns.append(metrics['max_drawdown'])
                all_avg_hold_durations.append(metrics['avg_holding_duration_minutes'])
                all_max_hold_durations.append(metrics['max_holding_duration_minutes'])
            
            # Calculate combined metrics
            avg_win = np.mean(all_avg_wins) if all_avg_wins else 0
            avg_loss = np.mean(all_avg_losses) if all_avg_losses else 0
            max_drawdown = min(all_max_drawdowns) if all_max_drawdowns else 0
            avg_hold_duration = np.mean(all_avg_hold_durations) if all_avg_hold_durations else 0
            max_hold_duration = max(all_max_hold_durations) if all_max_hold_durations else 0
            
            print(f"Total Profit: {combined['total_profit']:.4f}")
            print(f"Win Rate: {combined['win_rate']:.2f}%")
            print(f"Max Drawdown: {max_drawdown:.4f}")
            print(f"Average Holding Duration (minutes): {avg_hold_duration:.2f}")
            print(f"Max Holding Duration (minutes): {max_hold_duration:.2f}")
            print(f"Average Win: {avg_win:.4f}")
            print(f"Average Loss: {avg_loss:.4f}")
    
    def save_results(self, base_filename='multi_period'):
        """Save all results to CSV files"""
        for strategy_name, strategy_results in self.results.items():
            for period_name, period_result in strategy_results['period_results'].items():
                filename = f"{base_filename}_{strategy_name}_{period_name}.csv"
                period_result['results'].to_csv(filename, index=True)
                print(f"Saved {filename}")
    
    def plot_pnl_drawdown(self):
        """Generate P&L and Drawdown plots with SEPARATE curves for each period"""
        for strategy_name, strategy_results in self.results.items():
            print(f"\nGenerating P&L and Drawdown plots for {strategy_name.upper()} strategy...")
            
            # Create simple subplot with just P&L and Drawdown
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(f'{strategy_name.upper()} - Cumulative P&L by Period', f'{strategy_name.upper()} - Drawdown by Period'),
                vertical_spacing=0.1
            )
            
            # Colors for different periods with RGB values
            color_map = [
                {'color': 'blue', 'rgba': 'rgba(0, 0, 255, 0.3)'},
                {'color': 'green', 'rgba': 'rgba(0, 128, 0, 0.3)'},
                {'color': 'orange', 'rgba': 'rgba(255, 165, 0, 0.3)'},
                {'color': 'purple', 'rgba': 'rgba(128, 0, 128, 0.3)'},
                {'color': 'red', 'rgba': 'rgba(255, 0, 0, 0.3)'},
                {'color': 'brown', 'rgba': 'rgba(165, 42, 42, 0.3)'}
            ]
            
            # Plot separate P&L curve for each period
            for i, (period_name, period_result) in enumerate(strategy_results['period_results'].items()):
                results_df = period_result['results']
                period_info = period_result['period_info']
                color_info = color_map[i % len(color_map)]
                
                # Calculate drawdown for this period
                cum_pnl = results_df['cumulative_pnl']
                peak = cum_pnl.expanding().max()
                drawdown = cum_pnl - peak
                
                # 1. Separate P&L curve for each period
                fig.add_trace(
                    go.Scatter(
                        x=results_df.index,
                        y=results_df['cumulative_pnl'],
                        name=f'{period_name} P&L',
                        line=dict(color=color_info['color'], width=2),
                        hovertemplate=f'<b>{period_name}</b><br><b>Time</b>: %{{x}}<br><b>P&L</b>: %{{y:.4f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # 2. Separate Drawdown curve for each period
                fig.add_trace(
                    go.Scatter(
                        x=results_df.index,
                        y=drawdown,
                        name=f'{period_name} DD',
                        line=dict(color=color_info['color'], width=1),
                        fill='tonexty' if i == 0 else None,
                        fillcolor=color_info['rgba'] if i == 0 else None,
                        hovertemplate=f'<b>{period_name}</b><br><b>Time</b>: %{{x}}<br><b>Drawdown</b>: %{{y:.4f}}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Add zero line for drawdown
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title=f'{strategy_name.upper()} Strategy - P&L and Drawdown by Period',
                height=600,
                showlegend=True,
                template='plotly_white',
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="P&L", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown", row=2, col=1)
            
            # Save plot
            filename = f'{strategy_name}_pnl_drawdown_by_period.html'
            fig.write_html(filename)
            print(f"Saved plot: {filename}")


if __name__ == "__main__":
    # Load original data
    print("Loading original data...")
    data = DataPreprocessor.load_and_preprocess('data.parquet')
    
    # Analyze and split data around large gaps
    print("\nAnalyzing data for large gaps...")
    splitter = DataSplitter()
    periods = splitter.split_data_around_gaps(data, min_gap_days=7)
    
    if len(periods) == 1:
        print("No major gaps found. Running on full dataset.")
    else:
        print(f"\nData split into {len(periods)} clean periods.")
    
    # Run multi-period backtesting
    print("\nRunning multi-period backtesting...")
    backtester = MultiPeriodBacktester(periods)
    
    # Run all three strategies
    backtester.run_zscore_strategy(lookback_window=120, entry_threshold=2.0, exit_threshold=0.5)
    backtester.run_kalman_strategy(observation_covariance=0.5, state_transition_covariance=1e-4, 
                                  entry_threshold=1.8, exit_threshold=0.5)
    backtester.run_ou_strategy(lookback_window=120, estimation_window=60,
                              min_mean_reversion_speed=0.01, entry_threshold=1.8,
                              exit_threshold=0.5, max_hold_minutes=2000)
    
    # Print simplified summary
    backtester.print_summary()
    
    # Auto-save all results to CSV
    print("\nSaving detailed results to CSV files...")
    backtester.save_results()
    
    # Generate simple P&L and drawdown plots
    backtester.plot_pnl_drawdown()
    
    print("\nMulti-period backtesting completed!")