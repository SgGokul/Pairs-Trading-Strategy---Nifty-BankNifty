import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import existing classes - NO DUPLICATION
from base_strat_bt import DataPreprocessor, PerformanceAnalytics

class OUProcessStrategy:
    """Ornstein-Uhlenbeck process based pairs trading strategy"""
    
    def __init__(self, lookback_window=120, estimation_window=60, 
                 min_mean_reversion_speed=0.01, entry_threshold=1.8, 
                 exit_threshold=0.5, min_observations=60, max_hold_minutes=2000):
        self.lookback_window = lookback_window
        self.estimation_window = estimation_window
        self.min_mean_reversion_speed = min_mean_reversion_speed
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_observations = min_observations
        self.max_hold_minutes = max_hold_minutes  # Maximum holding time limit
    
    def estimate_ou_parameters(self, spread_series):
        """
        Estimate OU process parameters using maximum likelihood
        Returns: (mean_reversion_speed, long_term_mean, volatility, quality_score)
        """
        if len(spread_series) < 10:
            return 0, np.mean(spread_series), np.std(spread_series), 0
        
        spreads = np.array(spread_series)
        
        # Calculate first differences
        dt = 1.0  # Assuming unit time steps
        diff_spreads = np.diff(spreads)
        lagged_spreads = spreads[:-1]
        
        try:
            # OU process: dX = a(μ - X)dt + σdW
            # Discretized: ΔX = a(μ - X_{t-1})Δt + σ√Δt * ε
            # Rearranged: ΔX = aμΔt - aΔt*X_{t-1} + noise
            # Linear regression: y = α + β*x + ε where y=ΔX, x=X_{t-1}
            
            X = np.column_stack([np.ones(len(lagged_spreads)), lagged_spreads])
            y = diff_spreads
            
            # Solve using least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta = coeffs[0], coeffs[1]
            
            # Extract OU parameters
            mean_reversion_speed = -beta / dt
            long_term_mean = -alpha / beta if beta != 0 else np.mean(spreads)
            
            # Estimate volatility from residuals
            if len(residuals) > 0 and residuals[0] > 0:
                volatility = np.sqrt(residuals[0] / (len(diff_spreads) - 2)) / np.sqrt(dt)
            else:
                volatility = np.std(diff_spreads) / np.sqrt(dt)
            
            # Quality score based on R-squared and parameter stability
            ss_res = np.sum((y - X @ coeffs) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Additional quality checks
            parameter_stability = 1.0 if mean_reversion_speed > 0 else 0.0
            quality_score = r_squared * parameter_stability
            
            return mean_reversion_speed, long_term_mean, volatility, quality_score
            
        except Exception as e:
            # Fallback to simple estimates
            return 0, np.mean(spreads), np.std(spreads), 0
    
    def backtest(self, df):
        """Run OU Process backtest"""
        results = df.copy()
        
        # Calculate spread
        results['spread'] = results['banknifty'] - results['nifty']
        
        # Initialize columns (base strategy columns)
        results['spread_mean'] = np.nan
        results['spread_std'] = np.nan
        results['z_score'] = 0.0
        results['position'] = 0
        results['trade_pnl'] = 0.0
        results['cumulative_pnl'] = 0.0
        
        # Initialize OU-specific columns for detailed analysis
        results['ou_speed'] = 0.0
        results['ou_mean'] = np.nan
        results['ou_volatility'] = np.nan
        results['ou_quality'] = 0.0
        results['ou_distance'] = 0.0
        results['ou_half_life'] = np.nan
        results['trade_signal'] = 0  # 1=entry, -1=exit, 0=hold
        results['hold_minutes'] = 0.0
        
        # Track trade state
        position = 0
        entry_spread = 0
        entry_tte = 0
        entry_time = None
        cumulative_pnl = 0
        
        # Track positions for debugging
        open_positions = []
        
        # Loop until len-1 to avoid index out of bounds for i+1
        for i in range(self.min_observations, len(results) - 1):
            current_spread = results['spread'].iloc[i]
            current_tte = results['tte'].iloc[i]
            current_time = results.index[i]
            
            # Next period's data for execution (avoid look-ahead bias)
            next_spread = results['spread'].iloc[i + 1]
            next_tte = results['tte'].iloc[i + 1]
            
            # Calculate holding time if in position
            hold_minutes = 0
            if position != 0 and entry_time is not None:
                hold_minutes = (current_time - entry_time).total_seconds() / 60
            
            # Get historical data for OU parameter estimation (no look-ahead)
            start_idx = max(0, i - self.lookback_window)
            estimation_data = results['spread'].iloc[start_idx:i].values  # Only up to current time
            
            # Use more recent data for parameter estimation if available
            if len(estimation_data) > self.estimation_window:
                estimation_data = estimation_data[-self.estimation_window:]
            
            # Estimate OU parameters
            speed, mean, volatility, quality = self.estimate_ou_parameters(estimation_data)
            
            # Calculate OU half-life (time for deviation to decay to 50%)
            half_life = np.log(2) / speed if speed > 0 else np.inf
            
            # Store OU parameters
            results['ou_speed'].iloc[i] = speed
            results['ou_mean'].iloc[i] = mean
            results['ou_volatility'].iloc[i] = volatility
            results['ou_quality'].iloc[i] = quality
            results['ou_half_life'].iloc[i] = half_life if np.isfinite(half_life) else np.nan
            results['hold_minutes'].iloc[i] = hold_minutes
            
            # Calculate distance from OU mean and z-score
            distance_from_mean = current_spread - mean
            results['ou_distance'].iloc[i] = distance_from_mean
            
            # Store values (using same column names as base strategy for compatibility)
            results['spread_mean'].iloc[i] = mean
            results['spread_std'].iloc[i] = volatility
            
            # Calculate OU-based z-score
            if volatility > 0:
                z_score = distance_from_mean / volatility
            else:
                z_score = 0
            
            results['z_score'].iloc[i] = z_score
            
            # Trading logic - enhanced conditions
            trade_condition = (speed > self.min_mean_reversion_speed and 
                             quality > 0.05 and volatility > 0 and
                             np.isfinite(half_life) and half_life < 500)  # Reasonable half-life
            
            # Initialize trade signal
            trade_signal = 0
            
            if trade_condition:
                # Entry signals (detect at i, execute at i+1)
                if position == 0:
                    if z_score > self.entry_threshold:
                        position = -1  # Short spread (expecting mean reversion)
                        entry_spread = next_spread  # Execute at next period
                        entry_tte = current_tte     # Use current TTE for P/L calculation
                        entry_time = current_time
                        trade_signal = 1
                        open_positions.append({'entry_time': current_time, 'entry_spread': entry_spread})
                    elif z_score < -self.entry_threshold:
                        position = 1   # Long spread (expecting mean reversion)
                        entry_spread = next_spread  # Execute at next period
                        entry_tte = current_tte     # Use current TTE for P/L calculation
                        entry_time = current_time
                        trade_signal = 1
                        open_positions.append({'entry_time': current_time, 'entry_spread': entry_spread})
                
                # Exit signals (detect at i, execute at i+1)
                elif abs(z_score) < self.exit_threshold:
                    # Calculate P/L using next period's spread (realistic execution)
                    exit_spread = next_spread
                    spread_change = exit_spread - entry_spread
                    trade_pnl = position * spread_change * (entry_tte ** 0.7)
                    
                    results['trade_pnl'].iloc[i + 1] = trade_pnl  # Record P/L at execution time
                    cumulative_pnl += trade_pnl
                    trade_signal = -1
                    
                    # Reset position
                    position = 0
                    entry_spread = 0
                    entry_tte = 0
                    entry_time = None
                    if open_positions:
                        open_positions.pop()
            
            # Force exit conditions
            force_exit = False
            exit_reason = ""
            
            if position != 0:
                # Maximum holding time exit
                if hold_minutes > self.max_hold_minutes:
                    force_exit = True
                    exit_reason = "max_hold_time"
                
                # Quality deterioration exit
                elif speed < self.min_mean_reversion_speed or quality < 0.02:
                    force_exit = True
                    exit_reason = "quality_deterioration"
                
                # Extreme z-score exit (spread moved too far against mean reversion)
                elif abs(z_score) > 5.0:
                    force_exit = True
                    exit_reason = "extreme_zscore"
            
            if force_exit:
                # Emergency exit
                exit_spread = next_spread
                spread_change = exit_spread - entry_spread
                trade_pnl = position * spread_change * (entry_tte ** 0.7)
                
                results['trade_pnl'].iloc[i + 1] = trade_pnl
                cumulative_pnl += trade_pnl
                trade_signal = -1
                
                # Reset position
                position = 0
                entry_spread = 0
                entry_tte = 0
                entry_time = None
                if open_positions:
                    open_positions.pop()
            
            # Store current state
            results['trade_signal'].iloc[i] = trade_signal
            results['position'].iloc[i] = position
            results['cumulative_pnl'].iloc[i] = cumulative_pnl
        
        # Handle the final row properly - just carry forward the final state
        if len(results) > 0:
            results['position'].iloc[-1] = 0  # Ensure no open position at end
            results['cumulative_pnl'].iloc[-1] = cumulative_pnl  # Preserve final P&L
            
            # If there was an open position at end, close it with final spread
            if position != 0:
                final_spread = results['spread'].iloc[-1]
                spread_change = final_spread - entry_spread
                final_trade_pnl = position * spread_change * (entry_tte ** 0.7)
                results['trade_pnl'].iloc[-1] = final_trade_pnl
                cumulative_pnl += final_trade_pnl
                results['cumulative_pnl'].iloc[-1] = cumulative_pnl
        
        return results


if __name__ == "__main__":
    # Load and preprocess data using yfinance trading calendar
    data = DataPreprocessor.load_and_preprocess('data.parquet')
    
    # Run OU Process strategy with improved parameters
    strategy = OUProcessStrategy(
        lookback_window=120,
        estimation_window=60,
        min_mean_reversion_speed=0.01,
        entry_threshold=1.8,
        exit_threshold=0.5,
        max_hold_minutes=2000  # Maximum 33 hours holding time
    )
    results = strategy.backtest(data)
    
    # Save results to CSV
    results.to_csv('ou_strategy_results.csv', index=True)
    print("Results saved to ou_strategy_results.csv")
    
    # Analyze performance with unified analytics - NO DUPLICATION
    analytics = PerformanceAnalytics(results, "OU Process")
    metrics = analytics.calculate_metrics()
    
    # Display results
    analytics.print_summary()
    
    # Generate simple plots
    analytics.plot_results(save_html=True, filename='ou_pnl_drawdown.html')