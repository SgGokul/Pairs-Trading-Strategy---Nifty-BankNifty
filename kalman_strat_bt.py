import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import existing classes - NO DUPLICATION
from base_strat_bt import DataPreprocessor, PerformanceAnalytics

class KalmanFilterStrategy:
    """Kalman Filter based pairs trading strategy"""
    
    def __init__(self, initial_state_mean=0, initial_state_covariance=1, 
                 observation_covariance=0.5, state_transition_covariance=1e-4,
                 entry_threshold=1.8, exit_threshold=0.5, min_observations=60):
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.obs_cov = observation_covariance
        self.state_cov = state_transition_covariance
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_observations = min_observations
        
    def kalman_update(self, state_mean, state_covariance, observation):
        """Single Kalman filter update step"""
        # Prediction step
        predicted_state_mean = state_mean
        predicted_state_covariance = state_covariance + self.state_cov
        
        # Update step
        kalman_gain = predicted_state_covariance / (predicted_state_covariance + self.obs_cov)
        updated_state_mean = predicted_state_mean + kalman_gain * (observation - predicted_state_mean)
        updated_state_covariance = (1 - kalman_gain) * predicted_state_covariance
        
        return updated_state_mean, updated_state_covariance, kalman_gain, (observation - predicted_state_mean)
    
    def backtest(self, df):
        """Run Kalman Filter backtest"""
        results = df.copy()
        
        # Calculate spread
        results['spread'] = results['banknifty'] - results['nifty']
        
        # Initialize columns (following base structure)
        results['spread_mean'] = np.nan
        results['spread_std'] = np.nan
        results['z_score'] = 0.0
        results['position'] = 0
        results['trade_pnl'] = 0.0
        results['cumulative_pnl'] = 0.0
        
        # Initialize additional diagnostic columns
        results['kf_state_covariance'] = np.nan
        results['kf_kalman_gain'] = np.nan
        results['kf_innovation'] = np.nan
        results['kf_distance'] = 0.0
        results['trade_signal'] = 0  # 1=entry, -1=exit, 0=hold
        results['hold_minutes'] = 0.0
        results['entry_time'] = pd.NaT
        results['exit_reason'] = ''
        
        # Initialize Kalman Filter state
        state_mean = self.initial_state_mean
        state_covariance = self.initial_state_covariance
        
        # Track residuals for adaptive standard deviation
        residuals_history = []
        
        # Track trade state
        position = 0
        entry_spread = 0
        entry_tte = 0
        entry_time = None
        cumulative_pnl = 0
        
        # Loop until len-1 to avoid index out of bounds for i+1
        for i in range(len(results) - 1):
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
            
            # Update Kalman filter with current observation
            state_mean, state_covariance, kalman_gain, innovation = self.kalman_update(state_mean, state_covariance, current_spread)
            
            # Store additional diagnostic information
            results['kf_state_covariance'].iloc[i] = state_covariance
            results['kf_kalman_gain'].iloc[i] = kalman_gain
            results['kf_innovation'].iloc[i] = innovation
            results['hold_minutes'].iloc[i] = hold_minutes
            if entry_time is not None:
                results['entry_time'].iloc[i] = entry_time
            
            # Calculate residual
            residual = current_spread - state_mean
            residuals_history.append(residual)
            
            # Calculate distance from filter mean
            distance_from_mean = current_spread - state_mean
            results['kf_distance'].iloc[i] = distance_from_mean
            
            # Calculate adaptive standard deviation using recent residuals
            if len(residuals_history) >= self.min_observations:
                # Use rolling window for standard deviation (no look-ahead)
                recent_residuals = residuals_history[-120:]  # Last 120 observations
                residual_std = np.std(recent_residuals)
                
                # Calculate z-score
                z_score = residual / residual_std if residual_std > 0 else 0
                
                # Store values (using same column names as base strategy)
                results['spread_mean'].iloc[i] = state_mean
                results['spread_std'].iloc[i] = residual_std  # Store actual std, not covariance
                results['z_score'].iloc[i] = z_score
                
                # Initialize trade signal
                trade_signal = 0
                exit_reason = ''
                
                # Entry signals (detect at i, execute at i+1)
                if position == 0:
                    if z_score > self.entry_threshold:
                        position = -1  # Short spread (expecting mean reversion)
                        entry_spread = next_spread  # Execute at next period
                        entry_tte = current_tte     # Use current TTE for P/L calculation
                        entry_time = current_time   # Track entry time
                        trade_signal = 1            # Mark entry signal
                    elif z_score < -self.entry_threshold:
                        position = 1   # Long spread (expecting mean reversion)
                        entry_spread = next_spread  # Execute at next period
                        entry_tte = current_tte     # Use current TTE for P/L calculation
                        entry_time = current_time   # Track entry time
                        trade_signal = 1            # Mark entry signal
                
                # Exit signals (detect at i, execute at i+1)
                elif abs(z_score) < self.exit_threshold:
                    # Calculate P/L using next period's spread (realistic execution)
                    exit_spread = next_spread
                    spread_change = exit_spread - entry_spread
                    trade_pnl = position * spread_change * (entry_tte ** 0.7)
                    
                    results['trade_pnl'].iloc[i + 1] = trade_pnl  # Record P/L at execution time
                    cumulative_pnl += trade_pnl
                    trade_signal = -1           # Mark exit signal
                    exit_reason = 'normal_exit' # Track exit reason
                    
                    position = 0
                    entry_spread = 0
                    entry_tte = 0
                    entry_time = None           # Reset entry time
                
                # Store trade signal and exit reason
                results['trade_signal'].iloc[i] = trade_signal
                results['exit_reason'].iloc[i] = exit_reason
                
            else:
                # Store values for early periods (before enough observations)
                results['spread_mean'].iloc[i] = state_mean
                results['spread_std'].iloc[i] = np.sqrt(state_covariance) if state_covariance > 0 else 0.001
                results['z_score'].iloc[i] = 0.0
                # Store additional values for early periods
                results['trade_signal'].iloc[i] = 0
                results['exit_reason'].iloc[i] = ''
            
            # Store current state
            results['position'].iloc[i] = position
            results['cumulative_pnl'].iloc[i] = cumulative_pnl
        
        # Handle final position if still open
        if position != 0:
            results['position'].iloc[-1] = position
            results['cumulative_pnl'].iloc[-1] = cumulative_pnl
        
        return results


if __name__ == "__main__":
    # Load and preprocess data using yfinance trading calendar
    data = DataPreprocessor.load_and_preprocess('data.parquet')
    
    # Run Kalman Filter strategy
    strategy = KalmanFilterStrategy(
        observation_covariance=0.5,
        state_transition_covariance=1e-4,
        entry_threshold=1.8,
        exit_threshold=0.5
    )
    results = strategy.backtest(data)
    
    # Save results to CSV
    results.to_csv('kalman_strategy_results.csv', index=True)
    print("Results saved to kalman_strategy_results.csv")
    
    # Analyze performance with unified analytics - NO DUPLICATION
    analytics = PerformanceAnalytics(results, "Kalman Filter")
    metrics = analytics.calculate_metrics()
    
    # Display results
    analytics.print_summary()
    
    # Generate simple plots
    analytics.plot_results(save_html=True, filename='kalman_pnl_drawdown.html')