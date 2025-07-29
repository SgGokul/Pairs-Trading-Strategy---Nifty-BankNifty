# Pairs-Trading-Strategy Nifty-BankNifty

File Functions

base_strat_bt.py 

Implements Z-Score strategy with rolling mean/standard deviation
Includes data preprocessing (trading hours, yfinance calendar)
Generates interactive performance plots and CSV results
Run individually: python base_strat_bt.py

kalman_strat_bt.py

Implements Kalman Filter strategy with state space modeling
Tracks hidden state evolution with optimal estimation
Generates enhanced analytics with Kalman-specific diagnostics
Run individually: python kalman_strat_bt.py

ou_strat_bt.py

Implements OU Process strategy with dynamic parameter estimation
Includes quality controls and multiple exit mechanisms
Generates comprehensive OU-specific performance analytics
Run individually: python ou_strat_bt.py

masterfile.py

Runs all three strategies together on split data periods
Handles data gap analysis and period splitting
Compares performance across strategies and periods
Generates multi-period CSV files for each strategy
Run for complete analysis: python masterfile.py

Data Files
data.parquet

Main dataset containing Nifty and Bank Nifty minute-level data
Covers January 2021 to June 2022 (370 trading days, 122,952 data points)

Output Files
Multi-Period CSV Results:

multi_period_zscore_period_1.csv / multi_period_zscore_period_2.csv - Z-Score trade details by period
multi_period_kalman_period_1.csv / multi_period_kalman_period_2.csv - Kalman Filter trade details by period
multi_period_ou_period_1.csv / multi_period_ou_period_2.csv - OU Process trade details by period

Interactive Performance Plots:

zscore_pnl_drawdown_by_period.html - Z-Score performance visualization
kalman_pnl_drawdown_by_period.html - Kalman Filter performance visualization
ou_pnl_drawdown_by_period.html - OU Process performance visualization
