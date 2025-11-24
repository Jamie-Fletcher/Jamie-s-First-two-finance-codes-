import yfinance as yf
import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 1. Setup
ticker = 'DIS' # Disney
end = datetime.now()
start = end - timedelta(365)

# 2. Download Data
# auto_adjust=True handles stock splits and dividends automatically
data = yf.download(ticker, start=start, end=end, auto_adjust=True)

# FIX: Flatten columns if they come back as MultiIndex (removes the 'DIS' header)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Keep only the Close column to keep things clean
data = data[['Close']].copy()

# 3. Calculate Indicators
data['SMA20'] = data['Close'].rolling(window=20).mean()
data['STD20'] = data['Close'].rolling(window=20).std()

# Calculate Z-Score (How many standard deviations away from mean?)
data['Z_Score'] = (data['Close'] - data['SMA20']) / data['STD20']

# 4. Strategy Logic: State Machine Loop
# We need a loop because we need to remember if we are "in_position" or not
signals = []
in_position = False

# We iterate through the DataFrame. 
# getattr(row, 'ColumnName') is the safest way to access data in the loop.
for row in data.itertuples():
    # Logic for when we are currently CASH (not holding stock)
    if not in_position:
        # BUY SIGNAL: Price is statistically cheap (Z-Score < -2)
        if getattr(row, 'Z_Score') < -2:
            signals.append(1)
            in_position = True
        else:
            signals.append(0)
    
    # Logic for when we are currently HOLDING stock
    else:
        # SELL SIGNAL: Price has reverted to the mean (Price > SMA20)
        if getattr(row, 'Close') > getattr(row, 'SMA20'):
            signals.append(0)
            in_position = False
        else:
            # If we haven't hit the mean yet, KEEP HOLDING
            signals.append(1)

# Add the generated signal list back to the DataFrame
data['Signal'] = signals

# 5. Calculate Returns
data['Market Return'] = data['Close'].pct_change()

# We shift signals by 1 because we buy "at the close" of today, 
# so our profit starts tomorrow.
data['Strategy Return'] = data['Signal'].shift(1) * data['Market Return']

# 6. Calculate Cumulative Performance
data['Cumulative_Market'] = (1 + data['Market Return']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy Return']).cumprod()

# 7. Plot Results
plt.figure(figsize=(10, 5))
plt.plot(data['Cumulative_Market'], label='Buy & Hold (Market)')
plt.plot(data['Cumulative_Strategy'], label='Mean Reversion Bot', linewidth=2)
plt.title(f'{ticker} - Strategy vs. Buy & Hold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()