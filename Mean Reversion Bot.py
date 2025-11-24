import yfinance as yf
import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#Picking stock 
ticker = 'AAPL'

#getting today and one year agos dates
end=datetime.now()
start=end-timedelta(365)
end=end.strftime("%Y-%m-%d")
start=start.strftime('%Y-%m-%d')


data = yf.download(ticker, start=start, end=end, auto_adjust=True)

# FIX: Flatten the columns if they come back as a MultiIndex (e.g., Price, Ticker)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Now select just the Close column
data = data[['Close']].copy()

data['SMA20'] = data['Close'].rolling(window=20).mean()
data['VOL'] =data['Close'].rolling(window=20).std()
data['Devs Away']=(data['Close']-data['SMA20'])/data['VOL']

data['Signal']= np.where(data['Devs Away']<-2,1,0)

data['Market Return']= data['Close'].pct_change()
data['Strategy Return']= data['Signal'].shift(1)*data['Market Return']
data['Cumulative_Market'] = (1 + data['Market Return']).cumprod()

data['Cumulative_Strategy'] = (1 + data['Strategy Return']).cumprod()
data[['Cumulative_Market', 'Cumulative_Strategy']].plot(figsize=(10, 5), title='Strategy vs. Buy & Hold ($1 Investment)')
plt.show()