import yfinance as yf
import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#Picking stock 
ticker = 'AAPL'

#getting today and 10 year agos dates
end=datetime.now()
start=end-timedelta(3650)
end=end.strftime("%Y-%m-%d")
start=start.strftime('%Y-%m-%d')


data = yf.download(ticker, start=start, end=end, auto_adjust=True)

# FIX: Flatten the columns if they come back as a MultiIndex (e.g., Price, Ticker)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)


data = data[['Close']].copy()

mu=(np.log(data['Close']/data['Close'].shift(1))).mean()
sigma=(np.log(data['Close']/data['Close'].shift(1))).std()
startprice=data['Close'].iloc[-1]
time_horizon=1
steps = 252
sims=2000

drift=(mu-0.5*sigma**2)*(time_horizon)
volatility=sigma*np.sqrt(time_horizon)
allpaths=[]
for i in range(sims):
    current = startprice
    currentpath=[]
    for j in range(steps):
        rando=np.random.normal(loc=0,scale=1)
        shock=volatility*rando
        current = current * np.exp(shock+drift)
        currentpath.append(current)
    allpaths.append(currentpath)



fig, ax = plt.subplots(figsize=(10, 6))


# We must do this upfront because the axis won't auto-scale during animation
flat_list = [item for sublist in allpaths for item in sublist]
min_y = min(flat_list) * 0.95
max_y = max(flat_list) * 1.05
ax.set_ylim(min_y, max_y)
ax.set_xlim(0, steps)

ax.set_title(f"Monte Carlo Simulation: {ticker} (Animated)")
ax.set_xlabel("Trading Days")
ax.set_ylabel("Price")
ax.axhline(y=startprice, color='black', linestyle='--', linewidth=1.5, label='Start Price')

# We create empty lines now and will fill them with data later
lines = []
for _ in range(sims):
    # Create a line with no data yet, semi-transparent
    line, = ax.plot([], [], alpha=1, lw=1) 
    lines.append(line)


# This function runs once for every frame
def update(frame):
    # 'frame' is just a number that counts up (0, 1, 2... 252)
    
    # X-axis data for this frame (days 0 to current frame)
    x_data = np.arange(frame + 1)
    
    # Update every single line
    for i in range(sims):
        # Grab the price history for this specific run up to the current day
        y_data = allpaths[i][:frame+1]
        
        # Update the line on the graph
        lines[i].set_data(x_data, y_data)
        
    return lines


# frames = how long the movie is (steps)
# interval = speed (20ms is fast, 50ms is standard)
# blit = True makes it run much faster by only redrawing changed pixels
ani = FuncAnimation(fig, update, frames=steps, interval=30, blit=True)

plt.show()