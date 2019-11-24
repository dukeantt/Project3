# Import the yfinance. If you get module not found error the run !pip install yfiannce from your Jupyter notebook
import yfinance as yf
import matplotlib.pyplot as plt
# Get the data for the stock AAPL
data = yf.download('AAPL','2017-01-01','2019-11-24')
# Import the plotting library
# Plot the close price of the AAPL
data['Adj Close'].plot()
plt.show()