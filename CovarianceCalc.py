import numpy as np
import pandas as pd
import yfinance as yf 
from scipy.optimize import minimize

#Stocks being taken as part of the portfolio
stocks = ['AAPL','TSLA','NVDA','DAL','PFE','GOOG','META']
prices = yf.download(stocks,'2022-01-01')['Close']
#working out the anual returns and covariences of the stocks
#getting the covariences was much easier than i thought not knowing 
#there was a function for it 
logReturns = np.log(prices/prices.shift(1))
averageReturns = np.sum(logReturns.dropna()/(len(logReturns)-1))

covarianceMatrix =logReturns.dropna().cov() 
annualcov=covarianceMatrix*252
annualReturns=averageReturns*252
#this works out the total return by doing the dot product between the 
#wieghts for each stock and the annual average return 

def PortfolioReturn(weights,annualReturns):
    return np.dot(weights,annualReturns)
#risk is a but harder but i used the formula for the variance of 
#two random variables
def PortfolioRisk(weights,annualcov):
    variance = np.dot(weights.T,np.dot(annualcov,weights))
    risk=np.sqrt(variance)
    return risk

num_stocks = len(stocks)

guess = np.array([1/num_stocks] * num_stocks)
target_return = 0.2#
#constraints is a list of dictionaries
constraints = [{'type':'eq','fun':lambda weights: np.sum(weights)-1},
               {'type':'eq',
         'fun':lambda weights:PortfolioReturn(weights,annualReturns)-target_return}]

bounds = tuple((0,1)for _ in range(num_stocks))
print(bounds)
optimalportfolio=minimize(PortfolioRisk,guess,args=(annualcov,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints)
optimalweights=optimalportfolio.x
print(optimalweights)