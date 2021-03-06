# -*- coding: utf-8 -*-
"""
DSC 530
Term Project - Using 2020 NBA Data to Predict Positions
Author: David Brehm
"""

import os
os.chdir(r'D:\School\530\Final')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import statsmodels.graphics.gofplots as gp 
import statsmodels.api as sm

plt.style.use('fivethirtyeight')

#------------------------------------------------------------------------------
# Initial data import and cleaning.

data = pd.read_csv('2020NBAPlayerStats.csv', index_col = 'Rk')  # Read data.

dataGS = data[data.GS >= 36]  # Filter data to players who have started at least half the season.

dataFinal = dataGS[['Pos','3P','TRB','AST','STL','BLK','TOV','PTS']]

dataFinal.Pos = [x.split('-')[0] for x in dataGS.Pos]  # Assumption: Keep the first value if multiple positions. 
dataFinal.Pos.value_counts()  # Observe final values for position.

dataFinal.isnull().sum()  # Check if any columns have null values.

n_bin = int(np.sqrt(len(dataFinal)))  # Set number of bins to be sqrt of the samples.


#------------------------------------------------------------------------------
# Summary Statistics

summary = dataFinal.describe()  # Observe mean, std, min, max, and quartiles.

# Using value counts to find mode.
dataFinal['3P'].value_counts()
dataFinal['TRB'].value_counts()
dataFinal['AST'].value_counts()
dataFinal['STL'].value_counts()
dataFinal['BLK'].value_counts()
dataFinal['TOV'].value_counts()
dataFinal['PTS'].value_counts()


#------------------------------------------------------------------------------
# Histograms
# 3P
plt.figure(figsize=(5.5,4))
plt.hist(dataFinal['3P'], bins = n_bin)
plt.title('3P Distribution')
plt.xlabel('3P Per Game')
plt.ylabel('Number of Players')

plt.figure(figsize=(3,4))
box = sns.boxplot(y = '3P', data = dataFinal, showfliers = False)
box = sns.swarmplot(y = '3P', data = dataFinal, color='gray')
plt.ylabel('3P Per Game')

# TRB
plt.figure(figsize=(5.5,4))
plt.hist(dataFinal['TRB'], bins = n_bin)
plt.title('Total Rebound Distribution')
plt.xlabel('TRB Per Game')
plt.ylabel('Number of Players')

plt.figure(figsize=(3,4))
box = sns.boxplot(y = 'TRB', data = dataFinal, showfliers = False)
box = sns.swarmplot(y = 'TRB', data = dataFinal, color='gray')
plt.ylabel('Total Rebounds Per Game')

# AST
plt.figure(figsize=(5.5,4))
plt.hist(dataFinal['AST'], bins = n_bin)
plt.title('Assist Distribution')
plt.xlabel('Assists Per Game')
plt.ylabel('Number of Players')

plt.figure(figsize=(3,4))
box = sns.boxplot(y = 'AST', data = dataFinal, showfliers = False)
box = sns.swarmplot(y = 'AST', data = dataFinal, color='gray')
plt.ylabel('Assists Per Game')

# STL
plt.figure(figsize=(5.5,4))
plt.hist(dataFinal['STL'], bins = n_bin)
plt.title('Steal Distribution')
plt.xlabel('Steals Per Game')
plt.ylabel('Number of Players')

plt.figure(figsize=(3,4))
box = sns.boxplot(y = 'STL', data = dataFinal, showfliers = False)
box = sns.swarmplot(y = 'STL', data = dataFinal, color='gray')
plt.ylabel('Steals Per Game')

# BLK
plt.figure(figsize=(5.5,4))
plt.hist(dataFinal['BLK'], bins = n_bin)
plt.title('Block Distribution')
plt.xlabel('Blocks Per Game')
plt.ylabel('Number of Players')

plt.figure(figsize=(3,4))
box = sns.boxplot(y = 'BLK', data = dataFinal, showfliers = False)
box = sns.swarmplot(y = 'BLK', data = dataFinal, color='gray')
plt.ylabel('Blocks Per Game')

# TOV
plt.figure(figsize=(5.5,4))
plt.hist(dataFinal['TOV'], bins = n_bin)
plt.title('Turnover Distribution')
plt.xlabel('Turnovers Per Game')
plt.ylabel('Number of Players')

plt.figure(figsize=(3,4))
box = sns.boxplot(y = 'TOV', data = dataFinal, showfliers = False)
box = sns.swarmplot(y = 'TOV', data = dataFinal, color='gray')
plt.ylabel('Turnovers Per Game')

# PTS
plt.figure(figsize=(5.5,4))
plt.hist(dataFinal['PTS'], bins = n_bin)
plt.title('Scoring Distribution')
plt.xlabel('Point Per Game')
plt.ylabel('Number of Players')

plt.figure(figsize=(3,4))
box = sns.boxplot(y = 'PTS', data = dataFinal, showfliers = False)
box = sns.swarmplot(y = 'PTS', data = dataFinal, color='gray')
plt.ylabel('Points Per Game')


#------------------------------------------------------------------------------
# Other Distributions
# PMF
dataASThigh = dataFinal[dataFinal['AST'] >= 3.38]  # Split data by average assists.
dataASTlow = dataFinal[dataFinal['AST'] < 3.38]  # Split data by average assists.

# Get PMF values
hHeight, pmfBin = np.histogram(dataASThigh['TOV'], bins = n_bin)
lHeight, _ = np.histogram(dataASTlow['TOV'], bins = n_bin)
hHeight = hHeight/sum(hHeight)
lHeight = lHeight/sum(lHeight)
centers = 0.5*(pmfBin[1:] + pmfBin[:-1])
widths = np.diff(pmfBin)/2

# Plot PMF
fig, ax = plt.subplots()
ax.bar(centers-widths, hHeight, width = widths, label = 'Above Average Assists')
ax.bar(centers, lHeight, width = widths, label = 'Below Average Assists')
ax.legend(fontsize = '10')
ax.set_xlabel('Turnovers Per Game')
ax.set_ylabel('Probability')
ax.set_title('Turnovers PMF by Assists')


# TRB CDF
counts, bin_edges = np.histogram(dataFinal['TRB'], bins = n_bin)
cdf = np.cumsum(counts) / len(dataFinal)

plt.figure(figsize=(5.5,4))
plt.plot(bin_edges[1:], cdf)
plt.title('Total Rebounds CDF')
plt.xlabel('TRB Per Game')
plt.ylabel('CDF')


# PTS normal probability
gp.ProbPlot(dataFinal['PTS']).qqplot(line='s')
plt.title('Scoring QQ Plot')


#------------------------------------------------------------------------------
# Compare Variables
# 3P vs PTS
plt.figure(figsize=(5.5,4))
plt.scatter(dataFinal['3P'], dataFinal['PTS'])
plt.title('3P vs Points Per Game')
plt.xlabel('3P Per Game')
plt.ylabel('Points Per Game')

ss.pearsonr(dataFinal['3P'], dataFinal['PTS'])  # Pearson correlation
np.cov(dataFinal['3P'], dataFinal['PTS'])  # Covariance

# STL vs BLK
plt.figure(figsize=(5.5,4))
plt.scatter(dataFinal['BLK'], dataFinal['STL'])
plt.title('Blocks vs Steals Per Game')
plt.xlabel('Blocks Per Game')
plt.ylabel('Steals Per Game')

ss.pearsonr(dataFinal['BLK'], dataFinal['STL'])  # Pearson correlation
np.cov(dataFinal['BLK'], dataFinal['STL'])  # Covariance


#------------------------------------------------------------------------------
# Correlation Testing
# Correlation Matrix as a heatmap
corMat = dataFinal.corr()
heatmap = sns.heatmap(corMat, annot = True)
heatmap.tick_params(labelsize = 10)
plt.title('Correlation Matrix')

# Testing AST and TRB correlation
dfAST_TRB = pd.DataFrame(columns = ['Pearson Correlation', 'P-value'])
for i in range(0,1000):
    ast = np.random.permutation(dataFinal['AST'])  # Shuffle assist values
    cor = ss.pearsonr(ast, dataFinal['TRB'])  # Pearson correlation
    dfAST_TRB.loc[i] = [cor[0], cor[1]]  # Append correlation coefficient and p-value
    
    
#------------------------------------------------------------------------------
# Building a Logistic Regression model
# Divide data into test and train portions.
x_train, x_test, y_train, y_test = train_test_split(dataFinal.drop(['Pos'], axis=1), 
                                                    dataFinal.Pos ,test_size = 0.2, random_state=11)

lg = LogisticRegression().fit(x_train, y_train)  # Implement logistic regression model.
predictions = lg.predict(x_test)  # Use model to make predictions.
print(classification_report(y_test, predictions))

# Multinominal Logistic Regression
MN = sm.MNLogit(y_train, x_train)
result = MN.fit()
print(result.summary())