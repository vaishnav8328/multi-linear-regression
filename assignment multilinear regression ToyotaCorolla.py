# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 11:25:22 2022

@author: vaishnav
"""
#importing the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\anaconda\\ToyotaCorolla.csv",encoding=('Latin-1'))
df
list(df)


df = df.loc[:,["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]] #taking only required columns
df

df.rename({"Age_08_04":"Age","cc":"CC","Quarterly_Tax":"QT"},axis=1,inplace=True) #renaming the columns
df.columns

df.dtypes

df.isnull().sum()
#==============================================================================================================
#for outliers
#to know if there are outliers or not

for i in df:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3-Q1
    up = Q3 + 1.5*IQR
    low = Q1 - 1.5*IQR

    if df[(df[i] > up) | (df[i] < low)].any(axis=None):
        print(i,"yes")
    else:
        print(i, "no")

# we have examined all the stops, let's remove outliers
outliers=["Age","KM","HP","CC","Gears","QT","Weight"]

for i in df.loc[:,outliers]:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    up_lim = Q3 + 1.5 * IQR
    low_lim = Q1 - 1.5 * IQR
    df.loc[df[i] > up_lim,i] = up_lim
    df.loc[df[i] < low_lim,i] = low_lim

# we fixed outliers

#=====================================================================================================================================
#EDA
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

#boxplots=============================================================================================================
df.boxplot(column="Age",vert=False)

df.boxplot(column="CC",vert=False)

df.boxplot(column="HP",vert=False)

df.boxplot(column="KM",vert=False)

df.boxplot(column="Doors",vert=False)

df.boxplot(column="Gears",vert=False)

df.boxplot(column="QuarterlyTax",vert=False)

df.boxplot(column="Weight",vert=False)
#=============================================================================================================================
#histogram
df["Age"].hist()

df["CC"].hist()

df["HP"].hist()

df["KM"].hist()

df["Doors"].hist()

df["Gears"].hist()

df["QuarterlyTax"].hist()

df["Weight"].hist()

#=================================================================================================================


# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))
# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()

#=================================================================================================================
# Model Building
import statsmodels.formula.api as smf
model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=df).fit()

# Model Testing
# Finding Coefficient parameters
model.params

# Finding tvalues and pvalues
model.tvalues , np.round(model.pvalues,5)

# Finding rsquared values
model.rsquared , model.rsquared_adj 

slr_c=smf.ols('Price~CC',data=df).fit()
slr_c.tvalues , slr_c.pvalues

slr_d=smf.ols('Price~Doors',data=df).fit()
slr_d.tvalues , slr_d.pvalues 

mlr_cd=smf.ols('Price~CC+Doors',data=df).fit()
mlr_cd.tvalues , mlr_cd.pvalues


#=================================================================================================================
# Model Validation Techniques
# Two Techniques: 1. Collinearity Check & 2. Residual Analysis
# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables

rsq_age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=df).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=df).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',data=df).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',data=df).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',data=df).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',data=df).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',data=df).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',data=df).fit().rsquared
vif_WT=1/(1-rsq_WT)


#=================================================================================================================

# Putting the values in Dataframe format
d1={'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df

#=================================================================================================================
# 2) Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)
import statsmodels.api as sm
sm.qqplot(model.resid,line='q') # 'q' - A line is fit through the quartiles # line = '45'- to draw the 45-degree diagonal line
plt.title("Normal Q-Q plot of residuals")
plt.show()

list(np.where(model.resid>6000))

list(np.where(model.resid<-6000))

# Test for Homoscedasticity or Heteroscedasticity (plotting model's standardized fitted values vs standardized residual values)

def standard_values(vals) : return (vals-vals.mean())/vals.std()

plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 

#=================================================================================================================
# Test for errors or Residuals Vs Regressors or independent 'x' variables or predictors 
# using Residual Regression Plots code graphics.plot_regress_exog(model,'x',fig)  


fig=plt.figure(figsize=(12,5))
sm.graphics.plot_regress_exog(model,'Age',fig=fig)
plt.show()

fig=plt.figure(figsize=(12,5))
sm.graphics.plot_regress_exog(model,'KM',fig=fig)
plt.show()

fig=plt.figure(figsize=(12,5))
sm.graphics.plot_regress_exog(model,'HP',fig=fig)
plt.show()

fig=plt.figure(figsize=(12,5))
sm.graphics.plot_regress_exog(model,'CC',fig=fig)
plt.show()

fig=plt.figure(figsize=(12,5))
sm.graphics.plot_regress_exog(model,'Doors',fig=fig)
plt.show()

fig=plt.figure(figsize=(12,5))
sm.graphics.plot_regress_exog(model,'Gears',fig=fig)
plt.show()

fig=plt.figure(figsize=(12,5))
sm.graphics.plot_regress_exog(model,'QT',fig=fig)
plt.show()

fig=plt.figure(figsize=(12,5))
sm.graphics.plot_regress_exog(model,'Weight',fig=fig)
plt.show()

#=================================================================================================================

# Model Deletion Diagnostics (checking Outliers or Influencers)
# Two Techniques : 1. Cook's Distance & 2. Leverage value
# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
from statsmodels.graphics.regressionplots import influence_plot
(c,_)=model.get_influence().cooks_distance
c


# Plot the influencers using the stem plot
fig=plt.figure(figsize=(13,7))
plt.stem(np.arange(len(df)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)

# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
fig,ax=plt.subplots(figsize=(12,6))
fig=influence_plot(model,ax = ax)

# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=df.shape[1]
n=df.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff

#=================================================================================================================

# Improving the Model
# Creating a copy of data so that original dataset is not affected
toyo_new=df.copy()
toyo_new

# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
toyo_1=toyo_new.drop(toyo_new.index[[80]],axis=0).reset_index(drop=True)
toyo_1

# Model Deletion Diagnostics and Final Model
while np.max(c)>0.5 :
    model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyo_1).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    toyo_1=toyo_1.drop(toyo_1.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    toyo_1
else:
    final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyo_1).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)

if np.max(c)>0.5:
    model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyo_1).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    toyo_1=toyo_1.drop(toyo_1.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    toyo_1 
elif np.max(c)<0.5:
    final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyo_1).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)

final_model.rsquared

toyo_1

#=================================================================================================================

# Model Predictions
# save New data for prediction is
new_data=pd.DataFrame({'Age':12,"KM":40000,"HP":80,"CC":1300,"Doors":4,"Gears":5,"QT":69,"Weight":1012},index=[0])
new_data

# Manual Prediction of Price
final_model.predict(new_data)


pred_y=final_model.predict(toyo_1)
pred_y

#the model accurency improved to 84.3%
#=================================================================================================================
















    
