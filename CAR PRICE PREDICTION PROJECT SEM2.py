#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Importing relevant important Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[6]:


data = pd.read_csv(r"C:\Users\PRAPANJ K M\Downloads\CarPrice_Assignment.csv")
data


# In[7]:


data.info()


# #The dataset has total of 205 car models having total of 26 attributes (25 features + 1 label (price))
# 
# #There are no null values

# In[8]:


data.describe()


# #Lets see some visualization for the given data

# In[9]:


corr_matrix = data.corr()
corr_matrix["price"].sort_values(ascending=False)


# In[10]:


sns.pairplot(data)


# In[11]:


sns.pairplot(data[["price", "enginesize", "curbweight", "horsepower", "carwidth", "carlength", "wheelbase", "boreratio"]])


# In[12]:


sns.pairplot(data[["price", "carheight", "stroke", "compressionratio", "symboling", "peakrpm", "wheelbase", "boreratio"]])


# #Looking at these histograms, and scatters, we can conclude that all numeric attributes are normal distributed except 'wheelbase', 'enginesize', 'compressionratio' and 'price' are right skewed which may cuase some effect on model training, so rescaling will be needed next.
# #Also, the pairplot plotted with the first set of feature shows more corelation with price than the one plotted above. This backs the conclusion made by looking at the corelation matrix

# #Also, the pairplot plotted with the first set of feature shows more corelation with price than the one plotted above. 
# #This backs the conclusion made by looking at the corelation matrix

# In[13]:


def categories():
    for i in range(len(data.columns)):
        #if data[data.columns[i]].nunique() < 20:
        if data[data.columns[i]].nunique():
            if data[data.columns[i]].nunique() <= 7: 
                print("Number of Unique categories in",data.columns[i],data[data.columns[i]].nunique())
categories()


# In[14]:


dummies_data = pd.get_dummies(data[["symboling","fueltype","aspiration","doornumber","carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber"]])
dummies_data


# In[15]:


final_data = pd.concat([data, dummies_data],axis=1)
final_data


# # #Separating Training and Testing dataset from original dataset for developing machine learning algorithms
# 
# 

# In[16]:


final_data = final_data.drop(["CarName","symboling","fueltype","aspiration","doornumber","carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber","fuelsystem"],axis=1)


# In[17]:


final_data


# In[18]:


X = final_data.drop(columns=["price"], axis=1)


# In[19]:


X


# In[20]:


y = final_data["price"]
y


# In[21]:


# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[22]:


# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


final_data.info()


# # Now all the dataset is in numerical format after converting all the categorical features into numerical using dummies
# 

# # Now applying Machine Learning Algos

# # Linear Regression

# In[24]:


#Import Required Libraries for Linear Regression

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


ml_regr = linear_model.LinearRegression()
ml_regr.fit(X_train, y_train)

#Print the results of intercept and Coefficient values
print('Intercept: ', ml_regr.intercept_)
print('Coefficients: ', ml_regr.coef_)

# Prediction step
y_pred_ml = ml_regr.predict(X_test)  

# prediction accuracy score check
score=r2_score(y_test,y_pred_ml)
print('R2 socre is',score)
print('mean_sqrd_error is==',mean_squared_error(y_test,y_pred_ml))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_pred_ml)))

# scatter for actual vs predition in MultiLinear Model

plt.scatter(x=y_test, y=y_pred_ml,c='crimson')
p1 = max(max(y_pred_ml), max(y_test))
p2 = min(min(y_pred_ml), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Multi-Linear reg Prediction')


# # Decision Tree Regressor 

# In[25]:


# Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
decision_Tree_reg =DecisionTreeRegressor()
decision_Tree_reg.fit(X_train,y_train)

# Model Prediction
prediction_tree = decision_Tree_reg.predict(X_test)


# predicting the accuracy score
score=r2_score(y_test,prediction_tree)
print('R2 socre is',score)
print('mean_sqrd_error is==',mean_squared_error(y_test,prediction_tree))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,prediction_tree)))


# find scatter plot for prediction and actual

plt.scatter(x=y_test, y=prediction_tree, c= 'crimson')
p1 = max(max(prediction_tree), max(y_test))
p2 = min(min(prediction_tree), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual')
plt.ylabel('Decision Tree Prediction')
plt.title('Actual vs. Decision Tree Prediction')


# # Random Forest

# In[29]:


from sklearn.ensemble import RandomForestRegressor

Random_FR = RandomForestRegressor(n_estimators=500, n_jobs=-1)

Random_FR.fit(X_train, y_train)
y_pred_RF = Random_FR.predict(X_test)

# Plot a scatter between predicted and actual
plt.scatter(y_test, y_pred_RF, s=20, c='crimson')
plt.title('Actual vs Predictions in Random Forest')
plt.xlabel('Actual')
plt.ylabel('Predicted in RF')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout()


# predicting the accuracy score
score=r2_score(y_test,y_pred_RF)
print('R2 socre is',score)
print('mean_sqrd_error is =',mean_squared_error(y_test,y_pred_RF))
print('root_mean_squared error of is =',np.sqrt(mean_squared_error(y_test,y_pred_RF)))


# In[28]:


models = pd.DataFrame({
    'Model' : ['Linear Regression', 'Decision Tree', 'Random Forest'],
    'Score' : [ml_regr.score(X_test, y_test), decision_Tree_reg.score(X_test, y_test), Random_FR.score(X_test, y_test)]
})


models.sort_values(by = 'Score', ascending = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




