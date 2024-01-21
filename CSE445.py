#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip3 install -U ucimlrepo')


# ## DATASET
# 

#  Bike sharing systems represent a new wave of bicycle rentals where membership, rental, and return are all handled automatically. These systems make it simple for users to borrow a bike from one location and drop it off at another. There are currently more than 500 bike-sharing programmes operating worldwide, with over 500 thousand bicycles. These systems are of tremendous interest now because of their significance in relation to transportation, environmental, and health issues. 

# Date/Time: Information about when the bike rides occurred.
# Weather Conditions: Parameters like temperature, humidity, wind speed, and weather type (e.g., sunny, rainy).
# Holiday Information: Indicating whether a particular day is a holiday or not.
# Rental Details: Including the number of bikes rented, both casual and registered users.
# Season: Categorizing the data into seasons (e.g., spring, summer, fall, winter).
# Working Day: Indicating whether a day is a working day or a weekend.
# User Type: Differentiating between casual and registered users.
# Rental Count: The target variable representing the number of bike rentals.
# 
# There are two dataset for day and hour with 16 and 17 entries respectively and 731 and 17379 index respectively. The datatypes are of int,object,float. There is no null count. 
# 

# In[3]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
bike_sharing_dataset = fetch_ucirepo(id=275) 
  
# data (as pandas dataframes) 
X = bike_sharing_dataset.data.features 
y = bike_sharing_dataset.data.targets 
  
# metadata 
print(bike_sharing_dataset.metadata) 
  
# variable information 
print(bike_sharing_dataset.variables) 


# In[4]:


#Import necessary libraries
#for DA and array processing
import pandas as pd
import numpy as np


# In[5]:


#For Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, probplot


# In[6]:


#For Statistical modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[7]:


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[30]:


#Load the dataset
ddf=pd.read_csv('C:/Users/samiu/Downloads/bike+sharing+dataset/day.csv')
df=pd.read_csv('C:/Users/samiu/Downloads/bike+sharing+dataset/hour.csv')


# In[31]:


#Read the data
df.head()


# In[36]:


#Exploring data frame,identifying Potential Errors and understanding the datatypes
pd.set_option('display.max_columns', None) 
def data_overview(df, head=5):
    print(" SHAPE ".center(125,'-'))
    print('Rows:{}'.format(df.shape[0]))
    print('Columns:{}'.format(df.shape[1]))
    print(" MISSING VALUES ".center(125,'-'))
    print(df.isnull().sum())
    print(" DUPLICATED VALUES ".center(125,'-'))
    print(df.duplicated().sum())
    print(" HEAD ".center(125,'-'))
    print(df.head(3))
    print(" DATA TYPES ".center(125,'-'))
    print(df.dtypes)
    
data_overview(df)


# In[37]:


#Checking ouliers in Target varibale "cnt".

Q1 = df['cnt'].quantile(0.25)
Q3 = df['cnt'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for the outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify the outliers
outliers = df[(df['cnt'] < lower_bound) | (df['cnt'] > upper_bound)]
outliers.style.background_gradient(cmap='Greys')


# In[38]:


#removing outliers
df = df[(df['cnt'] >= lower_bound) & (df['cnt'] <= upper_bound)]
print("shape after outliers removal :",df.shape)


# In[39]:


# converting discrete variable "season" to bins
df = pd.get_dummies(df, columns=['season'], dtype=int)
df.head()


# ##  Exploratory Data Analysis

# In[40]:


#EDA
plt.figure(figsize=(12,6))
ax = sns.boxplot(x='hr', y='cnt', data=df)
plt.title('Distribution of bike rentals per hour')
for i in ax.containers:
    ax.bar_label(i,)
plt.show()


# In[41]:


plt.figure(figsize=(12,6))
sns.boxplot(x='weekday', y='cnt', data=df)
plt.title('Distribution of bike rentals V/S days of the week')


# In[42]:


plt.figure(figsize=(12,6))
d = sns.FacetGrid(df, col="workingday")
d. map(sns.barplot, "hr", "cnt")


# In[43]:


plt.figure(figsize=(12,6))
sns.boxplot(x='mnth', y='cnt', data=df)
plt.title('Distribution of bike rentals V/S months')
plt.show()


# In[44]:


df['season'] = df[['season_1', 'season_2', 'season_3', 'season_4']].idxmax(axis=1)
df['season'] = df['season'].map({'season_1': 'spring', 'season_2': 'summer', 'season_3': 'fall', 'season_4': 'winter'})
plt.figure(figsize=(12,6))
sns.boxplot(x='season', y='cnt', data=df)
plt.title('Distribution of bike rentals V/S seasons')
plt.show()


# In[45]:


plt.figure(figsize=(12,6))
sns.boxplot(x='weathersit', y='cnt', data=df)
plt.title('Relationship between weather conditions V/S bike rentals')
plt.show()


# In[46]:


plt.figure(figsize=(20,6))
sns.boxplot(x='temp', y='cnt', data=df)
plt.title('Relationship between temperature conditions V/S bike rentals')
plt.show()


# In[47]:


plt.figure(figsize=(20,6))
sns.boxplot(x='atemp', y='cnt', data=df)
plt.title('Relationship between feeling temperature conditions V/S bike rentals')
plt.show()


# In[48]:


# Define a function to determine if a given hour is typically a rush hour
def is_rush_hour(hour):
    if (7 <= hour <= 9) or (17 <= hour <= 19):
        return 1
    else:
        return 0

# Apply the function to the 'hr' column to create the new 'rush_hour' feature
df['rush_hour'] = df['hr'].apply(is_rush_hour)

# Display the first few rows of the dataframe to verify the changes
df.head().style.background_gradient(cmap='Greys')


# ## Hyperparameter Experiment

# In[49]:


#creating dimensions for Modelling
x_data = df.drop(['cnt', 'dteday', 'season'], axis=1)
y_data = df['cnt']


# In[50]:


#Spliting test and train data with standard ratio
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=1)
print("shape of x_train",x_train.shape)
print("shape of y_train",y_train.shape)
print("shape of x_test",x_test.shape)
print("shape of y_test",y_test.shape)


# In[51]:


#creating object, using which we could generate instances.
lm = LinearRegression() 


# In[52]:


#Fitting model by supplying train data
lm.fit(x_train,y_train)


# In[53]:


yhat= lm.predict(x_test)
print("Predicted cnt for test data are:", yhat[0:5].tolist())


# In[54]:


plt.title("Actual cnt v/s Predicted cnt)")
ax1 = sns.distplot(y_data,hist=False,color="green",label="Actual Value")
sns.distplot(yhat,hist=False,color='red',label="Fitted Value",ax = ax1)
plt.legend(title='legend', loc='upper right', labels=['Actual Value', 'Predicted Value'])
plt.show()


# In[55]:


#Mean Squared Error(MSE)
mse = mean_squared_error(y_test, yhat)
print('The mean square error of cnt and predicted value is: ', mse)


# In[56]:


#R-Squared
print('the R-Squared value of fitted model is:',lm.score(x_train,y_train))


# In[ ]:





# ## Decision
# 
# 
# 

# Regression is the preferred method for predictive modelling because to its interpretability, compatibility with the features of the dataset, and use of well-established performance evaluation techniques. It offers a strong basis for comprehending and forecasting bike rental numbers.
# Continuous goal Variable: Since "cnt" (bike rental count) is a continuous goal variable, regression is deemed appropriate for forecasting numerical results.
# 
# Relationship between Predictors: The dataset contains a number of variables that may affect the number of bike rentals, such as the season, time of day, and weather. The links between these predictors and the target variable are captured using regression.
# Interpretability: Regression models provide interpretability, which helps stakeholders by enabling us to comprehend how various factors influence bike rentals.
# Baseline Model: Prior to examining more sophisticated methods, regression functions as the fundamental baseline model for predicting tasks.
# Regression Performance measures: For efficient model evaluation, regression models have well-established performance measures including Mean Squared Error (MSE) and R-squared (R2).
# 
# 
# OUTPUT:
# 
# Peak Business Hours: From 7 to 9 AM to 4 to 7 PM is when business is at its busiest. The importance of these hours to the company and the possible chances for improved services might be seized.
# Weekday Analysis: There are almost similar numbers of rentals seen on weekdays (280). Possibility of addressing distinct client segments with various techniques during weekdays.
# Holiday Rentals: During the holidays, the cost of rentals is generally higher. Suggested for marketing campaigns aimed at leveraging the desire for holidays.
# 
# Weather Impact: Distil the number of rentals according to the weather (Weathersit):
# The weather has a significant impact on user behaviour as it suggests potential modifications to business operations.
# There are almost 290 rentals when it's clear, partly cloudy, and little clouds.
# Mist has about 250 rentals. Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds
# 150 rentals are available for Light Snow, Light Rain + Thunderstorm + Scattered Clouds, and Light Rain + Scattered Clouds.
# When there is intense rain, ice pallets, a thunderstorm, mist, snow, and fog There will be 110 rentals.
# 
# As temperatures rise, the number of rentals increases.
# As would be expected, the rental count is inversely related to the humidity. Thus, operational and business strategy could be used to maximise total performance.
# 
# 
# 
# 
# 

# 
