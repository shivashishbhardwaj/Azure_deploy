#!/usr/bin/env python
# coding: utf-8

# ## Fare Price Prediction

# **Problem Statement**

# We have been provided a dataset of a cab facility providing company, which is basically providing or doing the business of earning money from providing customers the cabs to travel from their position to destination in a particular city. Given various independent features our task is to predict the fare amount of the rides by each customer every time.
# 
# So the problem is of regression only as we have to predict the continuous target variable. We are going to use some regression techniques to predict the fare price. Also I will be predicting the fare amount using PySpark.

# Fare Amount Prediction
# 
# The dataset “trips.csv” contains the following fields:
# 
# 
# **key** - a unique identifier for each trip
# 
# 
# **fare_amount** - the cost of each trip in usd
# 
# 
# **pickup_datetime** - date and time when the meter was engaged
# 
# 
# **passenger_count** - the number of passengers in the vehicle (driver entered value)
# 
# **pickup_longitude** - the longitude where the meter was engaged
# 
# **pickup_latitude** - the latitude where the meter was engaged
# 
# **dropoff_longitude** - the longitude where the meter was disengaged
# 
# **dropoff_latitude** - the latitude where the meter was disengaged
# 
# – We need to analyse the data and create an efficient model that will estimate the fare prices accurately. 
# 

# **Importing Required libraries**

# In[1]:


#Importing required libraries
import os #getting access to input files
import pandas as pd # Importing pandas for performing EDA
import numpy as np  # Importing numpy for Linear Algebric operations
import matplotlib.pyplot as plt # Importing for Data Visualization
import seaborn as sns # Importing for Data Visualization
from collections import Counter 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split #splitting dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import GridSearchCV    

get_ipython().run_line_magic('matplotlib', 'inline')


# **loading and reading the dataset**

# In[2]:


# Reading and viewing the csv file.

df = pd.read_csv('fare_price.csv')
df.head()


# ### checking the number of rows and columns

# In[3]:


df.shape


# In[4]:


df.columns


# ##### checking the data-types in dataset

# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.describe()


# #Data Cleaning & Missing Value Analysis :

# In[8]:


df.dropna(subset= ["pickup_datetime"])   #dropping NA values in datetime column


# # Here pickup_datetime variable is in object so we need to change its data type to datetime

# In[9]:


df['pickup_datetime'] =  pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')


# ### we will saperate the Pickup_datetime column into separate field like year, month, day of the week, etc

# In[10]:


df['year'] = df['pickup_datetime'].dt.year
df['Month'] = df['pickup_datetime'].dt.month
df['Date'] = df['pickup_datetime'].dt.day
df['Day'] = df['pickup_datetime'].dt.dayofweek
df['Hour'] = df['pickup_datetime'].dt.hour
df['Minute'] = df['pickup_datetime'].dt.minute


# In[11]:


#Re-checking datatypes after conversion
df.dtypes 


# #Missing Values treatment

# In[12]:


#removing datetime missing values rows
df = df.drop(df[df['pickup_datetime'].isnull()].index, axis=0)
print(df.shape)
print(df['pickup_datetime'].isnull().sum())


# ### Checking the passenger count variable :

# In[13]:


df["passenger_count"].describe()


# We can see maximum number of passanger count is 208 which is actually not possible. So reducing the passenger count to 6 (even if we consider the SUV)

# In[14]:


df = df.drop(df[df["passenger_count"]> 6 ].index, axis=0)


# In[15]:


df["passenger_count"].describe()


# In[16]:


df["passenger_count"].sort_values(ascending= True)


# There are passengers with count value of 0 which is not required. Hence we will remove 0 passenger values.
# 
# 
# 

# In[17]:


df = df.drop(df[df["passenger_count"] == 0 ].index, axis=0)
df.shape


# Next checking the Fare Amount variable :

# In[18]:


##finding decending order of fare to get to know whether the outliers are present or not
df["fare_amount"].sort_values(ascending=False)


# In[19]:


Counter(df["fare_amount"]<0)


# In[20]:


df = df.drop(df[df["fare_amount"]<0].index, axis=0)
df.shape


# In[21]:


#making sure that there is no negative values in the fare_amount variable column
df["fare_amount"].min()


# In[22]:


#Also remove the row where fare amount is zero
df = df.drop(df[df["fare_amount"]<1].index, axis=0)
df.shape


# In[23]:


df['fare_amount'].isnull().sum()


# In[24]:


df["fare_amount"].describe()


# **Now checking the pickup lattitude and longitude :**

# In[25]:


#Lattitude----(-90 to 90)
#Longitude----(-180 to 180)

# we need to drop the rows having  pickup lattitute and longitute out the range mentioned above

df[df['pickup_latitude']<-90]
df[df['pickup_latitude']>90]


# In[26]:


#Hence dropping one value of >90
df = df.drop((df[df['pickup_latitude']<-90]).index, axis=0)
df = df.drop((df[df['pickup_latitude']>90]).index, axis=0)


# In[27]:


df[df['pickup_longitude']<-180]
df[df['pickup_longitude']>180]


# In[28]:


df.shape


# In[29]:


df.isnull().sum()


# ### Now we have cleaned our datasets. Thus proceeding for further operations:
# 
# **Calculating distance based on the given coordinates :**

# In[30]:


#We have given pickup longitute and latitude values and same for drop. 
#So we need to calculate the distance Using the haversine formula and we will create a new variable called distance
from math import radians, cos, sin, asin, sqrt

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km
# 1min 


# In[31]:


df['distance'] = df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[32]:


df.head()


# In[33]:


df.nunique()


# In[34]:


##finding decending order of fare to get to know whether the outliers are presented or not
df['distance'].sort_values(ascending=False)


# As we can see that top few values in the distance variables are very high. It means that more than 8000 Kms distance they have travelled Also just after those values from the top, the distance goes down to 127, which means these values are showing some outliers We need to remove these values

# In[35]:



Counter(df['distance'] == 0)


# In[36]:


Counter(df['fare_amount'] == 0)


# In[37]:


###we will remove the rows whose distance value is zero

df = df.drop(df[df['distance']== 0].index, axis=0)
df.shape


# In[38]:


#we will remove the rows whose distance values is very high which is more than 129kms
df = df.drop(df[df['distance'] > 130 ].index, axis=0)
df.shape


# In[39]:


df.head()


# Now we have splitted the pickup date time variable into different varaibles like month, year, day etc so now we dont need to have that pickup_Date variable now. Hence we can drop that, Also we have created distance using pickup and drop longitudes and latitudes so we will also drop pickup and drop longitudes and latitudes variables.

# In[40]:


drop = ['key','pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']
df = df.drop(drop, axis = 1)


# In[41]:


df.head()


# In[42]:


df.info()


# In[43]:


df.dtypes


# In[44]:


df = df.drop('index', 1)


# In[45]:


df.head()


# In[46]:


dff = df.copy()


# #Data Visualization :
# **Visualization the following:**
# 
# 1. Number of Passengers effects the the fare
# 
# 2. Pickup date and time effects the fare
# 
# 3. Day of the week does effects the fare
# 
# 4. Distance effects the fare

# In[47]:


# Count plot on passenger count
plt.figure(figsize=(15,7))
sns.countplot(x="passenger_count", data=df)


# In[48]:


#Relationship beetween number of passengers and Fare

plt.figure(figsize=(15,7))
plt.scatter(x=df['passenger_count'], y=df['fare_amount'], s=10)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.show()


# **Observations :**
# 
# By seeing the above plots we can easily conclude that:
# 
# 1. single travelling passengers are most frequent travellers.
# 2. At the sametime we can also conclude that highest Fare are coming from single & double travelling passengers.

# In[49]:


#Relationship between date and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=df['Date'], y=df['fare_amount'], s=10)
plt.xlabel('Date')
plt.ylabel('Fare')
plt.show()


# In[50]:


plt.figure(figsize=(15,7))
df.groupby(df["Hour"])['Hour'].count().plot(kind="bar")
plt.show()


# Lowest cabs at **5 AM** and highest at and around **7 PM** i.e the office rush hours.

# In[51]:


#Relationship between Time and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=df['Hour'], y=df['fare_amount'], s=10)
plt.xlabel('Hour')
plt.ylabel('Fare')
plt.show()


# From the above plot We can observe that the cabs taken at 7 am and 23 Pm are the costliest. Hence we can assume that cabs taken early in morning and late at night are costliest

# In[52]:


#impact of Day on the number of cab rides
plt.figure(figsize=(15,7))
sns.countplot(x="Day", data=df)


# **Observation :** The day of the week does not seem to have much influence on the number of cabs ride

# In[53]:


#Relationships between day and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=df['Day'], y=df['fare_amount'], s=10)
plt.xlabel('Day')
plt.ylabel('Fare')
plt.show()


# The highest fares seem to be on a Sunday, Monday and Thursday, and the low on Wednesday and Saturday. May be due to low demand of the cabs on saturdays the cab fare is low and high demand of cabs on sunday and monday shows the high fare prices

# In[54]:


#Relationship between distance and fare 
plt.figure(figsize=(15,7))
plt.scatter(x = df['distance'],y = df['fare_amount'],c = "g")
plt.xlabel('Distance')
plt.ylabel('Fare')
plt.show()


# It is quite obvious that distance will effect the amount of fare

# ### Feature Scaling :

# In[55]:


#Normality check of training data is uniformly distributed or not-

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(df[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# In[56]:


#since skewness of target variable is high, apply log transform to reduce the skewness-
df['fare_amount'] = np.log1p(df['fare_amount'])

#since skewness of distance variable is high, apply log transform to reduce the skewness-
df['distance'] = np.log1p(df['distance'])


# In[57]:


#Normality Re-check to check data is uniformly distributed or not after log transformartion

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(df[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# Here we can see bell shaped distribution. Hence our continous variables are now normally distributed, we will use not use any Feature Scalling technique. i.e, Normalization or Standarization for our training data

# ## Applying ML ALgorithms:

# **Train test split and defining model parameters**

# In[58]:


##train test split for further modelling
X_train, X_test, y_train, y_test = train_test_split( dff.iloc[:, df.columns != 'fare_amount'], 
                         dff.iloc[:, 0], test_size = 0.20, random_state = 1)


# In[59]:


print(X_train.shape)
print(X_test.shape)


# In[60]:


def train_model(model,X_train,y_train,X_test,y_test):
  
  model.fit(X_train,y_train)
  pred_value=model.predict(X_test)
  MSE=mean_squared_error(y_test,pred_value)
  RMSE=np.sqrt(MSE)
  r2=r2_score(y_test,pred_value)
  adj_r2=1-(1-r2_score(y_test,pred_value))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))
  print(f'Evaluation matrix-- \n MSE: {MSE}\n RMSE: {RMSE}\n r2Score: {r2}\n adj_r2: {adj_r2}\n')
  print('Evaluation Graph')
  plt.figure(figsize=(10,5))
  p1=plt.plot(pred_value[:100])
  p2=plt.plot(np.array(y_test[:100]))
  plt.legend(["ACTUAL","PREDICTED"],prop={'size': 10})
  plt.show()


# In[61]:


#Implementing linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

train_model(reg,X_train,y_train,X_test,y_test)


# **Using Random Forest Model :**

# In[62]:


rf_model = RandomForestRegressor()
n_estimators=[60,80,100]
max_depth=[15,20]
max_leaf_nodes=[40,60,80]
params = {'n_estimators':n_estimators,'max_depth':max_depth ,'max_leaf_nodes':max_leaf_nodes}
rf_grid= GridSearchCV(rf_model,param_grid=params,verbose=0)
train_model(rf_grid,X_train,y_train,X_test,y_test)


# **Prediction of fare from provided test dataset :**

# We have already cleaned and processed our test dataset along with our training dataset. Hence we will be predicting using grid search CV for random forest model

# In[63]:


model = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=10, bootstrap=True)
model.fit(X_train, y_train)
predict = model.predict(X_test)
print(predict)
print(predict.shape)


# In[64]:


import pickle


# In[65]:


with open ('model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[66]:


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# In[67]:


prediction = model.predict(X_test)


# In[68]:


print(prediction)


# In[ ]:




