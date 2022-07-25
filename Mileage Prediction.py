#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df=pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/MPG.csv')


# In[10]:


df.head()


# In[11]:


df.nunique()


# # Data Preprocessing

# In[12]:


df.info()


# In[13]:


df.corr()


# # Removing missing values

# In[14]:


df=df.dropna()


# In[15]:


df.info()


# # Data visualization

# In[17]:


sns.pairplot(df,x_vars=['displacement','horsepower','weight','acceleration','mpg'],y_vars=['mpg'])


# In[18]:


sns.regplot(x='displacement',y='mpg',data=df)


# # Define Target variable y and features x

# In[19]:


df.columns


# In[20]:


y=df['mpg']


# In[21]:


y.shape


# In[22]:


y


# In[23]:


x=df[[ 'displacement', 'horsepower', 'weight',
       'acceleration']]


# In[24]:


x.shape


# In[25]:


x


# # Scaling

# In[26]:


from sklearn.preprocessing import StandardScaler


# In[27]:


ss=StandardScaler()


# In[28]:


x=ss.fit_transform(x)


# In[29]:


x


# # Train Test split data 

# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=2529)


# In[32]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# # Linear Regression

# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


lr=LinearRegression()


# In[36]:


lr.fit(x_train,y_train)


# In[37]:


lr.intercept_


# In[38]:


lr.coef_


# # Predict Test Data

# In[39]:


y_pred=lr.predict(x_test)


# In[40]:


y_pred


# # Model Accuracy

# In[45]:


from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score


# In[46]:


mean_absolute_error(y_test,y_pred)


# In[47]:


mean_absolute_percentage_error(y_test,y_pred)


# In[48]:


r2_score(y_test,y_pred)


# # Polynomial Regression

# In[53]:


from sklearn.preprocessing import PolynomialFeatures


# In[57]:


poly=PolynomialFeatures(degree=2,interaction_only=True,include_bias= False,order='C')


# In[58]:


x_train2= poly.fit_transform(x_train)


# In[59]:


x_test2=poly.fit_transform(x_test)


# In[60]:


lr.fit(x_train2,y_train)


# In[61]:


lr.intercept_


# In[63]:


lr.coef_


# In[64]:


y_pred_poly=lr.predict(x_test2)


# # Model Accuracy

# In[65]:


from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score


# In[66]:


mean_absolute_error(y_test,y_pred_poly)


# In[67]:


mean_absolute_percentage_error(y_test,y_pred_poly)


# In[68]:


r2_score(y_test,y_pred_poly)


# In[ ]:




