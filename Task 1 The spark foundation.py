#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation

# # Data Science and Business Analytics Internship- Grip July'21

# # Task1: Prediction Using Supervised ML

# ## Author:  Drishti Baranwal 

# # ( Level - Beginner)

# ### Probelm statement 

# ### Predict the percentage of a student based on the number of study hours.

# # 1. Import all the required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Loading the data

# In[7]:


df= pd.read_csv("http://bit.ly/w-data")
print("Data imported successfully")


# # 3. Printing the first 10 lines 

# In[9]:


df.head(10)


# # 4. Basic Data Exploration

# In[10]:


df.shape


# In[12]:


df.info()


# In[13]:


df.describe()


# # 5. Checking for null values 

# In[14]:


df.isnull== True


# # 6. Data Visualization

# In[25]:


df.plot(x='Hours', y='Scores',style='*', markerfacecolor='blue')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ### From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# # 7. Preparing the data

# ### The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[27]:


x= df.iloc[:, :-1].values  
y = df.iloc[:, 1].values


# In[32]:


x


# In[33]:


y


# # 8. Train the model by spliting dataset into training & testing set

# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state=0)


# In[38]:


# Fitting the model on training dataset
from sklearn.linear_model import LinearRegression

model= LinearRegression()
model.fit(X_train, y_train)


# # 9.Printing intercept and coefficient 

# In[43]:


model.intercept_


# In[44]:


model.coef_


# # 10. Plotting the regression line for the  test data 

# In[46]:


# Plotting the regression line
regg_line = model.coef_*X+model.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, regg_line);
plt.show()


# # 11. Making Predictions

# In[48]:


# We are printing testing data in hours 
print(X_test)


# In[50]:


# Predicting the scores 
y_pred= model.predict(X_test)


# In[51]:


y_pred


# In[52]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# # 12. Model Evaluation 

# In[54]:


from sklearn import metrics 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# # 13. Testing with your own data 

# In[66]:


hours = 9.25
predicted_score= model.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(predicted_score[0]))

