# irisData
 Predict the class of the flower based on available attributes.
This is my first attempt to put my codes on git.


# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[ ]:


iris = pd.read_csv('Iris.csv')


# In[ ]:


iris.describe()


# In[ ]:


iris.head()


# In[ ]:


iris.groupby('Species').size()


# In[ ]:


iris.plot(kind='box', subplots=True, figsize=(12,12), layout=(5,5),sharex=False,sharey=False)


# In[ ]:


iris.hist(figsize=(12,12))


# In[ ]:


sns.pairplot(iris)


# In[ ]:


X = iris[['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lmModel = LogisticRegression()


# In[ ]:


predict_var= [ 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
outcome= ['Species']


# In[ ]:


lmModel.fit(X_train,y_train)


# In[ ]:


prediction= lmModel.predict(X_test)
prediction


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,prediction))

