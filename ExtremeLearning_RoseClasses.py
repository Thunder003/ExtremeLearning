#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
train = pd.read_csv("Fisheriris data.csv")


# In[106]:


train.head()


# In[107]:


df=pd.DataFrame(train)


# In[108]:


df.head()


# In[109]:


df.loc[df["LABEL"]=="'setosa'","LABEL"]=1
df.loc[df["LABEL"]=="'versicolor'","LABEL"]=2
df.loc[df["LABEL"]=="'virginica'","LABEL"]=3


# In[110]:


df.LABEL.dtype


# In[111]:


df.FA.dtype


# In[112]:


a=df.sample(frac=1).reset_index(drop=True)
  


# In[113]:


x_train = a.iloc[:, 0:4]
x_train.head()


# In[114]:


label=a.iloc[:,4]
label.head()


# In[115]:


label.shape[0]


# In[116]:


classes=3
y_train=np.zeros([label.shape[0],classes])
for i in range(label.shape[0]):
    y_train[i][label[i]-1]=1
  


# In[117]:


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)


# In[118]:


x_train.shape


# In[119]:


y_test.shape


# In[120]:


input_length=x_train.shape[1]
hidden_unit=100

win=np.random.normal(input_length,hidden_unit)


# In[121]:


def fun(x):
    q=np.dot(x,win)
    ac=np.maximum(q,0,q)
    return ac


# In[122]:


X=fun(x_train)
xt=np.transpose(X)
wout=np.dot(np.linalg.inv(np.dot(xt,X)),np.dot(xt,y_train))
print(wout.shape)


# In[123]:


def predict(x):
    x=fun(x)
    y=np.dot(x,wout)
    return y
    


# In[124]:


y=predict(x_test)
print(y.shape)
correct=0
total=y.shape[0]
for i in range(total):
    predicted=np.argmax(y[i])
    test=np.argmax(y_test[i])
    correct=correct+(1 if predicted==test else 0)
print("Accuracy ={:f}".format(correct/total))    

