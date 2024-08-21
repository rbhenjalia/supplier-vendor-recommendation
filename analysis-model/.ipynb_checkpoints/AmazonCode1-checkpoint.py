#!/usr/bin/env python
# coding: utf-8

# In[317]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from ast import literal_eval
from collections import defaultdict
get_ipython().run_line_magic('matplotlib', 'inline')


# In[318]:


t1 = pd.read_csv('D:/amazon_1.csv')
t2 = pd.read_csv('D:/amazon_2.csv')
t3 = pd.read_csv('D:/amazon_3.csv')
t4 = pd.read_csv('D:/amazon_4.csv')
t5 = pd.read_csv('D:/amazon_5.csv')
t6 = pd.read_csv('D:/amazon_6.csv')
t7 = pd.read_csv('D:/amazon_7.csv')
t8 = pd.read_csv('D:/amazon_8.csv')
t9 = pd.read_csv('D:/amazon_9.csv')
t10 = pd.read_csv('D:/amazon_10.csv')


t2.columns


# In[319]:


t1 = t1[['Product_Name','Seller_Name','Seller_Price','Star_Rating']]
t2 = t2[['Product_Name','Seller_Name','Seller_Price','Star_Rating']]
t3 = t3[['Product_Name','Seller_Name','Seller_Price','Star_Rating']]


# In[320]:


finSet = pd.concat([t1,t2,t3], axis=0)


# In[321]:


finSet.columns


# In[322]:


finSet.shape


# In[326]:


sns.heatmap(finSet.isnull(),yticklabels=False,cbar=False)


# In[327]:


df = finSet[pd.notnull(finSet['Seller_Price'])]
df = df[pd.notnull(df['Seller_Name'])]


# In[328]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[299]:


df.head()
fdata = df
df.columns


# In[300]:


rate = fdata['Star_Rating']
price = fdata['Seller_Price']
x = rate.values
y = price.values
pr = pd.Series(x).str.replace(' out of 5 stars', '', regex=True)
fdata['Rate'] = pr
pr2 = pd.Series(y).str.replace('Rs. ', '', regex=True)
pr3 = pd.Series(pr2).str.replace(',', '', regex=True)
fdata['Price'] = pr3


# In[301]:


fdata['Product_Seller'] = fdata['Product_Name'] + ' ##### ' + fdata['Seller_Name']


# In[302]:


fdata.head()


# In[303]:


fdata = fdata[['Product_Seller','Price','Rate']]


# In[304]:


fdata.head()


# In[ ]:





# In[305]:


col = 'Rate'

fdata[col] = fdata[col].astype(float)


# In[306]:


fdata.dtypes


# In[307]:


fdata.head()


# In[308]:


fdata["Rating"] = fdata.groupby("Product_Seller").transform(lambda x: x.fillna(x.mean()))


# In[309]:


fdata.shape


# In[315]:


sns.heatmap(fdata.isnull(),yticklabels=False,cbar=False)


# In[311]:


fdata = fdata[['Product_Seller','Price','Rating']]


# In[312]:


fdata.fillna(value=2.5,inplace=True)


# In[313]:


fdata.shape


# In[316]:


fdata.to_csv('D:/FinishedDataSet_1.csv')


# In[ ]:




