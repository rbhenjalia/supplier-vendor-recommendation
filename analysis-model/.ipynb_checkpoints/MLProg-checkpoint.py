#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from ast import literal_eval
from collections import defaultdict
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import MinMaxScaler


# In[3]:


tr = pd.read_csv('D:/finData.csv')


# In[4]:


tr.head()


# In[5]:


tr = tr[['Product_Seller', 'Price', 'Rate']]


# In[6]:


tr.head()


# In[7]:


tab1 = pd.read_csv('D:/FinishedDataSet_1.csv')
tab2 = pd.read_csv('D:/FinishedDataSet_2.csv')
tab3 = tr


tab1 = tab1[['Product_Seller', 'Price', 'Rating']]
tab2 = tab2[['Product_Seller', 'Price', 'Rating']]

tab1['Product'], tab1['Seller'] = tab1['Product_Seller'].str.split(' ##### ', 1).str
tab1 = tab1[['Product', 'Seller','Price', 'Rating']]


tab2['Product'], tab2['Seller'] = tab2['Product_Seller'].str.split(' ##### ', 1).str
tab2 = tab2[['Product', 'Seller','Price', 'Rating']]


# In[8]:


tab3 = tab3.rename(columns={'Rate':'Rating'})
tab3['Product'], tab3['Seller'] = tab3['Product_Seller'].str.split(' - ', 1).str
tab3 = tab3[['Product', 'Seller','Price', 'Rating']]


# In[9]:


finData = pd.concat([tab1,tab2,tab3]).reset_index(drop=True)


# In[43]:


finData.head()


# In[10]:


finData.describe()

finData = pd.read_csv('D:/erp.csv')
finData['Rating'] = finData['Rating'].astype(float)

finData['Price'] = finData['Price'].astype(float)


# In[11]:


finData = finData[['Product','Seller', 'Price', 'Rating']]


# In[16]:


finData.head()
finData['Rating'].fillna(value=2.5)
finData.to_csv('D:/Data.csv')


# In[17]:


sns.distplot(finData['Price'],kde=False,bins=70)


# In[18]:


sns.jointplot(x='Price',y='Rating',data=finData)


# In[19]:


sns.distplot(finData['Rating'],kde=False,bins=10)


# In[20]:


sns.pairplot(finData)


# In[21]:


ratings = finData.pivot_table(index='Product',columns='Seller',values='Rating')
price = finData.pivot_table(index='Product',columns='Seller',values='Price')


# In[22]:


X_train = finData[['Product', 'Price', 'Rating']]


# In[23]:


Y_train = finData['Seller']
E = 'Acer Aspire 3 UN.GNVSI.001 15.6-inch Laptop (AMD Dual-Core Processor E2-9000/4GB/1TB/windows 10 Home 64Bit/Integrated Graphics), Obsidian Black'


# In[49]:


#LET's ASSUME THAT THE GIVEN PRODUCT IS SEARCHED & CLICKED UPON
Product = 'Acer Aspire 3 UN.GNVSI.001 15.6-inch Laptop (AMD Dual-Core Processor E2-9000/4GB/1TB/windows 10 Home 64Bit/Integrated Graphics), Obsidian Black'

#LOCATING THE DATA WITH SEARCHED PRODUCT
df = finData.loc[finData['Product'] == Product]


# In[25]:


#FILTERING ONLY SELLER, PRICE & RATING
df = df[['Seller', 'Price', 'Rating']]

#SINCE THERE ARE MULTIPLE RATINGS ON A PARTICULAR SELLER FOR A PARTICULAR PRODUCT
#NUMBER OF RATINGS RECEIVED BY EACH SELLER ON THE PRODUCT
avg_rating_count = pd.DataFrame(df.groupby('Seller')['Rating'].count().sort_values(ascending=False)).reset_index()
avg_rating_count


# In[26]:


#AVERAGE RATING RECEIVED BY EACH SELLER ON THE PRODUCT
avg_rating = pd.DataFrame(df.groupby('Seller')['Rating'].mean().sort_values(ascending=False)).reset_index()
avg_rating


# In[27]:


#AVERAGE OF PRICE PROVIDED BY EACH SELLER ON THE PRODUCT
avg_price = pd.DataFrame(df.groupby('Seller')['Price'].mean().sort_values()).reset_index()
avg_price


# In[28]:


#CREATING A NEW FRAME BASED ON SELLERS
new_Frame = pd.merge(avg_rating,avg_price,on='Seller', how='outer')
new_Frame = pd.merge(new_Frame,avg_rating_count,on='Seller', how='outer')
new_Frame = new_Frame.rename(columns={'Rating_y':'Count','Rating_x':'Rating'})
new_Frame


# In[29]:


corr = new_Frame.corr()
corr


# In[30]:


sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[31]:


#SCALING DATA TO SCORE(0-5) 
from sklearn.preprocessing import MinMaxScaler
data = new_Frame[['Price','Rating','Count']]
scaler = MinMaxScaler(feature_range=(0, 5))
scaler.fit(data)


# In[32]:


df = pd.DataFrame()


# In[44]:


df = pd.DataFrame(scaler.transform(data))

#LOWER THE PRICE HIGHER THE SCORE, HENCE REVERSE SCALING
df[0] = 5 - df[0]
df


# In[34]:


#CALCULATING TOTAL SCORE
df['result'] = df[0] + df[1] + df[2]


# In[35]:


new_Frame = new_Frame.join(df['result'])


# In[36]:


new_Frame


# In[37]:


#RETURNING THE SELLER WITH HIGHEST SCORE
pdr = new_Frame[new_Frame['result'] == new_Frame['result'].max()]
x = pdr.iat[0,0]
x


# In[38]:


#A FUNCTION WHICH TAKES PRODUCT NAME AND FINISHED DATASET TO RETURN A BEST SELLER FOR THE PRODUCT

def Best_Seller(product,dataSet):
    df = finData.loc[finData['Product'] == product]
    df = df[['Seller', 'Price', 'Rating']]
    
    avg_rating_count = pd.DataFrame(df.groupby('Seller')['Rating'].count().reset_index())
    avg_rating = pd.DataFrame(df.groupby('Seller')['Rating'].mean().reset_index())
    avg_price = pd.DataFrame(df.groupby('Seller')['Price'].mean().reset_index())
    
    new_Frame = pd.merge(avg_rating,avg_price,on='Seller', how='outer')
    new_Frame = pd.merge(new_Frame,avg_rating_count,on='Seller', how='outer')
    new_Frame = new_Frame.rename(columns={'Rating_y':'Count','Rating_x':'Rating'})
      
    data = new_Frame[['Price','Rating','Count']]
    scaler = MinMaxScaler(feature_range=(0, 5))
    scaler.fit(data)

    df = pd.DataFrame()     
    df = pd.DataFrame(scaler.transform(data))
                             
    df[0] = 5 - df[0]
    df['result'] = df[0] + df[1] + df[2]
    pdr = new_Frame[new_Frame['result'] == new_Frame['result'].max()]
    x = pdr.iat[0,0]
    return x


# In[46]:


pdf = finData.groupby('Product')['Seller'].nunique().count()


# In[47]:


pdf


# In[ ]:





# In[ ]:





# In[ ]:




