#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string 
import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', 100) 
pd.set_option('display.max_rows', 10) 
pd.set_option('display.width', 1000) 
from itertools import cycle 


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn
import datetime
####
# scikit-learn version is 0.24.1.
# pandas version is 1.2.3.
# numpy version is 1.20.1.


# In[3]:


df=pd.read_csv(r'D:\учеба\Kaggle\Биржа\US1.AAPL_210501_210901.csv' ,sep=',')


# In[4]:


#фичи delta1 - разница <CLOSE> с предыдущим временным промежутком,
# delta2 - с точкой двумя промежутками раньше
delta1=[0 for i in range(0,df["<TICKER>"].size)]
delta2=[0 for i in range(0,df["<TICKER>"].size)]
i=1
while (i<df["<TICKER>"].size):
    delta1[i]=df["<CLOSE>"][i]-df["<CLOSE>"][i-1]
    i+=1
df['delta1']=delta1
delta1=[0 for i in range(0,df["<TICKER>"].size)]

i=2
while (i<df["<TICKER>"].size):
    delta2[i]=df["<CLOSE>"][i]-df["<CLOSE>"][i-2]
    i+=1
df['delta2']=delta2
df


# In[5]:


#добавляю фичу "день недели" + форматирую дату с дд\мм\гг на дд\мм\гггг
day=[0 for i in range(0,df["<TICKER>"].size)]
from datetime import datetime, date, time
i=0
while (i<df["<TICKER>"].size):
    df["<DATE>"][i]=df["<DATE>"][i][0:6]+"20"+df["<DATE>"][i][6:8]
    day[i]=datetime.weekday(datetime.strptime(df["<DATE>"][i]+" "+df["<TIME>"][i], 
                             "%d/%m/%Y %H:%M"))
    i+=1
df['day']=day


# In[6]:


#добавляю фичу "время" - в секундах с начала торгов (16:30)
size=len(df["<TICKER>"])
hours=[0 for i in range(size)]
mins=[0 for i in range(size)]
secs=[0 for i in range(size)]
for i in range(size):
    hours[i]=int(df["<TIME>"][i][0:2])
    mins[i]=int(df["<TIME>"][i][3:5])
TIME=[0 for i in range(0,df["<TICKER>"].size)]
for i in range(size):
    TIME[i]=(hours[i]-16)*3600+mins[i]*60-1860
    
df['time']=TIME


# In[7]:


#метки для обучения (значения <CLOSE> на следующем временном промежутке)
labels=[float(0.0) for i in range(0,df["<TICKER>"].size)]
labels=np.float32(labels)
df['label']=labels
for i in range(size-1):
    df['label'][i]=(df['<CLOSE>'][i+1])


# In[8]:


#формирование обучающей, тестовой выборок
train={
    'time':df['time'][0:size-1],
    '<CLOSE>':df['<CLOSE>'][0:size-1],
      'delta1':df['delta1'][0:size-1],
    'delta2':df['delta2'][0:size-1],
'day':df['day'][0:size-5]}
df_train=pd.DataFrame(data=train)
X = df_train

y=df['label'][0:size-1]

from sklearn.model_selection import train_test_split


# In[9]:


y=df['label'][0:size-1]


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =0)


# In[11]:


X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())


# In[19]:


X_train


# In[12]:


y_test=y_test.reset_index(drop=True)


# In[13]:


# обучение
count=5
mean_square_err=[0]*count
deviation=[0]*count
mae=[0]*count
for depth in range(count):
    regr = RandomForestRegressor(max_depth=5+depth, random_state=0,min_samples_leaf=5,criterion='mse')
    regr.fit(X_train,y_train)
    y_pred=regr.predict(X_test)
    for i in range(y_pred.size):
        mean_square_err[depth]+=(y_test[i]-y_pred[i])**2
    mean_square_err[depth]/=y_pred.size
    mean_square_err[depth]=mean_square_err[depth]**0.5
    for i in range(y_pred.size):
        deviation[depth]+=abs(y_test[i]-y_pred[i])*100/y_test[i]
    deviation[depth]/=y_pred.size
    for i in range(y_pred.size):
        mae[depth]+=abs(y_test[i]-y_pred[i])
    mae[depth]/=y_pred.size


# In[14]:


err=pd.DataFrame(data={'mse':mean_square_err,'mae':mae,'dev':deviation})


# In[15]:


err


# In[16]:


# итоговая модель:
model = RandomForestRegressor(max_depth=8, random_state=0,min_samples_leaf=5)
model.fit(X_train,y_train)

