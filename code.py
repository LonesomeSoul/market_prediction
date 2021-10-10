#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string 
import math
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


df=pd.read_csv(r'D:\учеба\Kaggle\Биржа\US1.AAPL_200501_210901.csv' ,sep=',')


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


# In[ ]:





# In[ ]:





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


df_train.values


# In[10]:


y=df['label'][0:size-1]


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =0)


# In[12]:


X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())


# In[13]:


X_train


# In[ ]:





# In[14]:


y_test=y_test.reset_index(drop=True)


# In[ ]:





# In[15]:


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


# In[16]:


y_pred=regr.predict(X_test)
X_test, y_pred


# In[17]:


err=pd.DataFrame(data={'mse':mean_square_err,'mae':mae,'dev':deviation})


# In[18]:


err


# In[19]:


# итоговая модель:
model = RandomForestRegressor(max_depth=8, random_state=0,min_samples_leaf=5)
model.fit(X_train,y_train)


# In[20]:


model.feature_importances_


# In[21]:


#Модель нейронной сети PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

min_max_scaler = sklearn.preprocessing.MinMaxScaler()

train_data=(torch.tensor(min_max_scaler.fit_transform (X_train.values)))
train_target=(torch.tensor( (y_train.values)))
test_data=(torch.tensor(min_max_scaler.fit_transform (X_test.values)))
test_target=(torch.tensor( (y_test.values)))


# In[36]:


class Net(nn.Module):
    def __init__(self,drop):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(drop)
    def forward(self, x):
        #x = F.logsigmoid(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# In[38]:


#обучение модели и возврат mae

def checkloss(drop=0.1,learning_rate=0.01, batch_size=30):
    epochs=10
    train_batches=math.ceil(X_train["time"].size/batch_size)-1
    net = Net(drop)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    #criterion = nn.MSELoss()
    criterion=nn.L1Loss()
    for epoch in range(epochs):
        for batch in range(train_batches):
            optimizer.zero_grad()
            net_out=net(train_data[batch*batch_size:(batch_size*(batch+1))].float())
            loss = criterion(net_out, train_target[batch*batch_size:(batch_size*(batch+1))].float())
            loss.backward()
            optimizer.step()
    mae=0
    net_out=net(test_data.float())
    for i in range(test_data.shape[0]):
        mae+=abs(net_out[i]-test_target[i])
    mae/=test_data.shape[0]
    mae
    return mae


# In[39]:


#Обучение модели и ее возврат

def fit(drop=0.1,learning_rate=0.01, batch_size=30):
    epochs=10
    train_batches=math.ceil(X_train["time"].size/batch_size)-1
    net = Net(drop)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    #criterion = nn.MSELoss()
    criterion=nn.L1Loss()
    for epoch in range(epochs):
        for batch in range(train_batches):
            optimizer.zero_grad()
            net_out=net(train_data[batch*batch_size:(batch_size*(batch+1))].float())
            loss = criterion(net_out, train_target[batch*batch_size:(batch_size*(batch+1))].float())
            loss.backward()
            optimizer.step()
    return net


# In[37]:


#grid search

search=pd.DataFrame(columns=["drop","lr","batch","loss"])
count=0
for i in np.arange(0.0,0.51,0.05):
    for j in np.arange(0.001,0.01,0.001):
        for k in np.arange(20,71,5):
            search.loc[count]=[i,j,k,checkloss(i,j,k).detach()]
            count+=1
search


# In[ ]:


gsmin=search["loss"].min()
minid=0
for i in range(search["drop"].size):
    if (search["loss"][i]==gsmin):
        minid=i
search.loc[minid]
#drop=0, lr=0.006, batch=20, mae=66.7938


# In[ ]:


net=fit(0,0.006,20)


# In[40]:


#Итоговые модели:

net   #Pytorch nn
model #skelarn random forest


# In[ ]:




