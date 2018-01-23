
# coding: utf-8

# In[67]:

import pandas as pd
import numpy as np
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor
get_ipython().magic('pylab inline')


# In[23]:

path='C:\\Kaggle\\Restaurants\\data\\'


# In[24]:

complete=pd.read_csv(path+'complete.csv')


# In[25]:



complete=complete.reset_index()
training=complete[complete.visit<'2017-04-23']
testing=complete[complete.visit>='2017-04-23']


# In[29]:

def get_split_store(df, ratio=0.65):
    train_len=df.shape[0]*ratio
    train = df[df['counter']<=train_len]
    val=df[df['counter']>train_len]
    return train, val


# In[30]:

unique_stores=pd.DataFrame(training.air_store_id.unique())
unique_stores.columns=['air_store_id']


# In[31]:

train=pd.DataFrame()
valid=pd.DataFrame()
for item in unique_stores.air_store_id:
    _df=training.loc[training['air_store_id']==item].copy()
    tr,val=get_split_store(_df)
    train=train.append(tr)
    valid=valid.append(val)


# In[54]:

train.columns.tolist()
cols=[
 'mean_dow_visit',
 'median_dow_visit',
 'max_dow_visit',
 'std_dow_visit',
 'mean_month_visit',
 'median_month_visit',
 'max_month_visit',
 'std_month_visit',
 'mean_visit',
 'median_visit',
 'max_visit',
 'std_visit',
 'mean_week_visit',
 'median_week_visit',
 'max_week_visit',
 'std_week_visit',
 'dayofyear',
 'year',
 'day',
 'year_month',
 'fl_weekend',
 'counter',
 'holiday_flg',
 'latitude',
 'longitude',
 'visitors_lag',
 'sum_10',
 'sum_30',
 'max_10',
 'max_30',
 'mean_10',
 'mean_30',
 'median_10',
 'median_30']


# In[55]:

X_train=train[cols]
y_train=train['visitors']
X_valid=valid[cols]
y_valid=valid['visitors']


# In[63]:

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_valid=sc.transform(X_valid)


# In[37]:

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(mean_squared_error(y,y0))


# In[69]:

model1 = ExtraTreesRegressor(random_state=3, n_estimators=200,max_depth =10)


# In[70]:

model1.fit(training[cols], np.log1p(training.visitors.values))


# In[71]:

preds1 = model1.predict(X_train)


# In[72]:

print('RMSE GradientBoostingRegressor(train): ', rmsle(np.log1p(y_train), preds1))


# In[74]:

preds1 = model1.predict(X_valid)


# In[75]:

print('RMSE GradientBoostingRegressor(valid): ', rmsle(np.log1p(y_valid), preds1))


# In[76]:

preds1 = model1.predict(testing[cols])


# In[18]:

print('RMSE GradientBoostingRegressor: ', rmsle(np.log1p(y_train.values), preds1))


# In[50]:

X_test=testing[cols]
pred =np.expm1(model1.predict(X_test))


# In[78]:

preds1


# In[84]:

testing['visitors']=np.expm1(preds1)


# In[85]:

testing['id']=testing.apply(lambda x: x['air_store_id']+'_'+str(x['visit']),axis=1)


# In[87]:

submission=testing[['id','visitors']]


# In[88]:

submission.to_csv(path+'submission.csv', index=False)


# In[89]:

submission.head()


# In[45]:

testing.shape


# In[53]:

X_train.head()


# In[ ]:



