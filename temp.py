import pandas as pd
import numpy as np
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

path='C:\\Kaggle\\Restaurants\\data\\'

air_reserve=pd.read_csv(path+'air_reserve.csv')
air_reserve['visit']=pd.to_datetime(air_reserve['visit_datetime'])
air_reserve['reserve']=pd.to_datetime(air_reserve['reserve_datetime'])
air_reserve=air_reserve.drop(['visit_datetime','reserve_datetime'],axis=1)
store_id_relation=pd.read_csv(path+'store_id_relation.csv')
air_store_info=pd.read_csv(path+'air_store_info.csv')
air_visit_data=pd.read_csv(path+'air_visit_data.csv')
air_visit_data['visit']=pd.to_datetime(air_visit_data['visit_date'])
air_visit_data=air_visit_data.drop(['visit_date'], axis=1)

hpg_reserve=pd.read_csv(path+'hpg_reserve.csv')
hpg_reserve['visit']=pd.to_datetime(hpg_reserve['visit_datetime'])
hpg_reserve['reserve']=pd.to_datetime(hpg_reserve['reserve_datetime'])
hpg_reserve=hpg_reserve.drop(['visit_datetime','reserve_datetime'],axis=1)
hpg_reserve=pd.merge(hpg_reserve,store_id_relation, how='inner', on='hpg_store_id')
hpg_store_info=pd.read_csv(path+'hpg_store_info.csv')
hpg_store_info=pd.merge(hpg_store_info,store_id_relation, how='inner', on='hpg_store_id')
sample_submission=pd.read_csv(path+'sample_submission.csv')
reservations=pd.concat([air_reserve[['air_store_id','reserve_visitors','visit','reserve']],
                         hpg_reserve[['air_store_id','reserve_visitors','visit','reserve']]],axis=0)

# DATE INFO
date_info=pd.read_csv(path+'date_info.csv')
date_info['visit']=pd.to_datetime(date_info['calendar_date'])
date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 5
date_info=date_info.set_index('visit')
golden_week=pd.date_range('2016-04-29','2016-05-07', freq='D').union(pd.date_range('2017-04-28','2017-05-06', freq='D'))
date_info['fl_gw']=0
date_info.loc[golden_week,'fl_gw']=1
date_info=date_info.reset_index()
air_visit_data.loc[air_visit_data['visitors']>350,'visitors']=0
test_dates = pd.DataFrame(pd.date_range('2017-04-23','2017-05-31', freq='D'))
test_dates.columns=['visit']
unique_stores=pd.DataFrame(sample_submission['id'].apply(lambda x:x[:-11]).unique())
unique_stores.columns=['air_store_id']
test_dates['key']=1
unique_stores['key']=1
test_visit_data=unique_stores.merge(test_dates, on='key').drop('key',axis=1)
test_visit_data['visitors']=0
               
reservations['days_diff']=(reservations['visit'].dt.date-reservations['reserve'].dt.date).dt.days
reservations['dow']=reservations['visit'].dt.dayofweek
stores_res=reservations.groupby(['air_store_id',
                              reservations['visit'].
                              dt.date])['reserve_visitors'].agg({'res_visit':np.sum}).reset_index().set_index('air_store_id','visit')
tmp2=reservations.groupby(['air_store_id',
                              reservations['visit'].
                              dt.date])['days_diff'].agg({'mean_days_diff':np.mean}).reset_index().set_index('air_store_id','visit')
stores_res['mean_days_diff']=tmp2['mean_days_diff']
tmp3=reservations.groupby(['air_store_id','dow'])['reserve_visitors'].agg({'mean_visit':np.mean,
                                                                 'median_visit':np.median}).reset_index()
tmp=tmp3.pivot_table(index='air_store_id', columns='dow', fill_value=True).reset_index().set_index('air_store_id')
tmp.columns=['res_mean_visit_0','res_mean_visit_1','res_mean_visit_2',
             'res_mean_visit_3','res_mean_visit_4','res_mean_visit_5','res_mean_visit_6',
             'res_median_visit_0','res_median_visit_1','res_mediean_visit_2',
             'res_median_visit_3','res_median_visit_4','res_median_visit_5','res_median_visit_6']
stores_res[tmp.columns]=tmp
stores_res=stores_res.reset_index()
stores_res['visit']=pd.to_datetime(stores_res['visit'])

complete=pd.concat([air_visit_data, test_visit_data], axis=0)

complete['dow']=complete['visit'].dt.dayofweek
complete['month']=complete['visit'].dt.month
complete['week']=complete['visit'].dt.week

air_visit_data['dow']=air_visit_data['visit'].dt.dayofweek
air_visit_data['month']=air_visit_data['visit'].dt.month
air_visit_data['week']=air_visit_data['visit'].dt.week

dow_visit=air_visit_data.groupby(['air_store_id','dow'])['visitors'].agg({'mean_dow_visit':np.mean,
                                                                    'median_dow_visit':np.median,
                                                                    'max_dow_visit':np.max,
                                                                    'std_dow_visit':np.std}).fillna(0).reset_index()
complete=complete.merge(dow_visit,how='left', on=['air_store_id','dow'])           
month_visit=air_visit_data.groupby(['air_store_id','month'])['visitors'].agg({'mean_month_visit':np.mean,
                                                                        'median_month_visit':np.median,
                                                                        'max_month_visit':np.max, 
                                                                        'std_month_visit':np.std}).fillna(0).reset_index()
complete=complete.merge(month_visit,how='left', on=['air_store_id','month'])
tot_visit=air_visit_data.groupby(['air_store_id'])['visitors'].agg({'mean_visit':np.mean,
                                                              'median_visit':np.median,
                                                              'max_visit':np.max,
                                                              'std_visit':np.std}).reset_index()
complete=complete.merge(tot_visit,how='left', on=['air_store_id'])

week_visit=air_visit_data.groupby(['air_store_id','week'])['visitors'].agg({'mean_week_visit':np.mean,
                                                                      'median_week_visit':np.median,
                                                                      'max_week_visit':np.max, 
                                                                      'std_week_visit':np.std}).fillna(0).reset_index()
complete=complete.merge(week_visit,how='left', on=['air_store_id','week']) 


# ADD MISSING DATA
complete=complete.set_index('air_store_id')
dow_visit_missing=dow_visit.groupby(['air_store_id'])['mean_dow_visit'].agg({'mean_dow_visit':np.mean,
                                                                    'median_dow_visit':np.median,
                                                                    'std_dow_visit':np.std})
dow_visit_max_missing=dow_visit.groupby(['air_store_id'])['max_dow_visit'].agg({'max_dow_visit':np.max})
complete.loc[complete['mean_dow_visit'].isnull(),
             ['mean_dow_visit','median_dow_visit','std_dow_visit']]=dow_visit_missing[['mean_dow_visit','median_dow_visit','std_dow_visit']]
complete.loc[complete['max_dow_visit'].isnull(),'max_dow_visit']=dow_visit_max_missing['max_dow_visit']

month_visit_missing=month_visit.groupby(['air_store_id'])['mean_month_visit'].agg({'mean_month_visit':np.mean,
                                                                    'median_month_visit':np.median,
                                                                    'std_month_visit':np.std})
month_visit_max_missing=month_visit.groupby(['air_store_id'])['max_month_visit'].agg({'max_month_visit':np.max})
complete.loc[complete['mean_month_visit'].isnull(),
             ['mean_month_visit','median_month_visit','std_month_visit']]=month_visit_missing[['mean_month_visit','median_month_visit','std_month_visit']]
complete.loc[complete['max_month_visit'].isnull(),'max_month_visit']=month_visit_max_missing['max_month_visit']
week_visit_missing=week_visit.groupby(['air_store_id'])['mean_week_visit'].agg({'mean_week_visit':np.mean,
                                                                    'median_week_visit':np.median,
                                                                    'std_week_visit':np.std})
week_visit_max_missing=week_visit.groupby(['air_store_id'])['max_week_visit'].agg({'max_week_visit':np.max})
complete.loc[complete['mean_week_visit'].isnull(),
             ['mean_week_visit','median_week_visit','std_week_visit']]=week_visit_missing[['mean_week_visit','median_week_visit','std_week_visit']]
complete.loc[complete['max_week_visit'].isnull(),'max_week_visit']=week_visit_max_missing['max_week_visit']

complete=complete.reset_index()
complete['dayofyear']=complete['visit'].dt.dayofyear
complete['year']=complete['visit'].dt.year
complete['day']=complete['visit'].dt.day
complete['year_month']=complete['visit'].apply(lambda x: 100*x.year+x.month)
complete['fl_weekend']=complete['dow'].apply(lambda x: 1 if (x==5)or(x==6) else 0)

complete['counter']=complete.sort_values('visit').groupby(['air_store_id'])['visit'].cumcount()+1
complete=complete.merge(date_info[['visit','holiday_flg','fl_gw','weight']], how='left', on='visit')
complete=complete.merge(stores_res, how='left', on=['air_store_id','visit'])
complete['fl_reserve']=complete['mean_days_diff'].apply(lambda x: 0 if np.isnan(x) else 1)

air_store_info.loc[air_store_info['air_genre_name'] =='Asian','air_genre_name']='Other'
air_store_info.loc[air_store_info['air_genre_name'] =='International cuisine','air_genre_name']='Other'
air_store_info.loc[air_store_info['air_genre_name'] =='Karaoke/Party','air_genre_name']='Other'

lbl = LabelEncoder()
air_store_info['genre_name']=lbl.fit_transform(air_store_info['air_genre_name'].apply(lambda x: re.split('/| ',x)[0]))
air_store_info['area_name']=lbl.fit_transform(air_store_info['air_area_name'].apply(lambda x: re.split('-| ',x)[0]))
air_store_info['genre_name']=air_store_info['genre_name'].astype('category')
air_store_info['area_name']=air_store_info['area_name'].astype('category')
air_store_info['var_max_lat'] = air_store_info['latitude'].max() - air_store_info['latitude']
air_store_info['var_max_long'] = air_store_info['longitude'].max() - air_store_info['longitude']
air_store_info['lon_plus_lat'] = air_store_info['longitude'] + air_store_info['latitude']

store_info=air_store_info[['air_store_id','genre_name','area_name','latitude','longitude','var_max_lat',
                           'var_max_long','lon_plus_lat']].set_index('air_store_id')
store_info=pd.get_dummies(store_info).reset_index()
complete=complete.merge(store_info, on='air_store_id')
complete=complete.fillna(-1)

complete=complete.set_index(['air_store_id','visit'])
shifted_39 = complete.groupby(level='air_store_id').shift(39).fillna(method='bfill')
complete=complete.join(shifted_39[['visitors']].rename(columns=lambda x: x+"_lag39"))
complete=complete.join(shifted_39[['weight']].rename(columns=lambda x: x+"_lag39"))

complete['weight_visitors']=complete.apply(lambda x: x['visitors_lag39']*x['weight_lag39'],axis=1).fillna(method='bfill')
complete['weight_visitors_roll']=complete['weight_visitors'].rolling(7).sum().fillna(method='bfill')
complete['weight_roll']=complete['weight_lag39'].rolling(7).sum().fillna(method='bfill')
complete['avg_visitors_7']=complete.apply(lambda x: x['weight_visitors_roll']/x['weight_roll'],axis=1)
complete['weight_visitors_roll']=complete['weight_visitors'].rolling(15).sum().fillna(method='bfill')
complete['weight_roll']=complete['weight_lag39'].rolling(15).sum().fillna(method='bfill')
complete['avg_visitors_15']=complete.apply(lambda x: x['weight_visitors_roll']/x['weight_roll'],axis=1)
complete['weight_visitors']=complete.apply(lambda x: x['visitors_lag39']*x['weight_lag39'],axis=1).fillna(method='bfill')
complete['weight_visitors_roll']=complete['weight_visitors'].rolling(7).sum().fillna(method='bfill')
complete['weight_roll']=complete['weight_lag39'].rolling(3).sum().fillna(method='bfill')
complete['avg_visitors_3']=complete.apply(lambda x: x['weight_visitors_roll']/x['weight_roll'],axis=1)
complete['max_7']=complete.visitors_lag39.rolling(7).max().fillna(method='bfill')
complete['max_40']=complete.visitors_lag39.rolling(40).max().fillna(method='bfill')
complete['mean_7']=complete.visitors_lag39.rolling(7).apply(np.mean).fillna(method='bfill')
complete['mean_40']=complete.visitors_lag39.rolling(40).apply(np.mean).fillna(method='bfill')
complete['median_7']=complete.visitors_lag39.rolling(7).apply(np.median).fillna(method='bfill')
complete['median_40']=complete.visitors_lag39.rolling(40).apply(np.median).fillna(method='bfill')
complete=complete.reset_index()
complete.to_csv(path+'complete.csv', index=False)