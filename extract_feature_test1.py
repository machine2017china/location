# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from datetime import date
from sklearn import preprocessing
import mzgeohash

train=pd.read_csv('data/train_ccf_first_round_user_shop_behavior.csv',header=None)
test=pd.read_csv('data/AB_test_evaluation_public.csv',header=None)

train.columns=['user_id','shop_id','time_stamp','longitude','latitude','wifi_infos']
test.columns=['row_id','user_id','mall_id','time_stamp','longitude','latitude','wifi_infos']
########shop_id#########
#train
lbl=preprocessing.LabelEncoder()
lbl.fit(list(set(train['shop_id'].values)))
train['shop_id']=lbl.transform(train['shop_id'].values)

#test
#test['shop_id']=lbl.transform(test['shop_id'].values)



##########time_stamp##########
train['day_of_week']=train.time_stamp.astype('str').apply(lambda x:date(int(x[0:4]),int(x[5:7]),int(x[8:10])).weekday()+1)
train['hour']=train.time_stamp.astype('str').apply(lambda x:int(x[11:13]))
train['minute']=train.time_stamp.astype('str').apply(lambda x:int(x[14:16]))

train.drop(['time_stamp'],axis=1,inplace=True)
weekday_dummies=pd.get_dummies(train.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
train['is_weekend']=train.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
train=pd.concat([train,weekday_dummies],axis=1)

#test
test['day_of_week']=test.time_stamp.astype('str').apply(lambda x:date(int(x[0:4]),int(x[5:7]),int(x[8:10])).weekday()+1)
test['hour']=test.time_stamp.astype('str').apply(lambda x:int(x[11:13]))
test['minute']=test.time_stamp.astype('str').apply(lambda x:int(x[14:16]))

test.drop(['time_stamp'],axis=1,inplace=True)
weekday_dummies=pd.get_dummies(test.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
test['is_weekend']=test.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
test=pd.concat([test,weekday_dummies],axis=1)


################geohash######################
def get_geohash(x):
    return mzgeohash.encode((float(x[0]),float(x[1])))

def get_geohash_e(geohash):
    this_neighbors=mzgeohash.neighbors(geohash)
    return this_neighbors['e']

def get_geohash_n(geohash):
    this_neighbors=mzgeohash.neighbors(geohash)
    return this_neighbors['n']

def get_geohash_ne(geohash):
    this_neighbors=mzgeohash.neighbors(geohash)
    return this_neighbors['ne']

def get_geohash_nw(geohash):
    this_neighbors=mzgeohash.neighbors(geohash)
    return this_neighbors['nw']

def get_geohash_s(geohash):
    this_neighbors=mzgeohash.neighbors(geohash)
    return this_neighbors['s']

def get_geohash_se(geohash):
    this_neighbors=mzgeohash.neighbors(geohash)
    return this_neighbors['se']

def get_geohash_sw(geohash):
    this_neighbors=mzgeohash.neighbors(geohash)
    return this_neighbors['sw']

def get_geohash_w(geohash):
    this_neighbors=mzgeohash.neighbors(geohash)
    return this_neighbors['w']

train_location=train[['longitude','latitude']]
train_location['geohash']=train_location.apply(get_geohash,axis=1)
train=train.merge(train_location,on=['longitude','latitude'],how='inner')
train.drop(['longitude','latitude'],axis=1,inplace=True)

train['geohash_e']=train.geohash.apply(get_geohash_e)
train['geohash_n']=train.geohash.apply(get_geohash_n)
train['geohash_ne']=train.geohash.apply(get_geohash_ne)
train['geohash_nw']=train.geohash.apply(get_geohash_nw)
train['geohash_s']=train.geohash.apply(get_geohash_s)
train['geohash_se']=train.geohash.apply(get_geohash_se)
train['geohash_sw']=train.geohash.apply(get_geohash_sw)
train['geohash_w']=train.geohash.apply(get_geohash_w)

#test
test_location=test[['longitude','latitude']]
test_location['geohash']=test_location.apply(get_geohash,axis=1)
test=test.merge(test_location,on=['longitude','latitude'],how='inner')
test.drop(['longitude','latitude'],axis=1,inplace=True)

test['geohash_e']=test.geohash.apply(get_geohash_e)
test['geohash_n']=test.geohash.apply(get_geohash_n)
test['geohash_ne']=test.geohash.apply(get_geohash_ne)
test['geohash_nw']=test.geohash.apply(get_geohash_nw)
test['geohash_s']=test.geohash.apply(get_geohash_s)
test['geohash_se']=test.geohash.apply(get_geohash_se)
test['geohash_sw']=test.geohash.apply(get_geohash_sw)
test['geohash_w']=test.geohash.apply(get_geohash_w)

lbl=preprocessing.LabelEncoder()
lbl.fit(list(set(train['geohash'].values)|set(test['geohash'].values)|\
set(train['geohash_e'])|set(test['geohash_e'])|\
set(train['geohash_n'])|set(test['geohash_n'])|\
set(train['geohash_ne'])|set(test['geohash_ne'])|\
set(train['geohash_nw'])|set(test['geohash_nw'])|\
set(train['geohash_s'])|set(test['geohash_s'])|\
set(train['geohash_se'])|set(test['geohash_se'])|\
set(train['geohash_sw'])|set(test['geohash_sw'])|\
set(train['geohash_w'])|set(test['geohash_w'])))


train['geohash']=lbl.transform(train['geohash'].values)
train['geohash_e']=lbl.transform(train['geohash_e'].values)
train['geohash_n']=lbl.transform(train['geohash_n'].values)
train['geohash_ne']=lbl.transform(train['geohash_ne'].values)
train['geohash_nw']=lbl.transform(train['geohash_nw'].values)
train['geohash_s']=lbl.transform(train['geohash_s'].values)
train['geohash_se']=lbl.transform(train['geohash_se'].values)
train['geohash_sw']=lbl.transform(train['geohash_sw'].values)
train['geohash_w']=lbl.transform(train['geohash_w'].values)

test['geohash']=lbl.transform(test['geohash'].values)
test['geohash_e']=lbl.transform(test['geohash_e'].values)
test['geohash_n']=lbl.transform(test['geohash_n'].values)
test['geohash_ne']=lbl.transform(test['geohash_ne'].values)
test['geohash_nw']=lbl.transform(test['geohash_nw'].values)
test['geohash_s']=lbl.transform(test['geohash_s'].values)
test['geohash_se']=lbl.transform(test['geohash_se'].values)
test['geohash_sw']=lbl.transform(test['geohash_sw'].values)
test['geohash_w']=lbl.transform(test['geohash_w'].values)


######################wifi_info################
#train
s=train['wifi_infos'].str.split(';').apply(pd.Series,1).stack()
s.index=s.index.droplevel(-1)
s.name='wifi_info'
del train['wifi_infos']
train=train.join(s)
train=train.reset_index()
train['bssid']=train.wifi_info.apply(lambda s:s.split('|')[0])
train['signal']=train.wifi_info.apply(lambda s:s.split('|')[1])
train['wifi_flag']=train.wifi_info.apply(lambda s:s.split('|')[2])

train.drop('wifi_info',axis=1,inplace=True)

#test
s=test['wifi_infos'].str.split(';').apply(pd.Series,1).stack()
s.index=s.index.droplevel(-1)
s.name='wifi_info'
del test['wifi_infos']
test=test.join(s)
test=test.reset_index()
test['bssid']=test.wifi_info.apply(lambda s:s.split('|')[0])
test['signal']=test.wifi_info.apply(lambda s:s.split('|')[1])
test['wifi_flag']=test.wifi_info.apply(lambda s:s.split('|')[2])

test.drop('wifi_info',axis=1,inplace=True)

lbl=preprocessing.LabelEncoder()
lbl.fit(list(set(train['bssid'].values)|set(test['bssid'].values)))
train['bssid']=lbl.transform(train['bssid'].values)
test['bssid']=lbl.transform(test['bssid'].values)

train['wifi_flag']=train.wifi_flag.astype('str').apply(lambda s:0 if s=='false' else 1)
test['wifi_flag']=test.wifi_flag.astype('str').apply(lambda s:0 if s=='false' else 1)

train.to_csv('output/train.csv')
test.to_csv('output/test.csv')

print('train shape',train.shape)
print('test shape',test.shape)