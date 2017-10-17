
# coding: utf-8

# In[25]:

import pandas as pd
import matplotlib.pyplot as plt
shop_mall=pd.read_csv(r'C:\Users\mashiji\Desktop\test\train_shop_mall.csv')
user_shop=pd.read_csv(r'C:\Users\mashiji\Desktop\test\train_user_shop.csv')
test_evaluation=pd.read_csv(r'C:\Users\mashiji\Desktop\test\test_evaluation.csv')
print(shop_mall.shape)
# print(user_shop.shape)
# print(test_evaluation.shape)
table1=[['shop_id'],['category_id'],['longitude'],['latitude'],['price'],['mall_id']]
table2=[['user_id'],['shop_id'],['time_stamp'],['longitude'],['latitude'],['wifi_infos']]
table3=[['row_id'],['user_id'],['mall_id'],['time_stamp'],['longitude'],['latitude'],['wifi_infos']]
for i in table1:
    d1=pd.isnull(shop_mall[i])
    d1_true=d1[d1==True]
    d1_count=len(d1_true)
    print(i,d1_count)
print('\n')
print(user_shop.shape)
for i in table2:
    d1=pd.isnull(user_shop[i])
    d1_true=d1[d1==True]
    d1_count=len(d1_true)
    print(i,d1_count)
print('\n')    
print(test_evaluation.shape)
for i in table3:
    d1=pd.isnull(test_evaluation[i])
    d1_true=d1[d1==True]
    d1_count=len(d1_true)
    print(i,d1_count)

