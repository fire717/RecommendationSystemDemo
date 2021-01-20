 #coding:utf-8
# Fire
import numpy as np
import pandas as pd





df = pd.read_csv("user_rates_origin_bak.csv")

df['log_id']  = [x for x in range(1, df.shape[0]+1)]
df['rate_date']  = "0"
df['rate_timestamp']  = 0


df = df[['log_id','user_id','item_id','rate','rate_date','rate_timestamp']]
df.to_csv("log_user_rates.csv",index=False,header=True)

#print(df.info())






df = pd.read_csv("users_bak.csv")

df['update_date']  = "0"
df['update_timestamp']  = 0


df = df[['user_id','user_name','update_date','update_timestamp']]
df.to_csv("users.csv",index=False,header=True)



df = pd.read_csv("items_bak.csv")

df['update_date']  = "0"
df['update_timestamp']  = 0


df = df[['item_id','item_name','update_date','update_timestamp']]
df.to_csv("items.csv",index=False,header=True)