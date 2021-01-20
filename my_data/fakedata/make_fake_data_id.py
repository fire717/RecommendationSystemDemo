 #coding:utf-8
# Fire
import numpy as np
import pandas as pd
from PIL import Image
import pretty_errors
#from faker import Faker
import random

gen_user_num = 10
gen_item_num = 7

#fake = Faker(['zh_CN'])


def make_users():
    datas = []
    for i in range(gen_user_num):
        name = i
        gender = random.choice([0,1])#"男", "女"
        age = random.randint(12,70)
        isVIP = random.choice([0,1])#"是", "否"
        datas.append([name, gender, age, isVIP])
    return datas


def make_user_items(user_data):
    data_for_read = []
    data_for_database = []
    for i in range(len(user_data)):
        row_data = []
        name = user_data[i][0]
        row_data.append(name)
        for j in range(gen_item_num):
            item_like = random.choice([0,1,2])#"感兴趣", "无感", "不感兴趣"
            row_data.append(item_like)

            data_for_database.append([i, j, item_like])

        data_for_read.append(row_data)
        
    return data_for_read, data_for_database




user_data = make_users()
df = pd.DataFrame(user_data, columns=['name','gender','age','isVIP'])
df.to_csv(r"./fake_user_data.csv", index=False,header=True)
print(df.head(3))



user_item_data_read, user_item_data_database = make_user_items(user_data)
df = pd.DataFrame(user_item_data_read, columns=['name','item1','item2','item3','item4','item5',
                                    'item6','item7'])
df.to_csv(r"./user_item_data_read.csv", index=False,header=True)
print(df.head(3))

df = pd.DataFrame(user_item_data_database, columns=['name','item','rate'])
df.to_csv(r"./user_item_data_database.csv", index=False,header=True)
print(df.head(3))