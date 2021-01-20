 #coding:utf-8
# Fire
import numpy as np
from PIL import Image

from faker import Faker
import random

gen_user_num = 10
gen_item_num = 7

fake = Faker(['zh_CN'])


def make_users():
    datas = []
    for _ in range(gen_user_num):
        name = fake.name()
        gender = random.choice(["男", "女"])
        age = random.randint(12,70)
        isVIP = random.choice(["是", "否"])
        datas.append([name, gender, age, isVIP])
    return datas


def make_user_items(user_data):
    datas = []
    for i in range(len(user_data)):
        row_data = []
        name = user_data[i][0]
        row_data.append(name)
        for j in range(gen_item_num):
            item_like = random.choice(["感兴趣", "无感", "不感兴趣"])
            row_data.append(item_like)
        datas.append(row_data)
    return datas




user_data = make_users()
print(user_data[:3])
np.save("fake_user_data.npy", user_data)

user_item_data = make_user_items(user_data)
print(user_item_data[:3])
np.save("fake_user_item_data.npy", user_item_data)