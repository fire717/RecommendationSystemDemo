 #coding:utf-8
# Fire
import numpy as np
import pandas as pd
from faker import Faker

def makeItem(filename, savename):
    item_list = []
    with open(filename, "r", encoding='ISO-8859-1') as f:
        lines = f.readlines()
    for line in lines:
        tmp_splits = line.split("|")
        item_id = int(tmp_splits[0])
        item_name = tmp_splits[1]
        item_list.append([item_id, item_name])

    df = pd.DataFrame(item_list, columns=['movie_id','movie_name'])
    df.to_csv(savename, index=False,header=True)


def makeUser(filename, savename1, savename2, savename3):
    """
    943 users
    1682 items
    100000 ratings
    """
    fake = Faker(['zh_CN'])

    user_num = 943
    item_num = 1682

    user_list = []
    for i in range(user_num):
        user_list.append([i+1, fake.name()])

    df = pd.DataFrame(user_list, columns=['user_id','user_name'])
    df.to_csv(savename1, index=False,header=True)


    user_rate_list = np.zeros((user_num, item_num+1))
    for i in range(user_num):
        user_rate_list[i][0] = i+1

    user_rate_origin = []


    with open(filename, "r", encoding='ISO-8859-1') as f:
        lines = f.readlines()
        for line in lines:
            tmp_splits = line.strip().split("\t")
            user_id, item_id, rate = [int(x) for x in tmp_splits[:3]]

            user_rate_list[user_id-1][item_id] = rate

            user_rate_origin.append([user_id, item_id, rate])


    user_rate_list_head = ['user_id']+[str(x+1) for x in range(item_num)]
    df = pd.DataFrame(user_rate_list, columns=user_rate_list_head)
    df.to_csv(savename2, index=False,header=True)

    df = pd.DataFrame(user_rate_origin, columns=['user_id','item_id','rate'])
    df.to_csv(savename3, index=False,header=True)




if __name__ == "__main__":
    makeItem("u.item", "movies.csv")
    makeUser("u.data", "users.csv", "user_rates.csv", "user_rates_origin.csv")

    # df = pd.read_csv("user_rates.csv")
    # print(df.head(5))