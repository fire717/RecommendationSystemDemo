""" Algorithm predicting a random rating.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import xlearn as xl
import numpy as np
import pandas as pd
from surprise import AlgoBase
import random

import time

class FM(AlgoBase):
    """

    """

    def __init__(self):

        AlgoBase.__init__(self)
        self.feature_len = 0
        self.users = 0
        self.items = 0

        self.fm_model = None
        self.user_item_rated = {}

        self.res_dict = {}

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        #print(trainset)
        #print(trainset.ur)
        """
 ({... , 941: [(147, 3.0), (811, 1.0), (882, 3.0), (480, 3.0), (200, 4.0), (975, 4.0), 
 (696, 5.0), (214, 2.0), (183, 4.0), (238, 2.0), (18, 5.0), (179, 3.0), 
 (171, 2.0), (335, 3.0), (75, 1.0), (371, 2.0), (541, 3.0), (113, 2.0), 
 (484, 2.0)], 
 942: [(34, 4.0), (979, 3.0), (142, 4.0), (814, 4.0),
  (89, 5.0), (181, 3.0), (553, 2.0), (508, 3.0), (39, 4.0), (77, 3.0),
  (162, 3.0), (343, 4.0), (16, 2.0), (184, 3.0), (371, 3.0), (13, 4.0),
   (458, 3.0), (266, 4.0), (337, 5.0), (192, 1.0), (197, 3.0), (1086, 4.0),
    (383, 3.0), (22, 2.0), (5, 4.0), (528, 4.0)]})
        """
        #print(trainset.ir)
        #print(trainset.n_users)#943
        #print(trainset.n_items)#1646
        #print(trainset.n_ratings )#80001
        #print(trainset.rating_scale)#(1, 5)
        # print(trainset._raw2inner_id_users)
        #print(trainset._raw2inner_id_items)



        # 1. 数据转换/构造特征  trainset -> 稀疏one-hot -> DMatrix  
        # to do: 平均打分、打分次数...
        user_rate_list = []
        for k,v in trainset.ur.items():
            for item_id, rate in v:
                user_rate_list.append([k, item_id, rate])

                self.user_item_rated[(k,item_id)] = rate

        print("Done data to list:",len(user_rate_list))

        user_rate_array = np.array(user_rate_list)
        data_num = len(user_rate_array)

        # 获取所有userid
        self.users = list(set(user_rate_array[:,0]))
        print(len(self.users),self.users[:5])


        # 获取所有itemid
        self.items = list(set(user_rate_array[:,1]))
        print(len(self.items),self.items[:5])

        self.feature_len = len(self.users)+len(self.items)+1
        feature_data = []

        for i in range(data_num):
            if i%10000 == 0:
                print("done ", i)
            x = [0]*self.feature_len

            u_idx = self.users.index(user_rate_array[i][0])
            i_idx = self.items.index(user_rate_array[i][1])

            x[0] = user_rate_array[i][2]
            x[u_idx+1] = 1
            x[len(self.users)+i_idx+1] = 1

            # user_rates = df[df['user_id']==df.loc[i,'user_id']]
            # rate_times = len(user_rates)
            # implicit = 1.0/rate_times
            # # avg_score = np.mean(user_rates['rate'])
            # # print(user_rates)
            # # print(rate_times, implicit, avg_score)
            # # b
            # for j in user_rates['item_id'].values:
            #     x[len(users)+len(items)+items.index(j)+1] = implicit
            # x[-1] = rate_times    

            feature_data.append(x)

        random.shuffle(feature_data)
        print("Done data to features.")

        split_ratio = 0.9
        split_idx = int(data_num*split_ratio)

        data_train = feature_data[:split_idx]
        data_val = feature_data[split_idx:]

        # with open("feature_data_train.csv", "w") as f:
        #     for i in range(len(feature_data)):
        #         if i%10000 == 0:
        #             print("done ", i)
        #         line = ','.join([str(x) for x in feature_data[i]])+"\n"
        #         f.write(line)

        # with open("feature_data_val.csv", "w") as f:
        #     for i in range(len(feature_data)):
        #         if i%10000 == 0:
        #             print("done ", i)
        #         line = ','.join([str(x) for x in feature_data[i]])+"\n"
        #         f.write(line)
        # print("Done save")

        # 2. 创建FM
        # data_train = pd.read_csv("feature_data_train.csv", header=None, sep=",")
        # data_val = pd.read_csv("../my_data/suprisedata/data/feature_data_val.csv", header=None, sep=",")
        data_train = pd.DataFrame(data_train)
        data_val = pd.DataFrame(data_val)
        print("Done data to DF.")

        X_train = data_train[data_train.columns[1:]]
        y_train = data_train[0]

        X_val = data_val[data_val.columns[1:]]
        y_val = data_val[0]

        xdm_train = xl.DMatrix(X_train, y_train)
        xdm_val = xl.DMatrix(X_val, y_val)


        # 3. 训练
        fm_model = xl.create_fm()
        fm_model.setTrain(xdm_train)   
        fm_model.setValidate(xdm_val)  


        param = {'task':'reg', 'lr':0.2, 
                 'lambda':0.001, 'metric':'rmse',
                 'k':40, 'epoch':10, 'stop_window':2}

        print("Start to train")
        t = time.time()
        fm_model.setTXTModel("./model.txt")
        fm_model.fit(param, './model_dm.out')
        print("cost time: ", time.time() - t)
        

        # 4.获取所有没有评分的user-item，统一pre
        # data_test = []
        # for iuid in self.users:
        #     for iiid in self.items:
        #         if (iuid,iiid) not in  self.user_item_rated:
        #             u_idx = self.users.index(iuid)
        #             i_idx = self.items.index(iiid)

        #             x[0] = 0
        #             x[u_idx+1] = 1
        #             x[len(self.users)+i_idx+1] = 1
        #             data_test.append(x)
        # print("len test: ", len(data_test))
        # data_test = pd.DataFrame(data_test[:100000])
        # X_test = data_test[data_test.columns[1:]]
        # y_test = data_test[0]
        # xdm_test = xl.DMatrix(X_test, y_test)
        # fm_model.setTest(xdm_test)  # Test data
        # res = fm_model.predict("./model_dm.out", "./output.txt")

        # 5.保存成可查询的结果  user-item:rate
        
        return self

    # def estimate(self, iuid, iiid):
    # 耗时：15小时   RMSE：1.0155
    #     print("iuid, iiid ",iuid, iiid)
    #     if (iuid,iiid) in  self.user_item_rated:
    #         return self.user_item_rated[(iuid,iiid)]

    #     # 5. 读取模型

    #     # 6. 数据转换
    #     data_test = []

    #     x = [0]*self.feature_len

    #     if iuid not in self.users or iiid not in self.items:
    #         return 3


    #     u_idx = self.users.index(iuid)
    #     i_idx = self.items.index(iiid)

    #     x[0] = 0
    #     x[u_idx+1] = 1
    #     x[len(self.users)+i_idx+1] = 1
    #     data_test.append(x)


    #     data_test = pd.DataFrame(data_test)

    #     X_test = data_test[data_test.columns[1:]]
    #     y_test = data_test[0]

    #     xdm_test = xl.DMatrix(X_test, y_test)


    #     # 7. 预测
    #     if self.fm_model is None:
    #         self.fm_model = xl.create_fm()
    #     self.fm_model.setTest(xdm_test)  # Test data
    #     res = self.fm_model.predict("./model_dm.out")

    #     # 8. 结果数据转换


    #     print(res[0])
    #     return res[0]



    def estimate(self, iuid, iiid):
        # 耗时7.14s  RMSE 0.9948
        #print("iuid, iiid ",iuid, iiid)
        if (iuid,iiid) in  self.user_item_rated:
            return self.user_item_rated[(iuid,iiid)]

        if iuid not in self.users or iiid not in self.items:
            return 3
        u_idx = self.users.index(iuid)
        i_idx = self.items.index(iiid)
        #print("u_idx, i_idx ",u_idx, i_idx)

        if len(self.res_dict) == 0:
            len_user = len(self.users)
            len_item = len(self.items)
            user_i = [0]*len_user
            item_i = [0]*len_item
            user_v = [0]*len_user
            item_v = [0]*len_item
            bias = 0

            with open("model.txt", "r") as f:
                lines = f.readlines()
                #print(len(lines))
                for line in lines:
                    k,v = line.strip().split(": ")
                    if k == "bias":
                        bias = float(v)
                    elif "_" in k:
                        k_type, k_value = k.split("_")
                        k_value = int(k_value)
                        if k_type == "i":
                            if k_value < len_user:
                                user_i[k_value] = float(v)
                            else:
                                item_i[k_value-len_user] = float(v)

                        elif k_type == "v":
                            if k_value < len_user:
                                user_v[k_value] = np.array([float(x) for x in v.split(" ")])
                            else:
                                item_v[k_value-len_user] = np.array([float(x) for x in v.split(" ")])

                        else:
                            print(k_type, k_value)
                    else:
                        print("wrong k: ",k)

            for i in range(len_user):
                for j in range(len_item):
                    rate = bias+user_i[i]+item_i[j]+np.sum(user_v[i]*item_v[j])
                    self.res_dict[(i,j)] = rate
            print("Done load res_dict.")

        return self.res_dict[(u_idx, i_idx)]

"""
RMSE: 1.0150
Finish runAlgorithmUserRate.
time:  3557.54731798172
1.0150452348328756
"""