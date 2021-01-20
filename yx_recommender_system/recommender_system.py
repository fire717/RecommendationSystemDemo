#coding:utf-8
# @Fire 
# Start from 2019/12/31

import os,sys
import random
import datetime,time
import numpy as np
import pandas as pd

from surprise import NormalPredictor,KNNBasic,KNNWithMeans,KNNWithZScore,KNNBaseline
from surprise import SVD,SVDpp,NMF,CoClustering
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import cross_validate,train_test_split


from algorithms import RandomPredictor,FM


# from .algorithms import PredictionImpossible
# from .algorithms import Prediction

from collections import defaultdict
import json

my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

class YXRecommenderSystem:
    """
    Just get data from database or csv,
    then use algorithm to compute,
    then output result table to databse or csv.

    NOT used to search one user or one item,
    which should be searched from computed table.
    """
    def __init__(self, dir_csv_data, dir_csv_recs, dir_csv_logs):
        ################ base ###############
        self.version = 0.0

        self.dir_csv_data = dir_csv_data
        self.dir_csv_recs = dir_csv_recs
        self.dir_csv_logs = dir_csv_logs

        ################ name ###############
        self.users_csv_name = "users.csv"
        self.items_csv_name = "items.csv"

        self.log_user_rate_csv_name = "log_user_rates.csv"
        self.log_runTrain_time_csv_name = "log_runTrain_time.csv"

        self.recs_item_manual_set_csv_name = "recs_item_manual_set.csv"
        self.recs_item_neighbor_csv_name = "recs_item_neighbor.csv"
        self.recs_item_hotest_csv_name = "recs_item_hotest.csv"
        self.recs_item_user_favorite_csv_name = "recs_item_user_favorite.csv"

        self.recs_final_csv_name_pre = "recs_final_"


        ################ data ###############
        self.algorithm_user_rate = None
        # algorithm for user-rate data 

        self.algorithm_item_rate = None
        # algorithm for item-rate data 

        self.count_users = 0
        self.count_items = 0

        self.user_rate_data = None
        self.user_rate_trainset = None # 'Trainset' object
        self.user_rate_testset = None #list

        self.user_rate_predictions = None #list
        self.user_rate_acc_rmse = 0.0

        self.use_test_split = False
        # split data to train and test OR use full data to train

        ############## dataframe #############
        self.users = None
        self.items = None

        self.items_user_favorite = None
        self.items_neighbors = None
        self.items_hotest = None



        ########## longtime keep in memory #########
        self.last_train_time_log = None
        self.final_rec = None


    ############################ Data #########################
    def loadData(self, file_path, data_type = "csv", rating_scale=(1, 5), 
                use_test_split = False, test_split_ratio = 0.2, file_encode = 'utf-8'):
        """
        rating_scale: tuple, user rate score range
                    (The minimum and maximal rating of the rating scale.)
                    . e.g. (0,2) means 0,1,2 3 kinds of socre
        data_type: str, csv or database

        test_size: (float or int ) – If float, it represents the proportion of 
                    ratings to include in the testset. 
                    If int, represents the absolute number of ratings in the testset. 
                    Default is .2.
        """
        if not isinstance(rating_scale,tuple):
            print("ERROR: rating_scale type is not tuple: ", type(rating_scale))
            return -1

        if not isinstance(use_test_split,bool):
            print("ERROR: use_test_split type is not bool: ", type(use_test_split))
            return -1


        if data_type == "csv":
            #load user/item id-name pair
            self.users = pd.read_csv(os.path.join(self.dir_csv_data,self.users_csv_name))
            self.items = pd.read_csv(os.path.join(self.dir_csv_data,self.items_csv_name))


            if not os.path.exists(file_path):
                print("ERROR: Not exists file_path: ", file_path)
                return -1

            dataframe = pd.read_csv(file_path, encoding = file_encode)
            df_titles = ['user_id','item_id','rate']#dataframe.columns.tolist()

            dataframe = dataframe.drop_duplicates(df_titles,keep='last')

            reader = Reader(rating_scale=rating_scale)
            self.user_rate_data = Dataset.load_from_df(dataframe[df_titles], reader)

            self.count_users = len(dataframe[df_titles[0]].drop_duplicates())
            self.count_items = len(dataframe[df_titles[1]].drop_duplicates())
            #print(self.count_users, self.count_items)

        elif data_type == "database":
            print("ERROR: database have not completed.")
            return -2

        else:
            print("ERROR: Unknown data_type: ", data_type)
            print("-Note: data_type must be 'csv' or 'database'.")
            return -1

        if use_test_split:
            self.user_rate_trainset, self.user_rate_testset = train_test_split(self.user_rate_data, test_size=test_split_ratio)
        else:  
            self.user_rate_trainset = self.user_rate_data.build_full_trainset()


        self.use_test_split = use_test_split

        return 0



    def saveDataTrained(self, file_path, type_name, data_type = "csv"):
        """
        
        data_type: str, 'csv' or 'database'.
        file_path: for csv is save name, for database is table name.
        type_name: must be in ["items_user_favorite", "items_neighbors","items_hotest","final_rec"]
        """
        ret = 0
        if data_type == "csv":
            if type_name == "items_user_favorite":
                if self.items_user_favorite is None:
                    print("ERROR: save data is None, have you called the getUserTopNFavoriteItems() ?")
                    ret = -1
                else:
                    self.items_user_favorite.to_csv(file_path, index=False,header=True)
                    self.items_user_favorite = None

            elif type_name == "items_neighbors":
                if self.items_neighbors is None:
                    print("ERROR: save data is None, have you called the getItemNeighbors() ?")
                    ret = -1
                else:
                    self.items_neighbors.to_csv(file_path, index=False,header=True)
                    self.items_neighbors = None

            elif type_name == "items_hotest":

                if self.items_hotest is None:
                    print("ERROR: save data is None, have you called the runAlgorithmItemRateHotest() ?")
                    ret = -1
                else:

                    if not os.path.exists(file_path):
                        with open(file_path, "w") as f:
                            f.write("rec_id,compute_date,compute_timestamp,items_count,item_list\n")
                        rec_id = 1
                    else:
                        with open(file_path, "r") as f:
                            rec_id = len(f.readlines())
                    with open(file_path, "a") as f:

                        self.items_hotest = [rec_id]+self.items_hotest
                        new_line = ",".join([str(x) for x in self.items_hotest])
                        f.write(new_line+"\n")
                    self.items_hotest = None

            elif type_name == "final_rec":
                if self.final_rec is None:
                    print("ERROR: save data is None, have you called the runTrain() ?")
                    ret = -1
                else:
                    self.final_rec.to_csv(file_path, index=False,header=True)
                    self.final_rec = None

            else:
                print("ERROR: Unknown type_name: ", type_name)
                print("-Note: type_name must be 'items_user_favorite' or 'items_neighbors'.")
                ret = -1

        elif data_type == "database":
            print("ERROR: database have not completed.")
            ret = -2

        else:
            print("ERROR: Unknown data_type: ", data_type)
            print("-Note: data_type must be 'csv' or 'database'.")
            ret = -1

        return ret


    def saveTrainTimeLog(self, cost_time, return_signal, save_name):
        compute_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        compute_timestamp = (int(round(time.time() * 1000)))    #毫秒级时间戳


        log_path = os.path.join(self.dir_csv_logs,self.log_runTrain_time_csv_name)
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("log_id,compute_date,compute_timestamp,cost_time,return_signal,version,save_name\n")
            log_id = 1
        else:
            with open(log_path, "r") as f:
                log_id = len(f.readlines())

        with open(log_path, "a") as f:
            line = [log_id,compute_date,compute_timestamp,cost_time,return_signal,self.version,save_name]
            line = ",".join([str(x) for x in line])
            f.write(line+"\n")
        return 0


    def saveManualSet(self, item_list_str, update_user='admin', remark=''):
        update_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        update_timestamp = (int(round(time.time() * 1000)))    #毫秒级时间戳

        manual_set_path = os.path.join(self.dir_csv_recs, self.recs_item_manual_set_csv_name)
        if not os.path.exists(manual_set_path):
            with open(manual_set_path, "w") as f:
                f.write("log_id,item_list,update_date,update_timestamp,update_user,remark\n")
            log_id = 1
        else:
            with open(manual_set_path, "r") as f:
                log_id = len(f.readlines())

        with open(manual_set_path, "a") as f:
            line = [log_id,item_list_str,update_date,update_timestamp,update_user,remark]
            line = ",".join([str(x) for x in line])
            f.write(line+"\n")
        return 0

    def saveUserBehavior(self, user_id, item_id, rate, rate_date, rate_timestamp):
        user_rate_path = os.path.join(self.dir_csv_logs, self.log_user_rate_csv_name)
        if not os.path.exists(user_rate_path):
            with open(user_rate_path, "w") as f:
                f.write("log_id,user_id,item_id,rate,rate_date,rate_timestamp\n")
            log_id = 1
        else:
            with open(user_rate_path, "r") as f:
                log_id = len(f.readlines())

        with open(user_rate_path, "a") as f:
            line = [log_id,user_id,item_id,rate,rate_date,rate_timestamp]
            line = ",".join([str(x) for x in line])
            f.write(line+"\n")
        return 0

    def saveUser(self, user_id, user_name, update_date, update_timestamp):
        users_path = os.path.join(self.dir_csv_data, self.users_csv_name)
        if not os.path.exists(users_path):
            with open(users_path, "w") as f:
                f.write("user_id, user_name, update_date, update_timestamp\n")
        
        df = pd.read_csv(users_path)
        if user_id in df['user_id'].values:
            df.loc[df[df['user_id']==user_id].index,['user_name','update_date','update_timestamp']] = [user_name, update_date, update_timestamp]
        else:
            df = df.append([{'user_id':user_id,
                            'user_name':user_name,
                            'update_date':update_date,
                            'update_timestamp':update_timestamp}], ignore_index=True)

        df.to_csv(users_path, index=False,header=True)
        return 0

    def saveItem(self, item_id, item_name, update_date, update_timestamp):
        items_path = os.path.join(self.dir_csv_data, self.items_csv_name)
        if not os.path.exists(items_path):
            with open(items_path, "w") as f:
                f.write("item_id, item_name, update_date, update_timestamp\n")
        
        df = pd.read_csv(items_path)
        if item_id in df['item_id'].values:
            df.loc[df[df['item_id']==item_id].index,['item_name','update_date','update_timestamp']] = [item_name, update_date, update_timestamp]
        else:
            df = df.append([{'item_id':item_id,
                            'item_name':item_name,
                            'update_date':update_date,
                            'update_timestamp':update_timestamp}], ignore_index=True)

        df.to_csv(items_path, index=False,header=True)
        return 0

    ################### Run Algorithm #########################
    def runAlgorithmUserRate(self, algorithm = "SVD"):
        """
        Recommendation based on users rates

        NormalPredicto
        KNNBasic
        KNNWithMeans
        KNNWithZScore
        KNNBaseline
        SVD
        SVDpp
        NMF
        CoClustering
        
        """
        if algorithm == "NormalPredictor":
            self.algorithm_user_rate = NormalPredictor()
        elif algorithm == "KNNBasic":
            self.algorithm_user_rate = KNNBasic(k=40, min_k=1)
        elif algorithm == "KNNWithMeans":
            self.algorithm_user_rate = KNNWithMeans(k=40, min_k=1)
        elif algorithm == "KNNWithZScore":
            self.algorithm_user_rate = KNNWithZScore(k=40, min_k=1)
        elif algorithm == "KNNBaseline":
            self.algorithm_user_rate = KNNBaseline(k=40, min_k=1)
        elif algorithm == "SVD":
            self.algorithm_user_rate = SVD()
        elif algorithm == "SVDpp":
            self.algorithm_user_rate = SVDpp()
        elif algorithm == "NMF":
            self.algorithm_user_rate = NMF()
        elif algorithm == "CoClustering":
            self.algorithm_user_rate = CoClustering()
        elif algorithm == "random":
            self.algorithm_user_rate = RandomPredictor()
        elif algorithm == "FM":
            self.algorithm_user_rate = FM()
        elif algorithm == "MyAlgor":
            self.algorithm_user_rate = MyAlgor()
        else:
            print("ERROR: Unknown algorithm: ", algorithm)
            print("-Note: data_type must be 'NormalPredictor' or 'SVD' or ... .")
            return -1

        if self.user_rate_trainset is None:
            print("ERROR: train data is None, have you called the loadData() ?")
            return -1
        else:
            self.algorithm_user_rate.fit(self.user_rate_trainset)

        if self.use_test_split:
            self.user_rate_predictions = self.algorithm_user_rate.test(self.user_rate_testset)
            # Compute RMSE
            self.user_rate_acc_rmse = accuracy.rmse(self.user_rate_predictions)
        else:
            self.user_rate_testset = (self.user_rate_trainset.build_anti_testset())
            self.user_rate_predictions = self.algorithm_user_rate.test(self.user_rate_testset)


        print("Finish runAlgorithmUserRate.")
        return 0


    def runAlgorithmItemRate(self, algorithm = "KNNBaseline"):
        """
        Recommendation based on items rates
        """
        sim_options = {'name': 'pearson_baseline', 'user_based': False}
        if algorithm ==  "KNNBaseline":
            self.algorithm_item_rate = KNNBaseline(k=40, min_k=1, sim_options=sim_options)
        elif algorithm == "KNNBasic":
            self.algorithm_item_rate = KNNBasic(k=40, min_k=1, sim_options=sim_options)
        elif algorithm == "KNNWithMeans":
            self.algorithm_item_rate = KNNWithMeans(k=40, min_k=1, sim_options=sim_options)
        elif algorithm == "KNNWithZScore":
            self.algorithm_item_rate = KNNWithZScore(k=40, min_k=1, sim_options=sim_options)


        else:
            print("ERROR: Unknown algorithm: ", algorithm)
            print("-Note: data_type must be 'KNNBaseline' or 'KNNWithMeans' or ... .")
            return -1

        if self.user_rate_trainset is None:
            print("ERROR: train data is None, have you called the loadData() ?")
            return -1
        else:
            self.algorithm_item_rate.fit(self.user_rate_trainset)


            item_rate_predictions = self.algorithm_item_rate.test(self.user_rate_testset)
            # Compute RMSE
            self.user_rate_acc_rmse = accuracy.rmse(item_rate_predictions)

        print("Finish runAlgorithmItemRate.")
        return 0


    def runAlgorithmItemFeature(self):
        """
        Recommendation based on items features
        """
        pass
        

    def runAlgorithmItemRateHotest(self, top_n = 20, top_n_backup_list_len = 20):
        """
        top_n: number of hotest items
        top_n_backup_list_len: first choose top_n+top_n_backup_list_len appear 
                        most times items,then choose top_n best scores items

        1st edition: choose item which has most rate and highest score
        2st edition(to do): compare with history so do not need to repeat compute  
                             / consider time
        """
        #print(self.user_rate_data.df.shape) #(100000, 3)

        # find appear most times items
        #res_top_n = [0,0,0]*(top_n+top_n_backup_list_len)
        if top_n<1:
            print("ERROR: top_n must be greater than 1, get: ", type(top_n))
            return -1

        if top_n_backup_list_len<0:
            print("ERROR: top_n_backup_list_len must be greater than 0, get: ", type(top_n_backup_list_len))
            return -1


        dict_item_times_score = {}

        for row in self.user_rate_data.df.itertuples(index=True, name='Pandas'):
            item_id = getattr(row, "item_id")
            item_rate = getattr(row, "rate")
            
            if item_id not in dict_item_times_score:
                dict_item_times_score[item_id] = [1, item_rate]
            else:
                dict_item_times_score[item_id][1] = (dict_item_times_score[item_id][1]*dict_item_times_score[item_id][0]+item_rate)/(dict_item_times_score[item_id][0]+1)
                dict_item_times_score[item_id][0] += 1

        #print(dict_item_times_score[242])

        top_n_backup_list = sorted(dict_item_times_score.items(), 
                                key=lambda x: x[1][0], reverse=True)[:top_n+top_n_backup_list_len]
        #print(top_n_backup_list)
        top_n_list = sorted(top_n_backup_list, 
                                key=lambda x: x[1][1], reverse=True)[:top_n]

        top_n_list = [[x[0], x[1][0], round(x[1][1],3)] for x in top_n_list]
        items_count = len(top_n_list)
        compute_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        compute_timestamp = (int(round(time.time() * 1000)))    #毫秒级时间戳
        top_n_list = ["|".join([str(y) for y in x]) for x in top_n_list]
        top_n_list_str = ";".join(top_n_list)
        self.items_hotest = [compute_date, compute_timestamp, items_count, top_n_list_str]
        """
        top_n_list: [compute date, compute timestamp, items count, items list]
        items list: item id, item comment count, average score;...
        """
        # df_titles = ['log_id', 'update_time', 'result'] 
        # self.items_hotest = pd.DataFrame(top_n_list, columns=df_titles)
        print("Finish runAlgorithmItemRateHotest.")
        return 0


    ################### Get result #########################
    def getUserTopNFavoriteItems(self, top_n = 10):
        # Predict ratings for all pairs (u, i) that are NOT in the training set.
        def _get_top_n(predictions, n=10):
            '''Return the top-N recommendation for each user from a set of predictions.

            Args:
                predictions(list of Prediction objects): The list of predictions, as
                    returned by the test method of an algorithm.
                n(int): The number of recommendation to output for each user. Default
                    is 10.

            Returns:
            A dict where keys are user (raw) ids and values are lists of tuples:
                [(raw item id, rating estimation), ...] of size n.
            '''

            # First map the predictions to each user.
            top_n_res = defaultdict(list)
            for uid, iid, true_r, est, _ in predictions:
                top_n_res[uid].append((iid, est))

            # Then sort the predictions for each user and retrieve the k highest ones.
            for uid, user_ratings in top_n_res.items():
                user_ratings.sort(key=lambda x: x[1], reverse=True)
                top_n_res[uid] = user_ratings[:n]

            return top_n_res

        if self.user_rate_predictions is None:
            print("ERROR: predictions data is None, have you called the runAlgorithmItemRate() ?")
            return -1
        else:
            top_n_res = _get_top_n(self.user_rate_predictions, n=top_n)


        items_user_favorite_list = []
        for uid, user_ratings in top_n_res.items():
            # print(uid, [iid for (iid, _) in user_ratings])
            one_user_top_n = ";".join([str(iid) for (iid, _) in user_ratings])
            items_user_favorite_list.append([uid, one_user_top_n])


        df_titles = ['user_id','item_list']
        self.items_user_favorite = pd.DataFrame(items_user_favorite_list, columns=df_titles)

        self.items_user_favorite = self.items_user_favorite.sort_values(["user_id"])



    def getItemNeighbors(self, top_n = 10):
        """
        If use this func, use_test_split must be False.
        """
        
        items_neighbors_list = []
        for i in range(self.count_items):

            inner_id = self.algorithm_item_rate.trainset.to_inner_iid(i+1)

            items_neighbors = self.algorithm_item_rate.get_neighbors(inner_id, k=top_n)
            
            items_neighbors = [self.algorithm_item_rate.trainset.to_raw_iid(inner_id)
                                   for inner_id in items_neighbors]
            one_items_neighbors = ";".join([str(x) for x in items_neighbors])
            items_neighbors_list.append([i+1,one_items_neighbors])

        df_titles = ['item_id','item_list']
        self.items_neighbors = pd.DataFrame(items_neighbors_list, columns=df_titles)

        self.items_neighbors = self.items_neighbors.sort_values(["item_id"])



    def getUserNeighbors(self, top_n = 10):
        pass





    ################### Get result By User ID ######################### 
    def getRecommendByUserIDFromManualSet(self, top_n = 10, data_type = "csv"):
        # 人为指定的冷启动物品
        if data_type == "csv":
            file_path = os.path.join(self.dir_csv_recs, self.recs_item_manual_set_csv_name)
            if not os.path.exists(file_path):
                print("ERROR: items_manual_set file does not exist.")
                return []
            items_manual_set = pd.read_csv(file_path)

            item_list = items_manual_set.iloc[-1]['item_list']
            item_list = item_list.strip().split(';')
            rec_ids = [int(x.split('|')[0]) for x in item_list]
            #rec_ids = np.array(items_manual_set['item_id']).tolist()
            #rec_names = [self.items.loc[item_id-1, 'item_name'] for item_id in rec_ids]

        elif data_type == "database":
            pass

        else: 
            print("ERROR: Unknown data_type: ", data_type)
            print("-Note: data_type must be 'csv' or 'database'.")
            return []

        return rec_ids[:top_n]


    def getRecommendByUserIDFromHotest(self, top_n = 10, data_type = "csv"):
        if data_type == "csv":
            file_path = os.path.join(self.dir_csv_recs,self.recs_item_hotest_csv_name)
            if not os.path.exists(file_path):
                print("ERROR: items_hotest file does not exist.")
                return []
            items_hotest = pd.read_csv(file_path)

            item_list = items_hotest.iloc[-1]['item_list']
            item_list = item_list.strip().split(';')
            rec_ids = [int(x.split('|')[0]) for x in item_list]
            #rec_ids = np.array(items_hotest['item_id']).tolist()
            #rec_names = [self.items.loc[item_id-1, 'item_name'] for item_id in rec_ids]

        elif data_type == "database":
            pass

        else:
            print("ERROR: Unknown data_type: ", data_type)
            print("-Note: data_type must be 'csv' or 'database'.")
            return []

        return rec_ids[:top_n]

    def getRecommendByUserIDFromUserFavorite(self, user_id, top_n = 10, data_type = "csv"):
        if data_type == "csv":
            file_path = os.path.join(self.dir_csv_recs,self.recs_item_user_favorite_csv_name)
            if not os.path.exists(file_path):
                print("ERROR: items_user_favorite file does not exist.")
                return []
            items_user_favorite = pd.read_csv(file_path)
            
            if user_id in items_user_favorite['user_id'].values:
                rec_ids = np.array(items_user_favorite[items_user_favorite['user_id']==user_id])[0].tolist()[-1]
                rec_ids = [int(x) for x in rec_ids.strip().split(';')]
            else:
                rec_ids = []

        elif data_type == "database":
            pass

        else:
            print("ERROR: Unknown data_type: ", data_type)
            print("-Note: data_type must be 'csv' or 'database'.")
            return []

        return rec_ids[:top_n]
        
    def getRecommendByUserIDFromItemNeighbors(self, item_id, top_n = 10, data_type = "csv"):
        if data_type == "csv":
            file_path = os.path.join(self.dir_csv_recs,self.recs_item_neighbor_csv_name)
            if not os.path.exists(file_path):
                print("ERROR: items_neighbors file does not exist.")
                return []
            items_neighbors = pd.read_csv(file_path)

            #name = self.items[self.items['item_id']==item_id].at[item_id-1,'item_name']
            rec_ids = np.array(items_neighbors[items_neighbors['item_id']==item_id])[0].tolist()[-1]
            rec_ids = [int(x) for x in rec_ids.strip().split(';')]
            #rec_names = [self.items.loc[item_id-1, 'item_name'] for item_id in rec_ids]
            
        elif data_type == "database":
            pass

        else:
            print("ERROR: Unknown data_type: ", data_type)
            print("-Note: data_type must be 'csv' or 'database'.")
            return []

        return rec_ids[:top_n]



    def filterItemsUserConsumed(self, rec_ids, user_id, time_range_day = 30):
        # 过滤用户消费过的物品, 在time_range_day内避免重复推荐，单位：天
        
        # 1.根据userid查询user消费过的物品
        file_path = os.path.join(self.dir_csv_logs,self.log_user_rate_csv_name)
        if not os.path.exists(file_path):
            print("ERROR: items_neighbors file does not exist.")
            return -1
        log_user_rate = pd.read_csv(file_path)

        df_titles = ["item_id","rate_timestamp"]
        comsumed_items = log_user_rate[log_user_rate["user_id"] == user_id][df_titles]
        comsumed_items = comsumed_items.drop_duplicates(["item_id"],keep='last')
        comsumed_items_id_set = set(comsumed_items["item_id"].values)
        #print(comsumed_items_id_set)
        #print(rec_ids)

        # 2.从rec_ids 删掉
        now_timestamp = (int(round(time.time() * 1000))) #毫秒级时间戳
        time_range_ms = time_range_day*24*60*60*1000

        filtered_items = []
        for rec_id in rec_ids:
            item_id = rec_id[0]
            if item_id not in comsumed_items_id_set:
                filtered_items.append(rec_id)
            else:
                rate_timestamp = comsumed_items[comsumed_items["item_id"] == item_id]["rate_timestamp"].values
                if (now_timestamp-rate_timestamp)>time_range_ms:
                    filtered_items.append(rec_id)

        # 3.返回
        return filtered_items

    def filterItemsDuplicate(self, item_list):
        # 过滤推荐列表中重复的物品
        items = []
        new_item_list = []

        for item in item_list:
            item_name = item[0]
            if item_name not in items:
                items.append(item_name)
                new_item_list.append(item)


        return new_item_list


    def getFinalRecommendation(self, top_n = 10):
        """
        Docs:combine all func AND add recommend result
        userCF_rec: 和你口味相似的用户也喜欢
        hotest_rec: 你可能喜欢的热门物品
        manual_rec: 编辑精选
        item_rec: 和你喜欢的xx类似的物品
        """

        # total = 1.0
        def _getOneRecommendByUserID(user_id):
            ratio_manual_rec = 0.5
            ratio_hotest_rec = 0.2
            ratio_userCF_rec = 0.3


            manual_rec = self.getRecommendByUserIDFromManualSet(top_n=top_n)
            hotest_rec = self.getRecommendByUserIDFromHotest(top_n=top_n)
            userCF_rec = self.getRecommendByUserIDFromUserFavorite(user_id=user_id,top_n=top_n)

            len_manual_rec = len(manual_rec)
            len_hotest_rec = len(hotest_rec)
            len_userCF_rec = len(userCF_rec)

            count_manual_rec = 0
            count_hotest_rec = 0
            count_userCF_rec = 0

            final_rec = []
            while (count_manual_rec<len_manual_rec or count_hotest_rec<len_hotest_rec or count_userCF_rec<len_userCF_rec):
                rd = random.random()
                cate = ''
                if rd<ratio_manual_rec:
                    if count_manual_rec<len_manual_rec:
                        cate = 'manual_rec'
                    elif count_hotest_rec<len_hotest_rec:
                        cate = 'hotest_rec'
                    else:
                        cate = 'userCF_rec'
                elif rd<(ratio_manual_rec+ratio_hotest_rec):
                    if count_hotest_rec<len_hotest_rec:
                        cate = 'hotest_rec'
                    elif count_manual_rec<len_manual_rec:
                        cate = 'manual_rec'
                    else:
                        cate = 'userCF_rec'
                else:
                    if count_userCF_rec<len_userCF_rec:
                        cate = 'userCF_rec'
                    elif count_manual_rec<len_manual_rec:
                        cate = 'manual_rec'                    
                    else:
                        cate = 'hotest_rec'

                if cate == 'manual_rec':
                    final_rec.append([manual_rec[count_manual_rec], "manual_rec"])
                    count_manual_rec+=1
                elif cate == 'hotest_rec':
                    final_rec.append([hotest_rec[count_hotest_rec], "hotest_rec"])
                    count_hotest_rec+=1
                else:
                    final_rec.append([userCF_rec[count_userCF_rec], "userCF_rec"])
                    count_userCF_rec+=1

            final_rec = self.filterItemsDuplicate(final_rec)
            final_rec = self.filterItemsUserConsumed(final_rec,user_id)

            final_rec = [[self.items.loc[x[0]-1, 'item_name'], x[1]] for x in final_rec]
            # id to name

            return final_rec

        final_recs = []
        for user_id in self.users["user_id"].values:
            one_rec = _getOneRecommendByUserID(user_id)
            one_rec_str = ["|".join(x) for x in one_rec]
            one_rec_str = ";".join(one_rec_str)
            final_recs.append([user_id,one_rec_str])


        df_titles = ['user_id','item_list']
        self.final_rec = pd.DataFrame(final_recs, columns=df_titles)

        return 0



    ######################## Read ########################
    def readLastTimeLog(self):
        #df type
        log_path = os.path.join(self.dir_csv_logs, self.log_runTrain_time_csv_name)
        if not os.path.exists(log_path):
            print("ERROR: time log file does not exist.")
            return -1
        time_log = pd.read_csv(log_path)
        self.last_train_time_log = time_log.iloc[-1]


    def readFinalRec(self):
        if self.last_train_time_log is None:
            self.readLastTimeLog()

        #print(self.last_train_time_log)
        save_name = self.last_train_time_log["save_name"]

        self.final_rec = pd.read_csv(os.path.join(self.dir_csv_recs,save_name))

    ################### Open API #########################

    #####  run 
    def runTest(self, test_type = "user_rate", algorithm = "SVD"):
        # 目前支持user-rate (CF)算法的RMSE评估、覆盖率的评估 
        """
        For movie 10000 test(RMSE):
        NormalPredictor: 1.5186  0.21s
        KNNBasic:        0.9872  2.96s
        KNNWithMeans:    0.9629  3.18s
        KNNWithZScore:   0.9631  3.38s
        KNNBaseline:     0.9419  3.72s
        SVD:             0.9475  3.80s
        SVDpp:           0.9286  141.13s
        NMF:             0.9741  4.22s
        CoClustering:    0.9695  1.63s
        """
        if test_type == "user_rate":
            print("Test: algorithm = ",algorithm)
            self.loadData(file_path=os.path.join(self.dir_csv_logs, self.log_user_rate_csv_name),
                     use_test_split=True)
            t = time.time()
            self.runAlgorithmUserRate(algorithm)
            print("time: ", time.time() - t)

            return self.user_rate_acc_rmse


        elif test_type == "coverage":
            #compute coverage = items in items_user_favorite/items in user_rate_log
            # 1. get item set in rec
            file_path = os.path.join(self.dir_csv_recs, self.recs_item_user_favorite_csv_name)
            if not os.path.exists(file_path):
                print("ERROR: items_user_favorite file does not exist.")
                return -1
            items_user_favorite = pd.read_csv(file_path)
            rec_items = ";".join(items_user_favorite['item_list'].values)
            rec_items = rec_items.strip().split(";")
            rec_items = [int(x) for x in rec_items]
            rec_items = set(rec_items)
            rec_items_num = len(rec_items)
            #print(rec_items_num)
            # 2. get all item ids in items.csv
            file_path = os.path.join(self.dir_csv_logs, self.log_user_rate_csv_name)
            if not os.path.exists(file_path):
                print("ERROR: log_user_rate_csv file does not exist.")
                return -1
            total_items = pd.read_csv(file_path)
            total_items = [int(x) for x in total_items['item_id'].values]
            #print(len(total_items))
            total_items = set(total_items)
            #print(len(total_items))
            total_items_num = len(total_items)
            
            # 3.compute
            coverage = rec_items_num*1.0/total_items_num
            return coverage

        else:
            print("ERROR: Unknown test_type: ", test_type)
            print("-Note: data_type must be 'user_rate' or 'coverage'.")
            return -1


    def runTrain(self, top_n = 10, algorithm = "SVD"):
        #return 0 means success
        ret = 0

        start_time = time.time()

        self.loadData(file_path=os.path.join(self.dir_csv_logs, self.log_user_rate_csv_name))

        self.runAlgorithmUserRate(algorithm = algorithm)
        self.getUserTopNFavoriteItems()
        self.saveDataTrained(file_path = os.path.join(self.dir_csv_recs, self.recs_item_user_favorite_csv_name), 
                    type_name = "items_user_favorite")

        self.runAlgorithmItemRate()
        self.getItemNeighbors()
        self.saveDataTrained(file_path = os.path.join(self.dir_csv_recs, self.recs_item_neighbor_csv_name), 
                    type_name = "items_neighbors")

        self.runAlgorithmItemRateHotest()
        self.saveDataTrained(file_path = os.path.join(self.dir_csv_recs, self.recs_item_hotest_csv_name), 
                    type_name = "items_hotest")


        print("Finsh run algorithm, start getFinalRecommendation.")
        self.getFinalRecommendation(top_n)
        save_name = self.recs_final_csv_name_pre+("%d_%d_%d.csv" % (datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day))
        self.saveDataTrained(file_path = os.path.join(self.dir_csv_recs, save_name),
                    type_name = "final_rec")
        
        cost_time = time.time() - start_time
        # Finish, save time log
        self.saveTrainTimeLog(cost_time, ret, save_name)


        ### release dataframe and keep longtime data
        self.user_rate_data = None
        self.user_rate_trainset = None
        self.user_rate_testset = None
        self.user_rate_predictions = None
        self.users = None
        self.items = None
        self.items_user_favorite = None
        self.items_neighbors = None
        self.items_hotest = None


        #self.readLastTimeLog()

        #self.readFinalRec()

        return ret


    #####  get 
    def getLastTrainTime(self):
        if self.last_train_time_log is None:
            self.readLastTimeLog()
        return self.last_train_time_log['compute_date']


    def getRecommendByUserID(self, user_id, top_n = 10):

        if self.final_rec is None:
            self.readFinalRec()

        final_rec_list = self.final_rec[self.final_rec['user_id']==user_id]['item_list'].values[0]

        # top_n
        final_rec_list = ";".join(final_rec_list.strip().split(';')[:top_n])

        return final_rec_list

    def getRecommendByUserIDRealTime(self, user_id):
        # 实时推荐
        pass
        


    #####  update 
    def updateUser(self, user_log):
        # 新增用户
        # '{"user_id":"10000","user_name":"Test2","update_date":"2020-01-08 13:40:32","update_timestamp":"1578462032419"}'
        if isinstance(user_log,dict):
            pass
        elif isinstance(user_log,str):
            user_log=json.loads(user_log)
        else:
            print("ERROR: user_log type unsupported!")
            print("-Note: must in dict or str!")
            return -1

        user_id = int(user_log['user_id'])
        user_name = user_log['user_name']
        update_date = user_log['update_date']
        update_timestamp = int(user_log['update_timestamp'])

        self.saveUser(user_id, user_name, update_date, update_timestamp)
        return 0

    def updateItem(self, item_log):
        # 新增物品
        # '{"item_id":"10000","item_name":"Test","update_date":"2020-01-08 13:40:32","update_timestamp":"1578462032419"}'
        if isinstance(item_log,dict):
            pass
        elif isinstance(item_log,str):
            item_log=json.loads(item_log)
        else:
            print("ERROR: item_log type unsupported!")
            print("-Note: must in dict or str!")
            return -1

        item_id = int(item_log['item_id'])
        item_name = item_log['item_name']
        update_date = item_log['update_date']
        update_timestamp = int(item_log['update_timestamp'])

        self.saveItem(item_id, item_name, update_date, update_timestamp)
        return 0

    def updateUserBehavior(self, user_behavior_log):
        # 处理用户行为日志，保存更新到现有数据表
        # '{"user_id":"1","item_id":"1","rate":"1","rate_date":"2020-01-08 13:40:32","rate_timestamp":"1578462032419"}'
        if isinstance(user_behavior_log,dict):
            pass
        elif isinstance(user_behavior_log,str):
            user_behavior_log=json.loads(user_behavior_log)
        else:
            print("ERROR: user_behavior_log type unsupported!")
            print("-Note: must in dict or str!")
            return -1
        user_id = int(user_behavior_log['user_id'])
        item_id = int(user_behavior_log['item_id'])
        rate = int(user_behavior_log['rate'])
        rate_date = user_behavior_log['rate_date']
        rate_timestamp = int(user_behavior_log['rate_timestamp'])

        self.saveUserBehavior(user_id, item_id, rate, rate_date, rate_timestamp)
        return 0

    def updateManualSet(self, update_item_log):
        # 上传手动推荐的物品
        # '{"item_list":"1,2,3,4,5","update_user":"admin","remark":"test"}'
        if isinstance(update_item_log,dict):
            pass
        elif isinstance(update_item_log,str):
            update_item_log=json.loads(update_item_log)
        else:
            print("ERROR: update_item_log type unsupported!")
            print("-Note: must in dict or str!")
            return -1

        item_list = update_item_log['item_list'].strip().split(",")
        item_list_str = ';'.join([str(x) for x in item_list])
        update_user = update_item_log['update_user']
        remark = update_item_log['remark']
        
        self.saveManualSet(item_list_str, update_user, remark)
        return 0




if __name__ == "__main__":


    # for train
    rs_test = YXRecommenderSystem(dir_csv_data="/Users/fire/A/workshop/recom/surprisedemo/my_data/surprisedata/data/",
                                  dir_csv_recs="/Users/fire/A/workshop/recom/surprisedemo/my_data/surprisedata/recs/",
                                  dir_csv_logs="/Users/fire/A/workshop/recom/surprisedemo/my_data/surprisedata/logs/")


    #       run 
    t = time.time()
    #print(rs_test.runTest(test_type = "user_rate", algorithm = "FM"))
    rs_test.runTrain(algorithm = "FM")
    ''
    print("cost time1:", time.time() - t)
    #b
    #       get
    #res = rs_test.filterItemsUserConsumed([1,2,3,12],186,30)
    #print("res:", res)

    t = time.time()
    print(rs_test.getRecommendByUserID(user_id = 1, top_n=10))
    print(rs_test.getLastTrainTime())
    print("cost time2:", time.time() - t)



    #     update
    # update_item_log_str = '{"item_list":"1,2,3,4,5","update_user":"admin","remark":"test"}'
    # update_item_log=json.loads(update_item_log_str)
    # rs_test.updateManualSet(update_item_log)
    
    # user_behavior_log_str = '{"user_id":"1","item_id":"1","rate":"1","rate_date":"2020-01-08 13:40:32","rate_timestamp":"1578462032419"}'
    # user_behavior_log = json.loads(user_behavior_log_str)
    # rs_test.updateUserBehavior(user_behavior_log)

    # user_log_str = '{"user_id":"10000","user_name":"Test2","update_date":"2020-01-08 13:40:32","update_timestamp":"1578462032419"}'
    # user_log = json.loads(user_log_str)
    # rs_test.updateUser(user_log)

    # item_log_str = '{"item_id":"10000","item_name":"Test","update_date":"2020-01-08 13:40:32","update_timestamp":"1578462032419"}'
    # item_log = json.loads(item_log_str)
    # rs_test.updateItem(item_log)

