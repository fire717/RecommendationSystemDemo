#coding:utf-8
import os
import pandas as pd
import numpy as np

import yx_recommender_system.recommender_system as yxrs



if __name__ == "__main__":
    rs_test = yxrs.YXRecommenderSystem(dir_csv_data="C:/Users/yw/Desktop/recommender/my_data/suprisedata/data/",
                        dir_csv_recs="C:/Users/yw/Desktop/recommender/my_data/suprisedata/recs/",
                        dir_csv_logs="C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/")
    print(rs_test.csv_data_path)

    #       run 
    #rs_test.runTest()
    rs_test.runTrain()
    

    #       get
    #res = rs_test.filterItemsUserConsumed([1,2,3,12],186,30)
    #print("res:", res)

    t = time.time()
    print(rs_test.getRecommendByUserID(user_id = 1, top_n=10))
    print(rs_test.getLastTrainTime())
    print("cost time:", time.time() - t)



