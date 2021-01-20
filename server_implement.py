import numpy as np
import pandas as pd

import datetime
import time
import json

import yx_recommender_system.recommender_system as yxrs


rs = yxrs.YXRecommenderSystem(dir_csv_data="C:/Users/yw/Desktop/recommender/my_data/suprisedata/data/",
                        dir_csv_recs="C:/Users/yw/Desktop/recommender/my_data/suprisedata/recs/",
                        dir_csv_logs="C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/")


def getRecommendFromRS(user_id):

    res = rs.getRecommendByUserID(user_id = user_id)
    return res


def uploadUserBehavior(user_behavior_logs):

    #save to csv
    for user_behavior_log in user_behavior_logs:
        #print(user_behavior_log)
        res = rs.updateUserBehavior(user_behavior_log)
    return res



if __name__ == "__main__":
    print(getRecommendFromRS(1))

    user_behavior_log = '{"user_id":"1","item_id":"1","rate":"1","rate_date":"2020-01-15 13:40:32","rate_timestamp":"1578462032419"}'
    print(uploadUserBehavior(user_behavior_log))