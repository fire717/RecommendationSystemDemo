#coding:utf-8
# @Fire 
# Start from 2020/1/6
# Test code for recommender_system.py
import recommender_system as yxrs
import os
"""
测试目标：
1.若在程序启动时不能避免的错误，应该有明确的错误提示，比如xx文件、路径不存在
2.若在运行时的错误，应该返回错误提示，同时不导致主程序崩溃

"""

class YXRecommenderSystemTest:
    """

    """
    def __init__(self):
        self.rs_test = None

    def test_init(self):
        #__init__(self, dir_csv_data, dir_csv_recs, dir_csv_logs)
        ###################
        test_cases = [[{"dir_csv_data":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/data/",
        "dir_csv_recs":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/recs/",
        "dir_csv_logs":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/",},0],
                    [{"dir_csv_data":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/data1/",
        "dir_csv_recs":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/recs/",
        "dir_csv_logs":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/",},FileNotFoundError],
                    [{"dir_csv_data":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/data/",
        "dir_csv_recs":"C:/Users/yw/Desktop/recommender/my_data/suprisedata1/recs/",
        "dir_csv_logs":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/",},FileNotFoundError],
                    [{"dir_csv_data":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/data/",
        "dir_csv_recs":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/recs/",
        "dir_csv_logs":"C:/Users/yw/Desktop/recommender/my_data1/suprisedata/logs/",},FileNotFoundError],
                    [{"dir_csv_data1":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/data/",
        "dir_csv_recs":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/recs/",
        "dir_csv_logs":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/",},TypeError],
                    [{"dir_csv_data":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/data/",
        "dir_csv_recs1":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/recs/",
        "dir_csv_logs":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/",},TypeError],
                    [{"dir_csv_data":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/data/",
        "dir_csv_recs":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/recs/",
        "dir_csv_logs1":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/",},TypeError],
                    [{},TypeError]]
        ###################

        res_passed = 0
        for i in range(len(test_cases)):
            try:
                res = yxrs.YXRecommenderSystem(**test_cases[i][0])
                if res:
                    res_passed+=1
                else:
                    print(res)
            except Exception as e:
               # print(e,type(e),test_cases[i][1])
                if type(e)==test_cases[i][1]:
                    res_passed+=1
                else:
                    print(type(e),e)

        print("-----------------------------------------")
        print("Done testInit, total:%d, passed:%d" % (len(test_cases), res_passed))
        print()
        self.rs_test = yxrs.YXRecommenderSystem(**test_cases[0][0])


    def test_loadData(self):
        #loadData(self, file_path, data_type = "csv", rating_scale=(1, 5), 
        #        use_test_split = False, test_split_ratio = 0.2, file_encode = 'utf-8')
        ###################
        test_cases = [
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    },0],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "data_type": "database"},-2],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "data_type" :"csvv"},-1],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "rating_scale": (1, 4)},0],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "rating_scale" :[1, 5]},-1],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "rating_scale" :5},-1],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "rating_scale" :(-1, 5)},0],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "use_test_split": "false"},-1],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "use_test_split": True,
                    "test_split_ratio": 0},ValueError],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "use_test_split": True,
                    "test_split_ratio": 1},0],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "use_test_split": True,
                    "test_split_ratio" :1.0},0],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "use_test_split": True,
                    "test_split_ratio": 1000000},ValueError],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "file_encode": 'gbk'},0],
                    [{"file_path":"C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/log_user_rates.csv",
                    "file_encode": 'utf-88'},LookupError],
                    [{},TypeError]
                    ]
        ###################
        self.rs_test = yxrs.YXRecommenderSystem(dir_csv_data="C:/Users/yw/Desktop/recommender/my_data/suprisedata/data/",
                        dir_csv_recs="C:/Users/yw/Desktop/recommender/my_data/suprisedata/recs/",
                        dir_csv_logs="C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/")

        res_passed = 0
        for i in range(len(test_cases)):
            try:
                res = self.rs_test.loadData(**test_cases[i][0])
                #print(res)
                if res == test_cases[i][1]:
                    res_passed+=1
            except Exception as e:
                #print(e)
                if type(e)==test_cases[i][1]:
                    res_passed+=1
                else:
                    print(type(e),e)

        print("-----------------------------------------")
        print("Done testInit, total:%d, passed:%d" % (len(test_cases), res_passed))
        print()

    def test_saveDataTrained(self):
        #loadData(self, file_path, data_type = "csv", rating_scale=(1, 5), 
        #        use_test_split = False, test_split_ratio = 0.2, file_encode = 'utf-8')
        ###################
        test_cases = [
                    [{"file_path":"./save.csv",
                    "type_name":"items_user_favorite"
                    },-1],
                    [{"file_path":"./save.csv",
                    "type_name":"items_user_favorite1"
                    },-1],
                    [{"file_path":"./save.csv",
                    "type_name":"items_user_favorite",
                    "data_type":"csvv"
                    },-1],
                    [{},TypeError]
                    ]
        ###################
        self.rs_test = yxrs.YXRecommenderSystem(dir_csv_data="C:/Users/yw/Desktop/recommender/my_data/suprisedata/data/",
                        dir_csv_recs="C:/Users/yw/Desktop/recommender/my_data/suprisedata/recs/",
                        dir_csv_logs="C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/")

        res_passed = 0
        for i in range(len(test_cases)):
            try:
                res = self.rs_test.saveDataTrained(**test_cases[i][0])
                #print(res)
                if res == test_cases[i][1]:
                    res_passed+=1
            except Exception as e:
                #print(e)
                if type(e)==test_cases[i][1]:
                    res_passed+=1
                else:
                    print(type(e),e)

        print("-----------------------------------------")
        print("Done testInit, total:%d, passed:%d" % (len(test_cases), res_passed))
        print()

    def test_runAlgorithmItemRateHotest(self):
        #loadData(self, file_path, data_type = "csv", rating_scale=(1, 5), 
        #        use_test_split = False, test_split_ratio = 0.2, file_encode = 'utf-8')
        ###################
        test_cases = [
                    [{},0],
                    [{"top_n":0,
                    },-1],
                    [{"top_n":1,
                    },0],
                    [{"top_n":-1,
                    },-1],
                    [{"top_n":1000,
                    },0],
                    [{"top_n_backup_list_len":0
                    },0],
                    [{"top_n_backup_list_len":-1
                    },-1],
                    [{"top_n_backup_list_len":100
                    },0],
                    ]
        ###################
        self.rs_test = yxrs.YXRecommenderSystem(dir_csv_data="C:/Users/yw/Desktop/recommender/my_data/suprisedata/data/",
                        dir_csv_recs="C:/Users/yw/Desktop/recommender/my_data/suprisedata/recs/",
                        dir_csv_logs="C:/Users/yw/Desktop/recommender/my_data/suprisedata/logs/")
        
        self.rs_test.loadData(file_path=os.path.join(self.rs_test.dir_csv_logs, self.rs_test.log_user_rate_csv_name))
        #self.rs_test.runTrain()
        
        res_passed = 0
        for i in range(len(test_cases)):
            try:
                res = self.rs_test.runAlgorithmItemRateHotest(**test_cases[i][0])
                #print(self.rs_test.items_hotest)
                if res == test_cases[i][1]:
                    res_passed+=1
            except Exception as e:
                #print(e)
                if type(e)==test_cases[i][1]:
                    res_passed+=1
                else:
                    print(type(e),e)

        print("-----------------------------------------")
        print("Done testInit, total:%d, passed:%d" % (len(test_cases), res_passed))
        print()

    def runTest(self):
        #self.test_init()

        #self.test_loadData()

        #self.test_saveDataTrained()

        self.test_runAlgorithmItemRateHotest()




if __name__ == "__main__":
    test = YXRecommenderSystemTest()
    test.runTest()
