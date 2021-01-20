#coding:utf-8
# @Fire 
# Start from 2020/1/6
# Test code for recommender_system.py



with open("log_user_rates.csv","r") as f:
    lines = f.readlines()



with open("log_user_rates.csv","w") as f:
    for line in lines:
        if "user_id" in line:
            new_line = line.strip()+",rate_timestamp\n"
        else:
            new_line = line.strip()+",0\n"
        f.write(new_line)
