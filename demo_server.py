#coding: utf-8
from flask import Flask
from flask import request

import json

from server_implement import getRecommendFromRS,uploadUserBehavior

app = Flask(__name__)


version = 'v1'






@app.route('/%s/user/<int:user_id>' % version, methods=['GET'])
def getRecommend(user_id):
    #print("111",type(user_id))  str
    assert request.method == 'GET'

    res = getRecommendFromRS(user_id)
    if isinstance(res, str):
        pass
    else:
        res = "400"
    return res

@app.route('/%s/log/<log_type>' % version, methods=['POST'])
def postUserBehavior(log_type):
    assert request.method == 'POST'

    print("get log_type:",log_type)
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    data_num = int(json_data['data_num'])
    json_data_list = []
    for i in range(data_num):
        json_data_list.append(json_data['data'][str(i)])

    res = uploadUserBehavior(json_data_list)
    if res == 0:
        res_code = "200"
    else:
        res_code = "400"
    return res_code



@app.route('/')
def appRoot():
    return 'This is YX-Recommender-System.'



if __name__ == '__main__':
    app.run(debug=True)