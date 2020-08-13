import numpy as np
import pandas as pd 
import pickle

from flask import Flask, jsonify, request, abort

my_dt_model = pickle.load(open('DT_model_widebot.sav','rb'))

app = Flask(__name__)

@app.route('/api',methods=['POST'])
def make_predict():
    data = request.get_json(force=True)
    variables = [f"variable{i}"for i in range(1,20)]
    del variables[-3]
    predict_request = [data[variable] for variable in variables]
    predict_request = np.array(predict_request)
    predicted_value = my_dt_model.predict(predict_request.reshape(1,-1))
    output = 'yes.' if predicted_value else 'no.'
    print(predicted_value)
    return jsonify(result=output)
    


if __name__ == '__main__':
    app.run(debug=True)