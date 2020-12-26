import os
import joblib as jb
import pandas as pd

from flask import Flask, request, Response

from pipeline.Pipeline import churn_pipeline



model_level_01 = jb.load(open('model/model_lgbm_level_01.pkl.z', 'rb'))
model_level_02 = jb.load(open('model/model_lgbm_level_02.pkl.z', 'rb'))


# Initialize API
app = Flask(__name__)

@app.route('/churn/predict', methods=['POST'])
def churn_predict():
	test_JSON = request.get_json()

	if test_JSON: #there is data
		if isinstance(test_JSON, dict):
			df = pd.DataFrame(test_JSON, index=[0]) #unique example
		else:
			df = pd.DataFrame(test_JSON, columns=test_JSON[0].keys()) #multiple examples

		# Instantiate
		pipeline = churn_pipeline()

		# Data Cleaning
		df1 = pipeline.data_cleaning(df)
		#  Feature Engineering
		df2 = pipeline.feature_engineering(df1)
		# Data Preparation
		df3 = pipeline.data_preparation(df2)
		# Prediction
		df_response = pipeline.get_prediction(model_level_01, model_level_02, df, df3)

		return df_response

	else:
		return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
	port = os.environ.get( 'PORT', 5000)
	app.run(host='localhost', port=port)
	#app.run(host='0.0.0.0', port=port)