import bisect
import inflection
import joblib as jb
import numpy as np
import pandas as pd




class churn_pipeline(object):
	def __init__(self):
		self.home_path                            = 'D:/01-DataScience/04-Projetos/00-Git/Churn-Prediction-TopBank/02-Notebooks/02-Scond_Cycle_CRISP/'
		self.rs_balance                           = jb.load(open(self.home_path + 'parameter/balance_scaler.pkl.z', 'rb'))
		self.rs_balance_current_account_balance   = jb.load(open(self.home_path + 'parameter/balance_current_account_balance.pkl.z', 'rb'))
		self.rs_amount_credit                     = jb.load(open(self.home_path + 'parameter/amount_credit.pkl.z', 'rb'))
		self.mms_cred_age                         = jb.load(open(self.home_path + 'parameter/cred_age.pkl.z', 'rb'))
		self.mms_age                              = jb.load(open(self.home_path + 'parameter/age.pkl.z', 'rb'))
		self.mms_amount_num_of_products           = jb.load(open(self.home_path + 'parameter/amount_num_of_products.pkl.z', 'rb'))
		self.mms_num_of_products_has_cr_card_mul  = jb.load(open(self.home_path + 'parameter/num_of_products_has_cr_card_mul.pkl.z', 'rb'))
		self.mms_cred_num_of_products             = jb.load(open(self.home_path + 'parameter/cred_num_of_products.pkl.z', 'rb'))

	
	def group_age(self, num, breakpoints=[10, 20, 30, 45, 60, 70, 80, 120], result='01234567'):
		i = bisect.bisect(breakpoints, num-1)
		age_mapping = {
			0: 'Child',
			1: 'Teenager',
			2: 'Young',
			3: 'Adult',
			4: 'Midlife',
			5: 'Senior',
			6: 'Mature Adulthood',
			7: 'Late Adulthood'
		}
		return age_mapping[i]

	
	def data_cleaning(self, df01):
		# Merge data with quandl data
		df_country_data =  pd.read_feather(self.home_path + '00-Data/quandl_data.feather')
		df01 = pd.merge(df01, df_country_data, how='left', on='Geography')
		
		# Rename Columns (snakecase)
		snakecase = lambda col: inflection.underscore(col)
		new_columns = list(map(snakecase, df01.columns))

		# rename
		df01.columns = new_columns
		
		return df01
		
		
	def feature_engineering(self, df02):
		# Tenure Vigency
		df02['tenure_vigency'] = df02['tenure'] + 1

		# credit_score / age
		df02['cred_age'] = df02['credit_score'] / df02['age']

		# amount
		df02['amount'] = df02['estimated_salary'] + df02['balance']

		# amount / credit_score
		df02['amount_credit'] = df02['amount'] / df02['credit_score']

		# amount / num_of_products
		df02['amount_num_of_products'] = df02['amount'] / df02['num_of_products']

		# credit score / num_of_products
		df02['cred_num_of_products'] = df02['credit_score'] / df02['num_of_products']

		# num_of_products * credit card
		df02['num_of_products_has_cr_card_mul'] = df02['num_of_products'] * df02['has_cr_card']

		# balance / current_account_balance
		df02['balance_current_account_balance'] = df02['balance'] / df02['current_account_balance']

		# num_of_products / inflation_index_average_consumer_prices
		df02['num_of_products_inflation_index'] = df02['num_of_products'] / df02['inflation_index_average_consumer_prices']

		# Group Age
		df02['age_group'] = df02['age'].apply(lambda row: churn_pipeline.group_age(self, row))
		
		return df02

	
	
	def data_preparation(self, df03):
		# balance
		df03['balance'] = self.rs_balance.transform(df03[['balance']].values)

		# balance_current_account_balance
		df03['balance_current_account_balance'] = self.rs_balance_current_account_balance.transform(df03[['balance_current_account_balance']].values)

		# amount_credit
		df03['amount_credit'] = self.rs_amount_credit.transform(df03[['amount_credit']].values)

		# cred_age
		df03['cred_age'] = self.mms_cred_age.transform(df03[['cred_age']].values)

		# age
		df03['age'] = self.mms_age.transform(df03[['age']].values)

		# amount_num_of_products
		df03['amount_num_of_products'] = self.mms_amount_num_of_products.transform(df03[['amount_num_of_products']].values)

		# num_of_products_has_cr_card_mul
		df03['num_of_products_has_cr_card_mul'] = self.mms_num_of_products_has_cr_card_mul.transform(df03[['num_of_products_has_cr_card_mul']].values)

		# cred_num_of_products
		df03['cred_num_of_products'] = self.mms_cred_num_of_products.transform(df03[['cred_num_of_products']].values)

		# age_group_Midlife
		df03['age_group_Midlife'] = np.where(df03['age_group'] == 'Midlife', 1, 0)
		
		to_keep = ['age',
			'balance',
			'num_of_products',
			'cred_age',
			'amount_credit',
			'amount_num_of_products',
			'cred_num_of_products',
			'num_of_products_has_cr_card_mul',
			'balance_current_account_balance',
			'num_of_products_inflation_index',
			'age_group_Midlife']

		df03 = df03[to_keep]
		
		return df03
	
	
	
	def get_prediction(self, model_level_01, model_level_02, orinal_dataset, dataset):
		model_level_01 = model_level_01
		model_level_02 = model_level_02
		# Prediction Proba
		yhat_proba_level_01 = model_level_01.predict_proba(dataset)[:,1]
		df_result_level_01 = pd.DataFrame({'yhat_proba_level_01': yhat_proba_level_01})
		dataset = pd.concat([dataset, df_result_level_01['yhat_proba_level_01']], axis=1)

		# Prediction Proba
		yhat_proba_level_02 = model_level_02.predict_proba(dataset)[:,1]
		df_result_level_02 = pd.DataFrame({'yhat_proba_level_02': yhat_proba_level_02})

		df_result_level_02['final_result'] = df_result_level_02['yhat_proba_level_02'].apply(lambda row: 1 if row >= 0.52 else 0)
		df_result = pd.concat([orinal_dataset, df_result_level_02['final_result']], axis=1)
		df_result = df_result[['CustomerId', 'Surname', 'CreditScore', 'Geography',
							   'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
							   'IsActiveMember', 'EstimatedSalary', 'final_result']]

		return df_result.to_json(orient='records', date_format='iso')