import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import collections

import matplotlib.pyplot as plt
import operator
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from mlaut.data import Data
from mlaut.experiments import Orchestrator
from mlaut.estimators.default_estimators import instantiate_default_estimators
from mlaut.resampling import Single_Split

data = pd.read_csv('data/all_downloaded_data.csv')

data=data.drop(['Unnamed: 0'], axis=1)
data=data.drop(['CRNCY()'], axis=1)
data=data.drop(['CURRENCY_OF_ISSUE'], axis=1)
data=data.drop(['DURATION()'], axis=1)
data=data.drop(['MULTIPLIER'], axis=1)
data=data.drop(['DATE'], axis=1)

column_names = [
    'AMT_ISSUED',
    'BBG_COMPOSITE_RATING', 
    'COUNTRY_OF_RISK',
    'CPN_TYPE',
    'CURRENCY',
    'ID',
    'INDUSTRY_SECTOR', 
    'ISSUE_DT',
    'IS_COVERED',
    'MARKET_ISSUE',
    'MATURITY',
    'MATURITY_TYPE',
    'PAYMENT_RANK',
    'SPREAD_G',
    'SPREAD_I',
    'SPREAD_OAS',
    'SPREAD_Z',
    'YIELD'    
]
data.columns=column_names

dt_2119 =  datetime.datetime(year=2119, day=24, month=3)# (datetime.datetime.today() + relativedelta(years=100)).strftime('%Y-%m-%d')
dt_days_calculation = datetime.datetime(year=2019, day=24, month=3) #datetime.datetime.today()
data.loc[:,'MATURITY'].fillna(dt_2119, inplace=True) #this needs to be changed to be filled only if it is a perp
data['MATURITY']=pd.to_datetime(data['MATURITY'])
data['DAYS_TO_MATURITY']=data.apply(lambda x: (x['MATURITY']- dt_days_calculation).days, axis=1)


filtered_data = data[['AMT_ISSUED',
                      'BBG_COMPOSITE_RATING',
                      'COUNTRY_OF_RISK',
                      'CPN_TYPE',
                      'CURRENCY',
                      'INDUSTRY_SECTOR',
                      'IS_COVERED',
                      'MARKET_ISSUE',
                      'MATURITY',
                      'MATURITY_TYPE',
                      'PAYMENT_RANK',
                      'SPREAD_Z']]
filtered_data = data[np.isfinite(filtered_data['SPREAD_Z'])]

#drop some columns
filtered_data=filtered_data.drop(['SPREAD_OAS'], axis=1)
filtered_data=filtered_data.drop(['SPREAD_I'], axis=1)
filtered_data=filtered_data.drop(['YIELD'], axis=1)
filtered_data=filtered_data.drop(['SPREAD_G'], axis=1)
filtered_data=filtered_data.drop(['ID'], axis=1)
filtered_data=filtered_data.drop(['ISSUE_DT'], axis=1)

#Where NAN maturity is perpetual. Set it to 1000 years
dt_2119=(datetime.datetime.today() + relativedelta(years=100)).strftime('%Y-%m-%d')
filtered_data.loc[:,'MATURITY'].fillna(dt_2119, inplace=True)
filtered_data['MATURITY']=pd.to_datetime(filtered_data['MATURITY'])
filtered_data['DAYS_TO_MATURITY']=filtered_data['MATURITY'] - datetime.datetime.today()
filtered_data['DAYS_TO_MATURITY']=filtered_data['DAYS_TO_MATURITY'].apply(lambda dt: dt.days)
filtered_data=filtered_data.drop(['MATURITY'], axis=1)

#set amount issued to 0 if nan
filtered_data[filtered_data['AMT_ISSUED'].isna()]=0


filtered_data['BBG_COMPOSITE_RATING']=filtered_data['BBG_COMPOSITE_RATING'].astype('category')
filtered_data['COUNTRY_OF_RISK']=filtered_data['COUNTRY_OF_RISK'].astype('category')
filtered_data['CPN_TYPE']=filtered_data['CPN_TYPE'].astype('category')
filtered_data['CURRENCY']=filtered_data['CURRENCY'].astype('category')
filtered_data['INDUSTRY_SECTOR']=filtered_data['INDUSTRY_SECTOR'].astype('category')
filtered_data['IS_COVERED']=filtered_data['IS_COVERED'].astype('category')
filtered_data['MARKET_ISSUE']=filtered_data['MARKET_ISSUE'].astype('category')
filtered_data['MATURITY_TYPE']=filtered_data['MATURITY_TYPE'].astype('category')
filtered_data['PAYMENT_RANK']=filtered_data['PAYMENT_RANK'].astype('category')

filtered_data_unchanged = filtered_data.copy() #save in case categories are needed later
filtered_data['BBG_COMPOSITE_RATING']=filtered_data['BBG_COMPOSITE_RATING'].cat.codes
filtered_data['COUNTRY_OF_RISK']=filtered_data['COUNTRY_OF_RISK'].cat.codes
filtered_data['CPN_TYPE']=filtered_data['CPN_TYPE'].cat.codes
filtered_data['CURRENCY']=filtered_data['CURRENCY'].cat.codes
filtered_data['INDUSTRY_SECTOR']=filtered_data['INDUSTRY_SECTOR'].cat.codes
filtered_data['IS_COVERED']=filtered_data['IS_COVERED'].cat.codes
filtered_data['MARKET_ISSUE']=filtered_data['MARKET_ISSUE'].cat.codes
filtered_data['MATURITY_TYPE']=filtered_data['MATURITY_TYPE'].cat.codes
filtered_data['PAYMENT_RANK']=filtered_data['PAYMENT_RANK'].cat.codes

filtered_data['AMT_ISSUED'] = (filtered_data['AMT_ISSUED']-filtered_data['AMT_ISSUED'].mean(axis=0))/filtered_data['AMT_ISSUED'].std(axis=0)
filtered_data['DAYS_TO_MATURITY']= (filtered_data['DAYS_TO_MATURITY'] - filtered_data['DAYS_TO_MATURITY'].mean(axis=0))/filtered_data['DAYS_TO_MATURITY'].std(axis=0)


# y = filtered_data['SPREAD_Z']
# X = filtered_data.drop('SPREAD_Z', axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

# idx = np.arange(len(X_train))
# predictions_lm = []

# lm =LinearRegression().fit(X_train, y_train)
# lm_pred = lm.predict(X_test)
# predictions_lm.append(lm_pred)

meta = {
    'class_name': 'SPREAD_Z',
    'source':'bloomberg',
    'dataset_name':'bonds'
}
data = Data(hdf5_datasets_group='fin_study')
data.set_io(input_data='data/fin_study_input.h5', output_data='data/fin_study_output.h5')
data.pandas_to_db(datasets=[filtered_data], dts_metadata=[meta])
data.split_datasets(random_state=1)


orcheststrator = Orchestrator()
orcheststrator.set_data(data)
est = instantiate_default_estimators(['Regression'])

orcheststrator.set_strategies(est)
orcheststrator.set_resampling(Single_Split())
orcheststrator.run()