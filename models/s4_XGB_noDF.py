print("START")

import os
import time
import pandas as pd
import numpy as np
import re

# Data split & Tuning params
from sklearn.model_selection import PredefinedSplit, GroupKFold
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

# XGBoost
from xgboost import XGBRegressor

# save model
import joblib

import hyperopt
from hyperopt import fmin, tpe, hp, anneal, Trials
from sklearn.model_selection import cross_val_score

############################################################################
# Define a function to get RMSE and save hyperopt results and the model:
def get_rmse_save_result (trials, best, X_train):
  # print condition
  if best is best1:
    condition = "numID"
  print("--------------------------")
  print("Results for "+condition+":")

  # Train the model on the full training set with the best hyperparameters
  best_model = XGBRegressor(random_state=random_state,
                            n_estimators=int(best['n_estimators']), eta=best['eta'],
                            max_depth=int(best['max_depth']), gamma=best['gamma'],
                            subsample=best['subsample'],
                            colsample_bytree=best['colsample_bytree'],
                            eval_metric="rmse")
  best_model.fit(X_train, y_train)

  # save the best model as a file
  joblib.dump(best_model, 'Model_s4_XGB_noDF_allfold_'+condition+'.joblib')
  
  # Save Hyperopt Iterations -----------------------------
  # extract
  rmse_df = pd.DataFrame(trials.results)
  rmse_df.columns = ['RMSE', 'status']
  # save
  result = pd.concat([pd.DataFrame(trials.vals), rmse_df], axis=1)
  result.to_csv('Hyper_s4_XGB_noDF_allfold_'+condition+'.csv', index=False)
  # Get summary statistics for the 'RMSE' column
  print("--- RMSE on TRAIN set ---")
  RMSE = result['RMSE'].describe()
  print(RMSE)
  # Print Mean and SD of RMSE
  print("Train RMSE Best:", RMSE['min'])
  print("Train RMSE Mean:", RMSE['mean'])
  print("Train RMSE SD:", RMSE['std'])

############################################################################

# Read Data ------------------------------------------------------------------

# Set directory of the processed data: change it to your own directory
path = "XXX/EPiC_2023/train_data/" 
os.chdir(path)

file_list = ["train_scenario_4_fold_0.csv", 
             "train_scenario_4_fold_1.csv"]

total = pd.DataFrame()

# concact data from different folds
for file_name in file_list:
    fold_data = re.findall('fold_''\d+',file_name)
    data = pd.read_csv(file_name)
    data['fold'] = fold_data[0]
    # Concatenate the data frames vertically
    total = pd.concat([total, data], axis=0)
    
# check if the counts are correct    
print(total['fold'].value_counts()) 

# drop duplicates
print("Before drop duplicated:", total.shape)
total = total.drop(['fold'], axis=1).drop_duplicates()
total = total.reset_index(drop = True)
print('After drop duplicated',total.shape)

# add a variable that indicates data from same elicitor
def create_elicitor(x):
    if x in [16, 20]:
        return 0;
    elif x in [0, 3]:
        return 1
    elif x in [22, 10]:
        return 2
    elif x in [4, 21]:
        return 3
    
total['elicitor'] = total['VIDEO_FILENAME'].apply(create_elicitor)


# Drop features------------------------------------------
total.drop(['SCENARIO'], axis=1, inplace=True)

# Split Target vs. Predictors -------------------------------------------
train = total

# select target 
target_col = ['valence', 'arousal']
# split train
y_train = train[target_col]
X_train = train.drop(target_col, axis=1)
print("X_train.columns:", X_train.columns)

# Specifiy GroupKfold CV --------------------------------------------------
gkf = GroupKFold(n_splits=4)

# TUNE --------------------------------------------------------------------

# Set directory where the best model will be saved. Change it to your own directory
save_path = "XXX/EPiC_2023/Modeling/XGBoost/s4/" 
os.chdir(save_path)

###### Hyperopt #######----------------------------------------------------
max_eval = 30 # number of iterations in hyperopt.

# Step 1: Initialize space for exploring hyperparameters:
space={'max_depth': hp.choice('max_depth', np.arange(3, 12, dtype=int)), #hp.quniform("max_depth", 3, 10, 1),
       'gamma': hp.uniform ('gamma', 0, 9),
       'subsample': hp.uniform('subsample', 0.5, 1),
       'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
       'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
       'eta': hp.quniform('eta', 0.05, 0.5, 0.025)
      }

# Step 2: Define objective function:
def hyperparam_tuning1(space):
  model = XGBRegressor(n_estimators=int(space['n_estimators']),
                     eta=space['eta'],
                     max_depth=int(space['max_depth']),
                     gamma=space['gamma'],
                     subsample=space['subsample'],
                     colsample_bytree=space['colsample_bytree'],
                     eval_metric="rmse", seed=12435)
  # cross validation
  score = -cross_val_score(estimator=model, 
                                 X = X_train, y = y_train, 
                                 groups = train['ID'],
                                 cv=gkf,
                                 scoring='neg_root_mean_squared_error',
                                 n_jobs = -1).mean()
  return score

# Step 3: Run Hyperopt function:
random_state=42
start = time.time()
trials1 = Trials()
best1 = fmin(fn=hyperparam_tuning1,
            space=space,
            algo=tpe.suggest,
            max_evals=max_eval,
            trials=trials1,
            rstate=np.random.default_rng(random_state))
print('It takes %s minutes' % ((time.time() - start)/60)) 
print ("Best params:", best1)

# Step 4: Get the results and save it:
get_rmse_save_result(trials1, best1, X_train)
