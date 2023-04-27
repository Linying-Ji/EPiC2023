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

#############################################################################
# Define a function to get RMSE and save hyperopt results/best_model:
def get_rmse_save_result (trials, best, X_train, X_test):
  # print condition
  if condition == 4:
    cond = "OHID_OHVID"
  print("--------------------------")
  print("Results for "+cond+":")

  # Train the model on the full training set with the best hyperparameters
  best_model = XGBRegressor(random_state=random_state,
                            n_estimators=int(best['n_estimators']), eta=best['eta'],
                            max_depth=int(best['max_depth']), gamma=best['gamma'],
                            subsample=best['subsample'],
                            colsample_bytree=best['colsample_bytree'],
                            eval_metric="rmse")
  best_model.fit(X_train, y_train)

  # save the best model as a file
  joblib.dump(best_model, 'Model_s1_XGB_noDF_'+cond+'.joblib')
  
  # Save Hyperopt Iterations -----------------------------
  # extract
  rmse_df = pd.DataFrame(trials.results)
  rmse_df.columns = ['RMSE', 'status']
  # save
  result = pd.concat([pd.DataFrame(trials.vals), rmse_df], axis=1)
  result.to_csv('Hyperopt_s1_XGB_noDF_'+cond+'.csv', index=False)
  # Get summary statistics for the 'RMSE' column
  print("--- RMSE on TRAIN set ---")
  RMSE = result['RMSE'].describe()
  print(RMSE)
  # Print Mean and SD of RMSE
  print("Train RMSE Best:", RMSE['min'])
  print("Train RMSE Mean:", RMSE['mean'])
  print("Train RMSE SD:", RMSE['std'])
  print("--- RMSE on TEST set ---")

  # Use the best model to make predictions on the test set & get RMSE
  y_pred = best_model.predict(X_test)
  test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  print('Test RMSE:', test_rmse)

#############################################################################

# Read files ------------------------------------------------------------

# Set directory of the processed data: change it to your own directory
path = "XXX/EPiC_2023/train_data/"
os.chdir(path)
os.getcwd()

file_list = ["train_scenario_1_fold_0.csv"]

fold = 0  
filename = file_list[fold]        

total = pd.read_csv(filename)
print("The data shape is:")
print(total.shape)


# Drop features -------------------------------------------------------
total.drop(['SCENARIO'], axis=1, inplace=True)

# Split Train / Valid/ Test set ----------------------------------------------
X_train_list = []
X_valid_list = []
X_test_list = []
y_train_list = []
y_valid_list = []
y_test_list = []

# Loop over unique ID and video combinations:
for (ID, VIDEO_FILENAME), group in total.groupby(['ID', 'VIDEO_FILENAME']):
    # Select target & predictors
    target_col = ['valence', 'arousal']
    y = group[target_col]
    X = group.drop(target_col, axis=1)
    
    # Split group into train and test sets
    X_train1, X_test, y_train1, y_test = train_test_split(
        X,  # Predictors
        y,  # response (arousal and valence)
        test_size = 0.2,  # use 20% as test set
        shuffle = False,  # Important for Scenario 1
        random_state = 1  # Set random seed for reproducibility
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train1,  # Predictors
        y_train1,  # response (arousal and valence)
        test_size = 0.25,  # used another 20% as validation set
        shuffle = False,   
        random_state = 1  
    )
    
    # Append train and test splits to lists
    X_train_list.append(X_train)
    X_valid_list.append(X_valid)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_valid_list.append(y_valid)
    y_test_list.append(y_test)

# Concatenate the train and test sets from the lists to create final splits
X_train1 = pd.concat(X_train_list)
X_valid = pd.concat(X_valid_list)
X_test = pd.concat(X_test_list)
y_train1 = pd.concat(y_train_list)
y_valid = pd.concat(y_valid_list)
y_test = pd.concat(y_test_list)

print(f"X_train1:{X_train1.shape}, X_valid:{X_valid.shape}, X_test:{X_test.shape}")
# First 4/5: TRAIN (in CV)
    # First 3/5: Train1
    # Next  1/5: Valid
# Last  1/5: TEST

###########################################################################
# Hyperparameter Tuning----------------------------------------------------

X_train = pd.concat([X_train1, X_valid], axis=0)
X_train.index = range(0, len(X_train))
y_train = pd.concat([y_train1, y_valid], axis=0)
y_train.index = range(0, len(y_train))

# Make Pre-defined Split: train1 + valid will be used as TRAIN set in cross validations.
cv_fold = np.array([-1] * len(X_train1) + [0] * len(X_valid))
cv_fold
ps = PredefinedSplit(cv_fold)
    

# OH encoding for predictors -------------------------------------------------

# decide columns to apply OH encoding
object_cols = ['ID', 'VIDEO_FILENAME'] 
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_test  = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))
# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_test.index  = X_test.index
# data only with numerical features: Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_test  = X_test.drop(object_cols, axis=1)
# Add one-hot encoded columns to data only with numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test  = pd.concat([num_X_test, OH_cols_test], axis=1)
# Ensure all column names are string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_test.columns  = OH_X_test.columns.astype(str)


# TUNE --------------------------------------------------------------------
# Set directory where the best model will be saved. Change it to your own directory
save_path = "XXX/EPiC_2023/Modeling/XGBoost/s1/" 
os.chdir(save_path)

##### Hyperopt ##### -----------------------------------------------------------
max_eval = 40 # number of iterations in Hyperopt

# Step 1: Initialize space for exploring hyperparameters:
space={
        'max_depth': hp.choice('max_depth', np.arange(3, 12, dtype=int)), #hp.quniform("max_depth", 3, 10, 1),
        'gamma': hp.uniform ('gamma', 0, 9),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'n_estimators': hp.quniform('n_estimators', 50, 400, 1),
        'eta': hp.quniform('eta', 0.05, 0.5, 0.025)
    }

condition = 4 
if condition == 4:
    print("OH ID + OH VID")
# Step 2: Define objective function:
    def hyperparam_tuning4(space):
        model = XGBRegressor(n_estimators=int(space['n_estimators']),
                            eta=space['eta'],
                            max_depth=int(space['max_depth']),
                            gamma=space['gamma'],
                            subsample=space['subsample'],
                            colsample_bytree=space['colsample_bytree'],
                            eval_metric="rmse", seed=12435)
        score = -cross_val_score(estimator=model, 
                                 X = OH_X_train, y = y_train, 
                                 cv=ps, scoring='neg_root_mean_squared_error',
                                 n_jobs = -1).mean()
        return score
    
# Step 3: Run Hyperopt function:
    random_state=42
    start = time.time()
    trials4 = Trials()
    best4 = fmin(fn=hyperparam_tuning4,
                space=space,
                algo=tpe.suggest,
                max_evals=max_eval,
                trials=trials4,
                rstate=np.random.default_rng(random_state))
    print('It takes %s minutes' % ((time.time() - start)/60)) 
    print ("Best params:", best4)
# Step 4: Get the results and save it:
    get_rmse_save_result(trials4, best4, OH_X_train, OH_X_test)

