# Databricks notebook source
# MAGIC %md ## GR5069 - Applied Data Science, Assignment #4
# MAGIC ### Cindy Chen, cjc2279

# COMMAND ----------

# install mlflow
%pip install mlflow

# COMMAND ----------

#packages to load
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
import os

from numpy import savetxt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from sklearn.metrics import *

import seaborn as sns
import tempfile

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q1: Build any model of your choice with tunable hyperparameters

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The model I decided to build is to use qualifying times to create a model to predict final race outcomes.

# COMMAND ----------

# Step 1. Load Data

# load qualifying race data
s3 = boto3.client('s3')
bucket = 'columbia-gr5069-main'
constructor_data = 'raw/qualifying.csv'
obj = s3.get_object(Bucket= bucket, Key= constructor_data) 
qualifying = pd.read_csv(obj['Body'])

#preview the table
display(qualifying)

# COMMAND ----------

# load final results
results_data = 'raw/results.csv'
obj = s3.get_object(Bucket= bucket, Key= results_data) 
results = pd.read_csv(obj['Body'])

display(results)

# COMMAND ----------

# Step 2. merge data 
df_final = qualifying.merge(results, how = 'left', left_on = ['raceId', 'driverId', 'constructorId'], right_on = ['raceId', 'driverId', 'constructorId']) # remove qualifyId since this is just an index
df_final = df_final.drop(['number_y', 'position_x', 'number_x', 'resultId', 'qualifyId', 'time', 'milliseconds'], axis = 1)       # drop certain columns that are duplicates or are equivalent to result variable

# remove rows with no position
# confirmed that removed rows are random
df_final = df_final[(df_final['positionText'] != 'R') &
                    (df_final['positionText'] != 'D') &
                    (df_final['positionText'] != 'N') &
                    (df_final['positionText'] != 'W') &
                   (df_final['positionText'] != 'null')]

# remove missing rank, which is my dependent variable
df_final = df_final[df_final['rank'].isnull() == False]

# COMMAND ----------

# MAGIC %md **Missing value imputation note:** To impute missing values (denoted as "\N" in the data set), I have set missing values to zero, rather than remove them from my dataset. I am comfortable setting values to zero, because it is not a value that would overlap with any legitimate values (for instance, no one has a position of zero, a lap time of 0, or an average lap speed of 0).  Since I am not running any type of model for inference, setting missing values to 0 will not compromise the quality or performance of my model.

# COMMAND ----------

# convert certain columns to strings
df_final['raceId'] = df_final['raceId'].astype(str)
df_final['driverId'] = df_final['driverId'].astype(str)
df_final['constructorId'] = df_final['constructorId'].astype(str)
df_final['statusId'] = df_final['statusId']
df_final['grid'] = df_final['grid']

# move everything to lowercase so we can impute 0
df_final['fastestLapSpeed'] = df_final['fastestLapSpeed'].str.lower()
df_final['rank'] = df_final['rank'].str.lower()
df_final['fastestLap'] = df_final['fastestLap'].str.lower()
df_final['position_y'] = df_final['position_y'].str.lower()

# assign these variables' missing values to 0, which is not a real value
# doing so allows us to convert the column to the desired type (numeric) and we can impute in a later step
df_final.loc[df_final['fastestLapSpeed'].isnull() == True, ['fastestLapSpeed']] = 0
df_final.loc[df_final['rank'].isnull() == True, ['rank']] = 0
df_final.loc[df_final['fastestLap'].isnull() == True, ['fastestLap']] = 0
df_final.loc[df_final['position_y'].isnull() == True, ['position_y']] = 0
df_final.loc[df_final['points'].isnull() == True, ['points']] = 0

# replace entries with "\N" to 0
df_final.replace(to_replace=[r'\\t|\\n|\\r', ''\t|\n|\r'], value=[0,0], regex=True, inplace=True)

# convert certain numeric values
df_final['fastestLapSpeed'] = df_final['fastestLapSpeed'].astype(float)
df_final['rank'] = df_final['rank'].astype(int)
df_final['fastestLap'] = df_final['fastestLap'].astype(int)
df_final['position_y'] = df_final['position_y'].astype(int)

# COMMAND ----------

# replace NaN
df_final.q1.replace(np.NaN, '0:00.000', inplace=True)
df_final.q2.replace(np.NaN, '0:00.000', inplace=True)
df_final.q3.replace(np.NaN, '0:00.000', inplace=True)
df_final.fastestLapTime.replace(np.NaN, '0:00.000', inplace=True)

# for times, we will repeat the process but alter the imputed value
df_final['q1'] = df_final['q1'].str.lower()
df_final['q2'] = df_final['q2'].str.lower()
df_final['q3'] = df_final['q3'].str.lower()
df_final['fastestLapTime'] = df_final['fastestLapTime'].str.lower()

# replace Nulls again; this section is required for the convert_times function to work
# we will not reimpute these with the mode
df_final.loc[df_final['q1'].isnull() == True, ['q1']] = '0:00.000'
df_final.loc[df_final['q2'].isnull() == True, ['q2']] = '0:00.000'
df_final.loc[df_final['q3'].isnull() == True, ['q3']] = '0:00.000'
df_final.loc[df_final['fastestLapTime'].isnull() == True, ['fastestLapTime']] = '0:00.000'

df_final.replace(to_replace=[r'\\t|\\n|\\r', ''\t|\n|\r'], value=['0:00.000','0:00.000'], regex=True, inplace=True)

# filter out a fastestLapTime that doesn't make sense
df_final = df_final[df_final['fastestLapTime'] != '192.074']

# COMMAND ----------

# define a function for converting lap times to milliseconds
def convert_times(column_name):
    
    new_col = []
    
    for i in  df_final[column_name]:
        hours, minutes, seconds = (['0', '0'] + i.split(':'))[-3:]
        minutes = int(minutes)
        seconds = float(seconds)
        milliseconds = int(60000 * minutes + 1000 * seconds)
        new_col.append(milliseconds)
        
    df_final[column_name] = new_col

# COMMAND ----------

# since we need times as numeric for our model, we need to convert these to milliseconds
convert_times('q1')
convert_times('q2')
convert_times('q3')
convert_times('fastestLapTime')

# COMMAND ----------

# remove these two position columns which are redundant since we have position_y
df_final = df_final.drop(['positionText', 'positionOrder'], axis = 1)

# convert certain values back to NA
df_final.loc[df_final['fastestLapSpeed'] == 0, ['fastestLapSpeed']] = None
df_final.loc[df_final['fastestLap'] == 0, ['fastestLap']] = None
df_final.loc[df_final['position_y'] == 0, ['position_y']] = None

#remove entries where 'rank' is 0 (because the rank was technically Null)
df_final = df_final.loc[df_final['rank'] != 0]

# print data types of each column
df_final.dtypes

# COMMAND ----------

df_final.head(10)

# COMMAND ----------

# Step 3. isolate dependent variable and train/test/split data
X_train, X_test, y_train, y_test = train_test_split(df_final.drop(['rank'], axis=1), df_final[['rank']].values.ravel(), random_state=42)

# COMMAND ----------

#identify each type of variable in our data set
categorical_features = ['raceId', 'driverId', 'constructorId', 'grid', 'statusId']
numeric_features = ['q1', 'q2', 'q3', 'fastestLapTime', 'fastestLapSpeed', 'points', 'laps', 'fastestLap', 'position_y']

#create transformers to scale data
numeric_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
preprocess = ColumnTransformer(
    transformers = [
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)])

# fit preprocessor to data
preprocess = preprocess.fit(X_train)

# COMMAND ----------

def preprocessor(data):
    preprocessed_data = preprocess.transform(data)
    return preprocessed_data
  
# preprocess data for both sets of X data (these are the scaled sets)
X_train_new = preprocessor(X_train)
X_test_new = preprocessor(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Q2: Create an experiment setup where - for each run - you log:
# MAGIC 
# MAGIC * the hyperparameters used in the model
# MAGIC * the model itself
# MAGIC * every possible metric from the model you chose
# MAGIC * at least two artifacts (plots, or csv files)

# COMMAND ----------

# Enable autolog()
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
mlflow.sklearn.autolog()

# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  


#### LOG MODEL FUNCTION 1: SKLEARN ####

# in this function: takes additional inputs for use of GridSearch and the model type

def log_my_model(run_name, model_type, params, use_GridSearch, X_train, X_test, y_train, y_test):
    
    with mlflow.start_run(run_name=run_name) as run:
        
        # run GridSearchCV if applicable
        if (use_GridSearch != 'GridSearch'):
            # Create and train mode
            if (model_type == 'RandomForest'):
                setup_model = RandomForestRegressor(**params)
                # Log model
                mlflow.sklearn.log_model(setup_model, 'random-forest-model')
            elif (model_type == 'GradientBoosting'):
                setup_model = GradientBoostingRegressor(**params)
                # Log model
                mlflow.sklearn.log_model(setup_model, 'gradient-boosting-model')
            elif (model_type == 'SVC'):
                setup_model = SVR(**params)
                # Log model
                mlflow.sklearn.log_model(setup_model, 'support-vector-machine-model')
            else:
                return('Model type not supported by this function')
            
            # if not using GridSearch, fit model directly
            setup_model.fit(X_train, y_train)
            # Use the model to make predictions on the test dataset.
            predictions = setup_model.predict(X_test)
        
        # if GridSearchCV is application
        else:
            # define model type being used for GridSearchCV
            if (model_type == 'RandomForest'):
                my_model_type = RandomForestRegressor()
                gc_regressor_name = 'random-forest-model'
            elif (model_type == 'GradientBoosting'):
                my_model_type = GradientBoostingRegressor()
                gc_regressor_name = 'gradient-boosting-model'
            elif (model_type == 'SVC'):
                my_model_type = SVR()
                gc_regressor_name = 'svr'

            # run GridSearch and then fit model (cross validation of 3 folds)
            setup_model = GridSearchCV(my_model_type, params, cv = 3)
            mlflow.sklearn.log_model(setup_model, gc_regressor_name)
            
            # translate the best params from gridsearchcv to a new model (required step for feature_importance to run)
            setup_model.fit(X_train, y_train)
            # Use the model to make predictions on the test dataset.
            predictions = setup_model.predict(X_test)
            
        # Log params
        [mlflow.log_param(param, value) for param, value in params.items()]
    
        # Model performance
        # Create metrics
        evs = explained_variance_score(y_test, predictions)
        max_err = max_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        med_ab_err = median_absolute_error(y_test, predictions)
        r2_score_ = r2_score(y_test, predictions)
        mtd = mean_tweedie_deviance(y_test, predictions)
        
        try:
            mpd = mean_poisson_deviance(y_test, predictions)
            mgd = mean_gamma_deviance(y_test, predictions)
            d2tws = d2_tweedie_score(y_test, predictions)
            mpl = mean_pinball_loss(y_test, predictions)
            msle = mean_squared_log_error(y_test, predictions)
        except:
            'do nothing'
        else:
            mpd = mean_poisson_deviance(y_test, predictions)
            mgd = mean_gamma_deviance(y_test, predictions)
            d2tws = d2_tweedie_score(y_test, predictions)
            mpl = mean_pinball_loss(y_test, predictions)
            msle = mean_squared_log_error(y_test, predictions)
            print('  Mean Poisson Deviance: {}'.format(mpd))
            print('  Mean Gamma Deviance: {}'.format(mgd))
            print('  D2 Tweedie Score: {}'.format(d2tws))
            print('  Mean Pinball Loss: {}'.format(mpl))
            print('  MSLE: {}'.format(msle))
            mlflow.log_metric('Mean Poisson Deviance', mpd)
            mlflow.log_metric('Mean Gamma Deviance', mgd)
            mlflow.log_metric('D2 Tweedie Score', d2tws)
            mlflow.log_metric('Mean Pinball Loss', mpl)
            mlflow.log_metric('msle', msle)
    
        print('Explained Variance: {}'.format(evs))
        print('Max Error: {}'.format(max_err))
        print('  MAE: {}'.format(mae))
        print('  MSE: {}'.format(mse))
        print('  Median Abs Error: {}'.format(med_ab_err))
        print('  R2 Score: {}'.format(r2_score_))
        print('  Mean Tweedie Deviance: {}'.format(mtd))
        
        # Log metrics
        mlflow.log_metric('evs', evs)
        mlflow.log_metric('max_err', max_err)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('Median_Abs_Error', med_ab_err)
        mlflow.log_metric('Mean_Tweedie_Deviance', mtd)
            
        # Create feature importance
        
        if (use_GridSearch != 'GridSearch'):
            importance = pd.DataFrame(list(zip(df_final.columns, setup_model.feature_importances_)), 
                                  columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
    
            # Log importances using a temporary file
            temp = tempfile.NamedTemporaryFile(prefix='feature-importance-', suffix='.csv')
            temp_name = temp.name
        
            try:
                importance.to_csv(temp_name, index=False)
                mlflow.log_artifact(temp_name, 'feature-importance.csv')
            finally:
                temp.close() # Delete the temp file
    
        # Create plot
        fig, ax = plt.subplots()

        sns.residplot(predictions, y_test, lowess=True)
        plt.xlabel('Predicted values ')
        plt.ylabel('Residual')
        plt.title('Residual Plot')

        # Log residuals using a temporary file
        temp = tempfile.NamedTemporaryFile(prefix='residuals-'', suffix='.png')
        temp_name = temp.name
        
        try:
            fig.savefig(temp_name)
            mlflow.log_artifact(temp_name, 'residuals.png')
        finally:
            temp.close() # Delete the temp file
      
        display(fig)
    
        #this gives us a unique ID
        runID = run.info.run_uuid
        experimentID = run.info.experiment_id
        
        print('Inside MLflow Run with run_id {} and experiment_id {}'.format(runID, experimentID))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q3: Track your MLFlow experiment and run at least 10 experiments with different parameters each
# MAGIC 
# MAGIC ##### _I ran 10 runs with Random Forest along with other types of runs in the same function to experiment with MLFlow_

# COMMAND ----------

# EXPERIMENT 1 - Random Forest
params = {
  'n_estimators': 100,
  'max_depth': 5,
  'random_state': 42
}

log_my_model('First Run - RF', 'RandomForest', params, 'no_GC', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 2 - Random Forest
params = {
  'n_estimators': 250,
  'max_depth': 4,
  'max_samples': 500,
  'random_state': 42
}

log_my_model('Second Run - RF', 'RandomForest', params, 'no_GC', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 3 - Gradient Boosting
params = {
  'n_estimators': 200,
  'max_depth': 3,
  'subsample': 0.9,
  'random_state': 42
}

log_my_model('Third Run - XGB', 'GradientBoosting', params, 'no_GC', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 4 - Gradient Boosting with GridSearchCV
params = {
  'n_estimators': np.arange(100, 300, 50),
  'max_depth': np.arange(1, 4, 1),
  'warm_start': [True, False]
}

log_my_model('4th Run - XGB with GridSearch', 'GradientBoosting', params, 'GridSearch', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 5 - Random Forest with GridSearchCV
params = {
  'n_estimators': np.arange(100, 300, 50),
  'max_depth': np.arange(1,3,1),
  'max_features': ['auto', 'sqrt', 'log2']
}

log_my_model('5th Run - Random Forest with GridSearch', 'RandomForest', params, 'GridSearch', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 6 - SVC with GridSearchCV
params = {
    'C': np.arange(0.1, 2, 0.2),
    'degree': np.arange(1, 4, 1),
    'kernel': ['linear', 'poly']
}

log_my_model('6th Run - SVC with GridSearch', 'SVC', params, 'GridSearch', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 7 - Random Forest
params = {
    'n_estimators': 100,
    'min_weight_fraction_leaf': 0.1
}

log_my_model('7th Run - Random Forest', 'RandomForest', params, 'noGC', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 8 - SVC
params = {
    'C': [0.4, 1, 2.5],
    'gamma': ['auto', 'scale']
}

log_my_model('8th Run - SVC with GridSearch', 'SVC', params, 'GridSearch', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 9 - GradientBoosting
params = {
  'n_estimators': 300,
  'max_depth': 4,
  'learning_rate': 0.2
}

log_my_model('9th Run - XGB', 'GradientBoosting', params, 'no', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 10 - Gradient Boosting
params = {
  'n_estimators': np.arange(100, 500, 100),
  'loss': ['huber', 'quantile']
}

log_my_model('10th Run - GradientBoosting with GridSearch', 'GradientBoosting', params, 'GridSearch', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 11 - Random Forest
params = {
    'n_estimators': 500,
    'criterion': 'poisson'
}

log_my_model('11th Run - Random Forest', 'RandomForest', params, 'noGC', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 12 - Random Forest
params = {
    'n_estimators': 350,
    'criterion': 'poisson',
    'max_depth': 4
}

log_my_model('12th Run - Random Forest', 'RandomForest', params, 'noGC', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 13 - Random Forest
params = {
    'n_estimators': 100,
    'max_depth': 4,
    'max_features': 'log2'
}

log_my_model('13th Run - Random Forest', 'RandomForest', params, 'noGC', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 14 - Random Forest
params = {
    'n_estimators': 130,
    'max_depth': 2,
    'max_features': 'log2'
}

log_my_model('14th Run - Random Forest', 'RandomForest', params, 'noGC', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# EXPERIMENT 15 - Random Forest
params = {
    'n_estimators': 280,
    'max_features': 'sqrt'
}

log_my_model('15th Run - Random Forest', 'RandomForest', params, 'noGC', X_train_new, X_test_new, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q4: Select your best model run and explain why

# COMMAND ----------

# MAGIC %md My best model was the "9th Run - Gradient Boosting Regressor model (with no GridSearchCV).  The parameters for this model were:
# MAGIC * 'n_estimators': 300,
# MAGIC * 'max_depth': 4,
# MAGIC * 'learning_rate': 0.2
# MAGIC 
# MAGIC I consider this my best run because when I went on the Experiment page and reviewed the metrics of these 10 runs, this run yielded the lowest error among several metrics, including MAE (mean absolute error), median absolute error, Mean Tweedie Deviance, mean squared error among the test set.  I ignored the training performance to compare the runs since it's more important to consider the test set performance, as this ensures that our model generalizes well to unseen data (and ideally balances the bias-variance tradeoff).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q5: As part of your GitHub classroom submission include screenshots of
# MAGIC * your MLFlow Homepage
# MAGIC * your detailed run page
