###########################################################################
# Load libraries                                                          #
###########################################################################
import pathlib as pathlib
import os as os
import pandas as pd  # data wrangling
import numpy as np
from category_encoders import OneHotEncoder
import h2o # currently supported in Python 3.6
###########################################################################
# Get data                                                                #
###########################################################################
# training data
trainData = pd.read_csv(pathlib.Path(os.getcwd(), 'input','train.csv'))
# test data
testData = pd.read_csv(pathlib.Path(os.getcwd(), 'input','test.csv'))
###########################################################################
# Some basic information about the training data                          #
###########################################################################
# shape
print(trainData.shape)
# column list
print(trainData.columns)
# a sample of the rows (or use head for the first rows)
print(trainData.sample(10))
# basic information
print(trainData.info())
# Statistical Summary
print(trainData.describe().transpose())
# Target class distribution
print(trainData.groupby('Survived').size())
# check for missing values
print(trainData.isnull().any())
# count the number of NaN values in each column
print(trainData.isnull().sum())
###########################################################################
# Create one dataset for feature engineering                              #
###########################################################################
trainData['DataPartition'] = 'train'
testData['DataPartition'] = 'test'
## create one data set
fullData = pd.concat([trainData, testData], sort = False)
## check the result
print(fullData.shape)
print(fullData.sample(10))
print(fullData.dtypes)
###########################################################################
# Missing values                                                          #
###########################################################################
# add indicators for columns which will not be imputed
fullData['hasAge'] = np.where(fullData.Age.isnull(), 0, 1)
fullData['hasCabin'] = np.where(fullData.Cabin.isnull(), 0, 1)
# impute with mode
fullData.Embarked = np.where(fullData.Embarked.isnull(), fullData.Embarked.value_counts().index[0], fullData.Embarked)
# impute with median
# fare is missing for one third class passenger
fullData.Fare = np.where(fullData.Fare.isnull(), fullData.groupby('Pclass')['Fare'].median()[3], fullData.Fare)
###########################################################################
# Feature engineering                                                     #
###########################################################################
# Extract title from name
fullData['Title'] = fullData.Name.str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
# Clean up title
cleanUpTitle = {'Title': {'Mlle': 'Miss', "Mme": 'Mrs', 'Ms': 'Miss', 'Col': 'Sir', 'Major': 'Sir', 'Don': 'Sir', 'Dona': 'Lady', 'Jonkheer': 'Sir', 'the Countess': 'Lady'}}
fullData.replace(cleanUpTitle, inplace=True)
fullData.groupby('Title').size()
# Pax per ticket
ticketPax = pd.DataFrame(fullData.groupby('Ticket')[['PassengerId', 'Fare']].nunique())
ticketPax = ticketPax.rename(columns={'PassengerId': 'TicketPax', 'Fare': 'TicketDiffFares'})
fullData = fullData.set_index('Ticket')
fullData = pd.merge(fullData, ticketPax, left_index=True, right_index=True, how='left').reset_index()
# Fare paid per person
fullData.loc[fullData.TicketDiffFares == 1, 'TicketFare'] = (fullData.Fare / fullData.TicketPax)
fullData.loc[fullData.TicketDiffFares > 1, 'TicketFare'] = fullData.Fare
# check results
fullData[fullData.TicketDiffFares == 1].sample(4)
# Family size
fullData['FamilySize'] = fullData.SibSp + fullData.Parch + 1
# remarkable survival of third class passengers
fullData['ChineseSailers'] = np.where(fullData.Ticket != '1601', 0, 1)
# all in the guarantee group died
fullData['GuaranteeGroup'] = np.where(((fullData.Pclass != 3) & (fullData.hasCabin == 1) & (fullData.Fare == 0)) | (fullData.Title == 'Capt'), 1,    0)
###########################################################################
# Binning                                                                 #
###########################################################################
# Bin family size with cut
fullData['FamilySizeBin'] = pd.cut(fullData.FamilySize, [0, 1, 3, 4, 5, 11], labels=['Single', 'Small', 'Medium', 'Large', 'Xl'], include_lowest=True)
# Age and fare
# The weight of evidence tells the predictive power of an independent variable in relation to the dependent variable.
# Information value is one of the most useful technique to select important variables in a predictive model.
# It helps to rank variables on the basis of their importance.
# The age and fare cuts bins have been determined by the smbinning package in R in relation to survival.
fullData['IsBoy'] = np.where((fullData.Age <= 12) & (fullData.Sex == "male"), 1, 0)
fullData['IsFemale'] = np.where((fullData.Sex != "male"), 1, 0)
fullData['FareLow'] = np.where(((fullData.Fare <= 9.8375) & (fullData.Sex == "female")) | ((fullData.Fare <= 23.55) & (fullData.Sex == "male")), 1,    0)
###########################################################################
# Ordinal features                                                        #
###########################################################################
from pandas.api.types import CategoricalDtype
# passenger class
cat_type = CategoricalDtype(categories=[1, 2, 3], ordered=True)
fullData.Pclass = fullData.Pclass.astype(cat_type)
# family size bin
cat_type = CategoricalDtype(categories=['Single', 'Small', 'Medium', 'Large', 'Xl'], ordered=True)
fullData.FamilySizeBin = fullData.FamilySizeBin.astype(cat_type)
# title
cat_type = CategoricalDtype(categories=fullData.Title.unique(), ordered=False)
fullData.Title = fullData.Title.astype(cat_type)
###########################################################################
# Re-engineer                                                             #
###########################################################################
# drop features that won't be used
# re-order so that DataPartition, PassengerId and Survived are the last three columns
# columns = list(fullData.columns.values)
fullData = fullData[['Pclass', 'SibSp', 'Parch', 'hasAge', 'hasCabin', 'Title', 'TicketPax', 'ChineseSailers', 'GuaranteeGroup', 'FamilySizeBin', 'IsBoy', 'IsFemale', 'FareLow', 'DataPartition', 'PassengerId', 'Survived']]
###########################################################################
# Split data into train and test                                          #
###########################################################################
trainData = fullData.loc[fullData.DataPartition == 'train']
testData = fullData.loc[fullData.DataPartition == 'test']
###########################################################################
# One hot encode                                                          #
###########################################################################
# https://github.com/scikit-learn-contrib/categorical-encoding
# http://contrib.scikit-learn.org/categorical-encoding/onehot.html
categories = list(set(trainData.select_dtypes(['category']).columns))
target = trainData.Survived
enc = OneHotEncoder(cols=categories,return_df = 1, handle_unknown = 'ignore').fit(trainData, target)
trainData = enc.transform(trainData)
testData = enc.transform(testData)
###########################################################################
# Drop multi collinear levels and no longer required                      #
###########################################################################
dropColumns = ['DataPartition']
trainData = trainData.drop(columns=dropColumns)
testData = testData.drop(columns=dropColumns)
testData = testData.drop(columns='Survived')
###########################################################################
# Start h2o cloud                                                         #
###########################################################################
h2o.init()
h2o.remove_all  # clean slate, in case cluster was already running
# upload data to h2o cloud
train = h2o.H2OFrame(trainData)
test = h2o.H2OFrame(testData)
# define target and feautures
target = 'Survived'
features = train.columns[:train.shape[1]-2]
train[target] = train[target].asfactor() # for binary classification, response should be a factor (ordinal / category)
# gbm
from h2o.estimators.gbm import H2OGradientBoostingEstimator
nFolds = 5
sampleRatePerClass = [0.62, 1]
gbm = H2OGradientBoostingEstimator(nfolds = nFolds, fold_assignment = "Modulo",keep_cross_validation_predictions = True,
                                   min_rows = 4, ntrees = 50, max_depth = 3, learn_rate = 0.01, balance_classes=True,
                                   stopping_metric = 'AUC', stopping_rounds = 3, stopping_tolerance = 1e-4, score_tree_interval = 10, seed = 333)
gbm.train(x = features, y = target, training_frame = train)
#gbm.confusion_matrix()
# gbm.varimp_plot()
# gbm.cross_validation_metrics_summary()
gbm.model_performance().auc()
# random forest
from h2o.estimators.random_forest import H2ORandomForestEstimator
rf = H2ORandomForestEstimator(nfolds = nFolds, fold_assignment = "Modulo",keep_cross_validation_predictions = True,
                                   min_rows = 4, ntrees = 100, max_depth = 6, balance_classes=True,
                                   stopping_metric = 'AUC', stopping_rounds = 3, stopping_tolerance = 1e-4, score_tree_interval = 10, seed = 333)
rf.train(x = features, y = target, training_frame = train)
rf.model_performance().auc()
# stacked ensemble
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
# metaLearnerParams = {'balance_classes': 'True'}
ensemble = H2OStackedEnsembleEstimator(base_models=[gbm, rf],
                                       metalearner_algorithm = 'glm', # metalearner_params = metaLearnerParams,
                                       seed = 333)
ensemble.train(x = features, y = target, training_frame = train)
ensemble.model_performance().auc()
# predict
finalPrediction = ensemble.predict(test[:-1])
# submit
submission = test.concat(finalPrediction,axis=1)[['PassengerId','predict']].as_data_frame(use_pandas=True)
submission.rename(columns={'predict': 'Survived'}, inplace=True)
submission.to_csv('submission.csv', index = False)
h2o.cluster().shutdown()