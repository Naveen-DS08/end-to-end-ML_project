import os 
import sys 
from dataclasses import dataclass 
from pathlib import Path 
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error 

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array:np.array,
                                test_array:np.array):
        try:
            logging.info("Model Trainer Initiated")
            X_train, y_train, X_test, y_test =(train_array[:,:-1], train_array[:,-1],
                                                test_array[:,:-1], test_array[:,-1]) 
            
            models = {
                 "Linear Regressor": LinearRegression(),
                 "Decision Tree": DecisionTreeRegressor(), 
                 "Random Forest": RandomForestRegressor(),
                 "K-Neighbors": KNeighborsRegressor(),
                 "Ada Boosting": AdaBoostRegressor(),
                 "Gradient Boosting" : GradientBoostingRegressor(),
                 "XGBRegressor": XGBRegressor(),
                 "Cat Boost": CatBoostRegressor()
                }
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2', None],
                    },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                    },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                    },
                "Linear Regressor":{},
                "K-Neighbors":{
                     'n_neighbors': np.arange(1, 11),
                     'weights': ['uniform', 'distance'],
                     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                     'p': [1, 2]
                     },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                    },
                "Cat Boost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                    },
                "Ada Boosting":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                    }
                }
            
            logging.info("Model Training and evaluation started") 
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test,
                                                models=models, params=params)
            
            # To get best model 
            best_model_score = max(model_report.values())
            if best_model_score < 0.6:
                 raise CustomException("No best model found")
            
            best_model_name = list(model_report.keys())[
                 list(model_report.values()).index(best_model_score)]
            logging.info(f"Best Model Found:[Name:{best_model_name} with R2_score:{best_model_score}]")
            best_model = models[best_model_name]

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj = best_model)
            
            logging.info("Model file saved Successfully")
            
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2            

        except Exception as e:
                raise CustomException(e, sys)

