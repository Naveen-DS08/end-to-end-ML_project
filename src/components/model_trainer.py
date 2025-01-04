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
                 "XG Boost": XGBRegressor(),
                 "Cat Boost": CatBoostRegressor()
                }
            
            logging.info("Model Training and evaluation started") 
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test,
                                                models=models)
            
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

