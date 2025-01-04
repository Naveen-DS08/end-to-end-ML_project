import os 
import sys 
from pathlib import Path
import numpy as np 
import pandas as pd
import pickle
import json
from src.exception import CustomException 
from sklearn.metrics import r2_score, root_mean_squared_error

def save_object(file_path:Path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
               pickle.dump(obj, file_obj)               

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,
                    X_test, y_test, models:dict):
    try:
        report = {}
        for i in range(len(list(models))):
             model = list(models.values())[i]

             model.fit(X_train, y_train)

             y_train_pred = model.predict(X_train)
             y_test_pred = model.predict(X_test)

             train_r2_score = r2_score(y_train, y_train_pred)
             train_rmse_score = root_mean_squared_error(y_train, y_train_pred)
             test_r2_score = r2_score(y_test, y_test_pred)
             test_rmse_score = root_mean_squared_error(y_test, y_test_pred) 

             report[list(models.keys())[i]] = test_r2_score 

             # Saving entire report
             with open("artifacts/model_report.txt", "w") as file:
                  file.write(json.dumps(report))
           

        return report    
                    
    except Exception as e:
        raise CustomException(e, sys)
