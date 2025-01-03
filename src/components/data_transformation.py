import os 
import sys 
from dataclasses import dataclass 
from pathlib import Path
import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path:Path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = [ "reading_score", "writing_score"]
            categorical_features = ["gender", "race_ethnicity", 
                                    "parental_level_of_education", "lunch",
                                     "test_preparation_course"]
            
            logging.info("Started building our preprocessing pipline")

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                    ]
                )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder())
                    ]
                )
            
            preprocessor = ColumnTransformer(
                [
                    ("Numerical_pipeline", numerical_pipeline, numerical_features),
                    ("Categorical_pipeline",categorical_pipeline, categorical_features)
                ]
            )
            logging.info("Preprocessing Pipeline has been created")

            return preprocessor             

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_file_path:Path, test_file_path:Path):
            try:
                logging.info("Initiation Data Transformation")

                train_df = pd.read_csv(train_file_path)
                test_df = pd.read_csv(test_file_path)
                logging.info("Imported train and test data set successfully")

                logging.info("Getting preprocessor object")
                preprocessor_object = self.get_data_transformer_object()

                target_feature = "math_score"
                X_train = train_df.drop(target_feature, axis=1)
                y_train = train_df[target_feature]
                X_test = test_df.drop(target_feature, axis=1)
                y_test = test_df[target_feature]
                logging.info("Completed splitting Dependent and Independent features")

                logging.info("Applying the preprocesor object")
                X_train_arr = preprocessor_object.fit_transform(X_train)
                X_test_arr = preprocessor_object.transform(X_test)
                logging.info("Preprocessing object applied successfully")

                train_arr = np.c_[X_train_arr, np.array(y_train)]
                test_arr = np.c_[X_test_arr, np.array(y_test)]

                logging.info("Saving Preprocessing object")
                save_object(file_path = self.data_transformation_config.preprocessor_file_path,
                            obj = preprocessor_object)

                return (train_arr, test_arr, 
                        self.data_transformation_config.preprocessor_file_path)
                 
            except Exception as e:
                raise CustomException(e, sys)


