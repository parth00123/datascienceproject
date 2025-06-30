import os 
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj

@dataclass
class Data_transformation_config:
    preprocess_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_conifg = Data_transformation_config()
    
    def get_dataTransformer(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('Scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical features Standardization completed")

            logging.info("Categorical features encoding completed")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read training and testing data")

            logging.info("going to preprocess the training and testing data")

            preprocessing_obj = self.get_dataTransformer()

            target_col = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_train_features = train_df.drop(columns=[target_col],axis = 1)
            target_train_feature = train_df[target_col]

            input_test_features=test_df.drop(columns=[target_col],axis=1)
            target_test_feature=test_df[target_col]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_train_features)
            input_feature_test_arr=preprocessing_obj.transform(input_test_features)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_train_feature)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_feature)]

            logging.info(f"Saved preprocessing object.")

            save_obj(
                file_path=self.data_transformation_conifg.preprocess_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_conifg.preprocess_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
        
