from src.logging import logging
from src.exception_handling import CustomException
import os,sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler 
from src.utilis import *
from  sklearn.model_selection import train_test_split
import boto3
from datetime import datetime
import subprocess
import time

@dataclass
class data_transformation_config:
    #timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    #output_dir = os.path.join("artifacts", timestamp)
    x:str = os.path.join("artifacts", "x.csv")
    y:str = os.path.join("artifacts", "y.csv")
    processor:str = os.path.join("artifacts", "Processor.pkl")
    X_train:str = os.path.join("artifacts","X_train.csv")
    X_test:str = os.path.join("artifacts","X_test.csv")
    y_train:str = os.path.join("artifacts", "y_train.csv")
    y_test:str = os.path.join("artifacts","y_test.csv")

class initiate_data_transformation:
    def __init__(self):
        self.Transformation_config = data_transformation_config()

    def transformation(self,path):
        try:
            logging.info("Reading CSV file")
            df = pd.DataFrame(pd.read_csv(path))
            logging.info(f"Data read. {df.head(10),df.shape}")


            logging.info(f"Drop the duplicate values if present -> {df.duplicated().sum()}")
            df.drop_duplicates(inplace = True)
            logging.info(f"Count of Duplicate values post dropping -> {df.duplicated().sum()}")

        
            logging.info(f"Check for null values")
            logging.info(f"Count of null values -> {df.isnull().sum().sum()}")
            null_columns = [col for  col in df.columns if df[col].isnull().sum() > 0]
            for col in null_columns:
                    df[col].fillna(df[col].median(), inplace=True)
                    null_columns.append(col)
            logging.info(f"Count of null values post filling them {df.isnull().sum().sum()}")

            logging.info("Split x and y")
            x = df.drop("Result", axis = 1)
            y = df[df.columns[-1]]
            logging.info(f" Feature and label split.{x.shape,y.shape}")

            os.makedirs(os.path.dirname(self.Transformation_config.x), exist_ok=True)
            x.to_csv(self.Transformation_config.x, index = False, header = True)
            y.to_csv(self.Transformation_config.y, index = False)
            
            logging.info("Calling processor funtion to create and store processor object")
            processor_obj_path = processor(self.Transformation_config.processor)
            
            logging.info("Splitting data in train and test")
            X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=.30, random_state=1)

            logging.info(f"{y_test.head(5)}")
            logging.info(f" Data splitted -> {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")

            logging.info("Calling the function to load processor object")
            processor_obj = load_processor_obj(processor_obj_path)
            logging.info(f"Processor object loaded - {processor_obj}")
            
            
            logging.info("Apply scaling to X_train ans X_test")
            X_train = pd.DataFrame(processor_obj.fit_transform(X_train), columns = processor_obj.get_feature_names_out())
            X_test = pd.DataFrame(processor_obj.transform(X_test), columns = processor_obj.get_feature_names_out())

            logging.info(f"Calling save_proessor_obj function post scaling")
            processor_obj_path = save_processor_obj(processor_obj_path,processor_obj)
            logging.info(f"Processor_object_save_successfully")

           
            logging.info("Saving  scaled data X_train,X_test,y_train,y_test")
            X_test.to_csv(self.Transformation_config.X_test, index = False)
            X_train.to_csv(self.Transformation_config.X_train, index = False)
            y_test.to_csv(self.Transformation_config.y_test, index = False)
            y_train.to_csv(self.Transformation_config.y_train, index = False)
            return {
                      self.Transformation_config.X_train,
                      self.Transformation_config.X_test,
                      self.Transformation_config.y_train,
                      self.Transformation_config.y_test,
                      processor_obj_path
                  }

    
            
        except Exception as e:
            raise CustomException(e,sys)

       
            '''          
            # Upload the model directly to S3 for inference
            logging.info("Upload the artifacts directly to S3 for inference")
            
            s3 = boto3.client("s3")
            bucket_name = "phishingartifacts"
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_mapping = {
                "Processor.pkl": self.Transformation_config.processor,
                "X_train.csv": self.Transformation_config.X_train,
                "X_test.csv": self.Transformation_config.X_test,
                "y_train.csv": self.Transformation_config.y_train,
                "y_test.csv": self.Transformation_config.y_test,
                "x.csv": self.Transformation_config.x,
                "y.csv": self.Transformation_config.y,
                            }
            for key, file_path in file_mapping.items():
                s3.upload_file(file_path, bucket_name, f"{key}_{timestamp}")

            logging.info("Processor uploaded to S3 as 'Processor.pkl'")

            return {
                    self.Transformation_config.X_train, 
                    self.Transformation_config.X_test, 
                    self.Transformation_config.y_train,
                    self.Transformation_config.y_test,
                    processor_obj_path}
        except Exception as e:
            raise CustomException(e,sys)
            '''

        












