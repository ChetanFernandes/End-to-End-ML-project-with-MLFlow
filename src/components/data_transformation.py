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

@dataclass
class data_transformation_config:
    x:str = os.path.join("artifacts","x.csv")
    y:str = os.path.join("artifacts","y.csv")
    processor:str = os.path.join("artifacts","Processor.pkl")
    X_train:str = os.path.join("artifacts","X_train.csv")
    X_test:str = os.path.join("artifacts","X_test.csv")
    y_train:str = os.path.join("artifacts","y_train.csv")
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
            logging.info(f"Count of null values are {df.isnull().sum().sum()}")
            null_columns = []
            for  col in list(df.columns):
                if df[col].isnull().sum() > 0:
                    df[col].fillna(pd.median(col))
                    null_columns.append(col)
                else:
                    pass  
            logging.info(f"Number of columns with null values are {null_columns}")
            logging.info(f"Count of null values post filling them {df.isnull().sum().sum()}")

            #split in to x an y
            x = df.drop("Result", axis = 1)
            y = df[df.columns[-1]]
            logging.info(f" {x.shape,y.shape}")

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

            logging.info("Saving X_train,X_test,y_train,y_test")
            X_test.to_csv(self.Transformation_config.X_test, index = False)
            X_train.to_csv(self.Transformation_config.X_train, index = False)
            y_test.to_csv(self.Transformation_config.y_test, index = False)
            y_train.to_csv(self.Transformation_config.y_train, index = False)

            return {
                    self.Transformation_config.X_train, 
                    self.Transformation_config.X_test, 
                    self.Transformation_config.y_train,
                    self.Transformation_config.y_test,
                    processor_obj_path}
        except Exception as e:
            raise CustomException(e,sys)

        












