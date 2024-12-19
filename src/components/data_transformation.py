from src.logging import logging
from src.exception_handling import CustomException
import os,sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler 
from src.utilis import save_processor_obj

@dataclass
class data_transformation_config:
    x:str = os.path.join("artifacts","x.csv")
    y:str = os.path.join("artifacts","y.csv")
    processor:str = os.path.join("artifacts","Processor.pkl")

class initiate_data_transformation:
    def __init__(self):
        self.Transformation_config = data_transformation_config

    def processor(self):
        try:
            logging.info("Inside Processor Function to create processor obj")
            processor = Pipeline([('scaler', RobustScaler())])
            logging.info("Calling save function in utilis to store processor object")
            path = save_processor_obj(self.Transformation_config.processor, processor)
            logging.info(f"processor object stored \n{path}")
            return path 
            
        except Exception as e:
            raise CustomException(e,sys)

    def transformation(self,path):
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
        processor_obj_path = self.processor()

        return self.Transformation_config.x, self.Transformation_config.y, processor_obj_path
        

        












