from src.exception_handling import CustomException
from src.logging import logging
import os,sys
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dataclasses import dataclass

@dataclass       
class Data_ingestion_config:
    #timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    #output_dir = os.path.join("artifacts", timestamp)
    raw_data_path:str = os.path.join("artifacts", "raw.csv")
   

class initiate_data_ingestion:
    def __init__(self):
        self.data_ingestion_config = Data_ingestion_config()

    def read_data_from_db(self,URL,MONGODB,Collection):
        try:
            logging.info("Reading the data from DB")
            client = MongoClient(URL)
            db = client[MONGODB]
            collection = db[Collection]
            df = pd.DataFrame(list(collection.find()))
            logging.info("Data Successully read")
            logging.info(f" \n{df.head()}, \n{df.shape}")

            logging.info("Removing columns '_id' from Dataframe")
            if df.columns[0] in df.columns:
                df.drop([df.columns[0]], axis = 1 , inplace = True)
                logging.info(f" Column removed successfully -> \n{df.head()}, \n{df.shape}")
            else:
                logging.info("No coloumn found")
            
            df.drop_duplicates(inplace = True)
            logging.info(f" \n{df.shape}")
            try:
                os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
                df.to_csv(self.data_ingestion_config.raw_data_path, index = False, header = True)
                logging.info(f"Raw data saved to {self.data_ingestion_config.raw_data_path}")
                return self.data_ingestion_config.raw_data_path
            except PermissionError as e:
                logging.info(f"PermissionError: {e}")
                raise CustomException(e, sys)
            except Exception as e:
                logging.info(f"Unexpected error: {e}")
                raise CustomException(e, sys)
            '''
            symlink_path = os.path.join("artifacts", "raw.csv")
            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                os.remove(symlink_path)
            os.symlink(self.data_ingestion_config.raw_data_path, symlink_path)
            # The symbolic link (raw.csv) points to the actual timestamped artifact 
            # file created during the pipeline execution.
            logging.info(f"Symbolic link created/updated at {symlink_path}.")
            '''

        except Exception as e:
            raise CustomException(e,sys)




