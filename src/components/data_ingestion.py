from src.exception_handling import CustomException
from src.logging import logging
import os,sys
import numpy as np
import pandas as pd
from pymongo import MongoClient
from src.utilis import upload_data_db
from dataclasses import dataclass


class Upload_data_to_db:
     def upload_data(self):
        try:
            url =  "mongodb+srv://chetan1:chetan1@cluster0.2c8ti.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
            Message = upload_data_db(url)
            return Message
        except Exception as e:
            raise Exception (e,sys)

@dataclass       
class Data_ingestion_config:
    raw_data_path:str = os.path.join("artifacts","raw_csv")


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

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index = False, header = True)
            return self.data_ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e,sys)







