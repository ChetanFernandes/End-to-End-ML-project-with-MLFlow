
from src.exception_handling import CustomException
from src.logging import logging
import os,sys
from src.components.data_ingestion import Upload_data_to_db, initiate_data_ingestion
from src.components.data_transformation import initiate_data_transformation
from src.constants.constants import URL,MONGODB,Collection


class Training_pipeline:
    def data_upload(self):
        try:
            obj = Upload_data_to_db
            message = obj.upload_data(self)
            logging.info(f"{message}")
            print(message)

        except Exception as e:
            raise CustomException(e,sys)
        
    def read_data_from_db(self,URL,MONGODB,Collection):
        try:
            logging.info("Under read_data_db fnction")
            obj1 = initiate_data_ingestion()
            raw_data_path = obj1.read_data_from_db(URL,MONGODB,Collection)
            logging.info("Calling function read_data_path")
            logging.info(f"{raw_data_path}")
            return raw_data_path

        except Exception as e:
            raise CustomException (e,sys)
        

    def transformation(self,raw_data_path):
        try:
            logging.info("Initaiting transformation")
            trans_config = initiate_data_transformation()
            x_path, y_path, processor_path = trans_config.transformation(raw_data_path)
            logging.info("Transformation Completed")
            return x_path, y_path, processor_path

        except Exception as e:
            raise CustomException (e,sys)   
        

pipeline = Training_pipeline()
#Data_ingestion.data_upload()
raw_data_path = pipeline.read_data_from_db(URL,MONGODB,Collection)
print(raw_data_path)
x_path, y_path, processor_path = pipeline.transformation(raw_data_path)
print(x_path, y_path, processor_path)