
from src.exception_handling import CustomException
from src.logging import logging
import os,sys
from src.components.data_ingestion import Upload_data_to_db, initiate_data_ingestion
from src.components.data_transformation import initiate_data_transformation
from src.components.model_trainer import model_training
from src.constants.constants import URL,MONGODB,Collection
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import argparse


def data_upload():
    try:
        obj = Upload_data_to_db
        message = obj.upload_data()
        logging.info(f"{message}")
        print(message)

    except Exception as e:
        raise CustomException(e,sys)
    
def read_data_from_db():
    try:
        logging.info("Under read_data_db function")
        obj1 = initiate_data_ingestion()
        raw_data_path = obj1.read_data_from_db(URL,MONGODB,Collection)
        logging.info("Data read successfully")
        logging.info(f"{raw_data_path}")
        print(raw_data_path)
        return raw_data_path

    except Exception as e:
        raise CustomException (e,sys)
    

def transformation(raw_data_path):
    try:
        logging.info("Initaiting transformation")
        trans_config = initiate_data_transformation()
        x_path, y_path, processor_path = trans_config.transformation(raw_data_path)
        logging.info("Transformation Completed")
        print(x_path, y_path, processor_path)
        return x_path, y_path, processor_path

    except Exception as e:
        raise CustomException (e,sys)  


def model_trainer(x_path, y_path, processor_path):
    try:
        logging.info("Inside Model trainer")
        training = model_training()
        path = training.iniiate_model_training(x_path, y_path, processor_path)
        return path
    except Exception as e:
        raise CustomException (e,sys)  
    
    
def main():
    try:
        parser = argparse.ArgumentParser(description="ML Pipeline")
        
        # Define command-line arguments
        parser.add_argument('--task', choices=['data_ingestion', 'transformation', 'model_trainer'], required=True, help="Task to perform")
        parser.add_argument('--raw_data_path', help="Path to raw data")
        parser.add_argument('--x_path', help="Path to feature data")
        parser.add_argument('--y_path', help="Path to target data")
        parser.add_argument('--processor_path', help="Path to processor file")

        args = parser.parse_args()

        # Execute the task based on the argument
        if args.task == 'data_ingestion':
            read_data_from_db()
        elif args.task == 'transformation':
            transformation(args.raw_data_path)
        elif args.task == 'model_trainer':
            path  = model_trainer(args.x_path,args.y_path,args.processor_path)
            logging.info("Training and Hyper Tunning completed Completed")
            print("Best model object post hyper_training saved in path ", path)

    except Exception as e:
        raise CustomException (e,sys)  

if __name__ == '__main__':
    main()
    


        


