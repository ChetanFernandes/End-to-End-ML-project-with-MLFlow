
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
    
def data_ingestion():
    try:
        logging.info("Under read_data_db function")
        obj1 = initiate_data_ingestion()
        raw_data_path = obj1.read_data_from_db(URL,MONGODB,Collection)
        logging.info("Data read successfully")
        logging.info(f" Raw data Path -> {raw_data_path}")
        return raw_data_path

    except Exception as e:
        raise CustomException (e,sys)
    

def data_transformation(raw_data_path):
    try:
        logging.info("Initaiting transformation")
        trans_config = initiate_data_transformation()
        x_train_path, x_test_path, y_train_path,y_test_path,processor_path = trans_config.transformation(raw_data_path)
        logging.info("Transformation Completed")
        logging.info(f"{x_train_path, x_test_path, y_train_path,y_test_path, processor_path}")
        return x_train_path, x_test_path, y_train_path,y_test_path,processor_path

    except Exception as e:
        raise CustomException (e,sys)  


def model_trainer(x_train_path,x_test_path, y_train_path,y_test_path):
    try:
        logging.info("Inside Model trainer")
        training = model_training()
        best_model_obj_path = training.iniiate_model_training(x_train_path, x_test_path, y_train_path,y_test_path)
        return  best_model_obj_path
    except Exception as e:
        raise CustomException (e,sys)  
    
    
def main():
    try:
        parser = argparse.ArgumentParser(description="ML Pipeline")
        
        # Define command-line arguments
        parser.add_argument('--task', choices=['data_ingestion', 'transformation', 'model_trainer'], required=True, help="Task to perform")
        parser.add_argument('--raw_data_path', help="Path to raw data")
        parser.add_argument('--x_train_path', help="Path to feature data")
        parser.add_argument('--x_test_path', help="Path to target data")
        parser.add_argument('--y_train_path', help="Path to feature data")
        parser.add_argument('--y_test_path', help="Path to target data")
        args = parser.parse_args()

        # Execute the task based on the argument
        if args.task == 'data_ingestion':
             data_ingestion()
        elif args.task == 'transformation':
            data_transformation(args.raw_data_path)
        elif args.task == 'model_trainer':
            best_hyper_tuned_model_obj_path = model_trainer(args.x_train_path,args.x_test_path,args.y_train_path,args.y_test_path)
            logging.info("Training and Hyper Tunning completed Completed")
            logging.info("Best model object post hyper_training saved in path ", best_hyper_tuned_model_obj_path)

    except Exception as e:
        raise CustomException (e,sys)  

if __name__ == '__main__':
    main()
    


        


