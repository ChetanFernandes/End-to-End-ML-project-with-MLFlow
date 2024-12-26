import sys,os
from  src.exception_handling import CustomException
from src.logging import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from  sklearn.model_selection import train_test_split
from src.utilis import *
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import subprocess
import time

@dataclass
class train_test_config:
    #timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    #output_dir = os.path.join("artifacts", timestamp)
    Trained_model:str = os.path.join("artifacts", "Model.pkl")
    yaml_file = os.path.join('config','model.yaml')

    
class model_training:
    def __init__(self):
        self.model_training_config = train_test_config()

    def initiate_model_training(self, x_train_path, x_test_path, y_train_path,y_test_path):
        try:

            logging.info("Read Data for model training")

            X_train = pd.DataFrame(pd.read_csv(x_train_path))
            X_test = pd.DataFrame(pd.read_csv(x_test_path))
            y_train = pd.DataFrame(pd.read_csv(y_train_path))
            y_test = pd.DataFrame(pd.read_csv(y_test_path))

            logging.info(f"Data read  for model training-> {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")

            best_model_obj = modeltraining(X_train,X_test,y_train,y_test)

            logging.info("Saving the best model object post Model training")
            path = save_model(self.model_training_config.Trained_model,best_model_obj)
            logging.info(f"Best model object post Model training saved -> {path}")
 
            
            logging.info("Initiating Hyper tunning")
            best_hyper_tuned_model_obj = hyperparameter_tuning(self.model_training_config.yaml_file,X_train,X_test,y_train,y_test)

            logging.info("Saving the best model object post Hyper Tunning")
            best_hyper_tuned_model_obj_path = save_model(self.model_training_config.Trained_model,best_hyper_tuned_model_obj)
            logging.info(f"Best model object post Hyper Tunning saved -> {best_hyper_tuned_model_obj_path}")
        
            return best_hyper_tuned_model_obj_path

        except Exception as e:
            raise CustomException(e,sys)
        
    
        







