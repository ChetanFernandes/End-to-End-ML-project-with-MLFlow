import sys,os
from  src.exception_handling import CustomException
from src.logging import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from  sklearn.model_selection import train_test_split
from src.utilis import modeltraining, load_processor_obj,save_processor_obj, save_model,hyperparameter_tuning
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.model_selection import GridSearchCV


@dataclass
class train_test_config:
    X_train:str = os.path.join("artifacts","X_train.csv")
    X_test:str = os.path.join("artifacts","X_test.csv")
    y_train:str = os.path.join("artifacts","y_train.csv")
    y_test:str = os.path.join("artifacts","y_test.csv")
    Trained_model:str = os.path.join("artifacts","Model.pkl")
    yaml_file = os.path.join('config','model.yaml')

class model_training:
    def __init__(self):
        self.training_config = train_test_config()

    def train_test_split(self,x_path, y_path):
        try:
            logging.info(f"Inside train_test_split function")
            x = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            logging.info(f"{x.shape, y.shape}")
            logging.info("Replacing -1 with 0 to accomodate X_boost classifier")
            y.replace({-1: 0}, inplace=True)
            logging.info(f"{y.shape,y.value_counts()}")
            
            X_train,X_test,y_train,y_test = train_test_split(x,y, test_size= .30, random_state=1)
            
            
            return X_train,X_test,y_train,y_test
        
        except Exception as e:
            raise CustomException(e,sys)
    

    def load_processor_obj(self,processor_path):
        try:
            processor_obj = load_processor_obj(processor_path)
            return processor_obj

        except Exception as e:
            raise CustomException(e,sys)

    
    def iniiate_model_training(self,x_path, y_path, processor_path):
        try:
            
            logging.info("Splitting data in train and test")
            X_train,X_test,y_train,y_test = self.train_test_split(x_path, y_path)

            logging.info(f"{y_test.head(5)}")
            logging.info(f" Data splitted -> {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")

            X_test.to_csv(self.training_config.X_test, index = False)
            X_train.to_csv(self.training_config.X_train, index = False)
            y_test.to_csv(self.training_config.y_test, index = False)
            y_train.to_csv(self.training_config.y_train, index = False)
           
      
            logging.info("Calling the function to load processor object")
            processor_obj = self.load_processor_obj(processor_path)
            logging.info(f"Processor object loaded - {processor_obj}")
            
            
            logging.info("Apply scaling to X_train ans X_test")
            X_train = pd.DataFrame(processor_obj.fit_transform(X_train), columns = processor_obj.get_feature_names_out())
            X_test = pd.DataFrame(processor_obj.transform(X_test), columns = processor_obj.get_feature_names_out())

    
           
            logging.info(f"Calling save_proessor_obj function post scaling")
            updated_processor_obj_path =  save_processor_obj(processor_path,processor_obj)
            logging.info(f"Processor_object_save_successfully")
            
            best_model_obj = modeltraining(X_train,X_test,y_train,y_test)

            logging.info("Saving the best model object post Model training")
            path = save_model(self.training_config.Trained_model,best_model_obj)
            logging.info(f"Best model object post Model training saved -> {path}")
            print("Best model object post training saved in path ", path)
            
            logging.info("Initiating Hyper tunning")
            #best_hyper_tuned_model_obj = hyperparameter_tuning(self.training_config.yaml_file,X_train,X_test,y_train,y_test)

            logging.info("Saving the best model object post Hyper Tunning")
            #path = save_model(self.training_config.Trained_model,best_hyper_tuned_model_obj)
            logging.info(f"Best model object post Hyper Tunning saved -> {path}")
            
            return path 


        except Exception as e:
            raise CustomException(e,sys)
        
    
        







