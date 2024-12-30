from src.exception_handling import CustomException
from src.logging import logging
import os,sys
import numpy as np
import pandas as pd
from pymongo import MongoClient
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import json
import warnings
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler 
from src.constants.constants import URL
import mlflow

def upload_data_db():
        logging.info("Uploading Data to MongoDB")
        try:
            client = MongoClient(URL)
            db = client['phishing']
            collection = db['phishing_collection']
            df = pd.read_csv('notebook/data/Phishing.csv')
            data = df.to_dict('records')
            collection.insert_many(data)
            logging.info("CSV uploaded successfully")
            logging.info(f" DB created successfully. Existing DB's are - {client.list_database_names()}")
            all_documents = collection.find()
            return "Data successfully uploaded to MongoDB"

        except Exception as e:
            raise CustomException(e,sys)
        

def processor(path):
        try:
            logging.info("Inside Processor Function to create processor obj")
            processor = Pipeline([('scaler', RobustScaler())])
            logging.info("Calling save function in utilis to store processor object")
            path = save_processor_obj(path, processor)
            logging.info(f"processor object stored \n{path}")
            return path 
        except Exception as e:
          raise CustomException(e,sys)
        
def save_processor_obj(path,obj):
     try:

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"wb") as processor_obj:
            pickle.dump(obj,processor_obj)
        return path   
          
     except Exception as e:
          raise CustomException(e,sys)
     
def save_model(path,obj):
    try:
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok = True)
        with open(path,"wb") as model_obj:
            pickle.dump(obj,model_obj)
        return path 
 
    except Exception as e:
          raise CustomException(e,sys)
     
def load_processor_obj(path):
    try:
        logging.info("Inside processor obj loading function")
        with open(path,"rb") as obj:
            return pickle.load(obj)
        logging.info("Loaded processor object")

    except Exception as e:
          raise CustomException(e,sys)
    
def load_model_obj(path):
    try:
        logging.info("Inside model  obj loading function")
        with open(path,"rb") as obj:
            return pickle.load(obj)
        logging.info("Loaded model object")
        
    except Exception as e:
          raise CustomException(e,sys)

       
def modeltraining(X_train,X_test,y_train,y_test):
      
    try:
    
        mlflow_tracking_uri = "https://dagshub.com/ChetanFernandes/Mlflow.mlflow"
        mlflow_username = os.environ.get("MLFLOW_TRACKING_USERNAME", "default_username")
        mlflow_password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "default_password")


        print(f"MLFlow URI: {mlflow_tracking_uri}")
        print(f"MLFlow Username: {mlflow_username}")
        print(f"MLFlow Password: {mlflow_password}")


        mlflow.set_tracking_uri(mlflow_tracking_uri)
         
        models = { "LR" : LogisticRegressionCV(),
            "LSVC" : LinearSVC(),
            "RFC" : RandomForestClassifier(),
            "ABC" : AdaBoostClassifier(),
            "GBC" : GradientBoostingClassifier(),
            "DTC" : DecisionTreeClassifier(),
            "GNB" : GaussianNB()
                }
        model_list = []
        report = []

        for model_name, model_obj in models.items():
                logging.info(f" Model name -> {model_name}")
                logging.info("Cross Validation")
                scores = cross_validate(model_obj, X_train, y_train, cv = 5, scoring = 'accuracy')
                logging.info(f" Scores , {scores['test_score']}")
                mean_scores = np.mean(scores['test_score'])
                logging.info(f"Model {model_name}: CV Mean Accuracy: {mean_scores * 100:.2f}%")

                logging.info("Fit and Predict")
                model_obj.fit(X_train,y_train)
                y_pred = model_obj.predict(X_test)
                logging.info("Evaluation Metrics")
                auc_score = roc_auc_score(y_test,y_pred)
                logging.info(f"{auc_score}")
            
                accuracy = accuracy_score(y_test,y_pred)
                metrics = {"auc_score": auc_score, "Accuracy Score": accuracy}

                logging.info(f"classification report - {classification_report(y_test,y_pred)}")
                report.append(accuracy * 100)
                model_list.append(model_name)
                logging.info(f"\n{report},\n{model_list}")
                artifact_path = model_name
                try:
                    mlflow.end_run()
                    with mlflow.start_run(run_name = model_name):
                        # Log the error metrics that were calculated during validation
                        mlflow.log_metrics(metrics)
                        mlflow.log_params(model_obj.get_params(), "value 🚀".encode("ascii", "ignore").decode())

                        # Log an instance of the trained model for later use
                        #signature = infer_signature(X_train, best_model.predict(X_train))
                        
                        # Log an instance of the trained model for later use
                        mlflow.sklearn.log_model(sk_model= model_obj, input_example= X_train, artifact_path = artifact_path)
                    
                except Exception as e:
                    logging.error(f"MLFlow error for {model_name}: {str(e)}")
        
   
        logging.info("Log and return the best model")
        best_index = np.argmax(report)
        logging.info(f"Best_index -> {best_index}")

        best_model_name = model_list[best_index]
        logging.info(f"best_model_name -> {best_model_name}")

        best_model_obj = models[best_model_name]
        logging.info(f"best_model_obj -> {best_model_obj}")
        
        logging.info(f"Best model: {best_model_name} with accuracy: {report[best_index]:.2f}%")
        return best_model_obj
      
    except Exception as e:
         raise CustomException(e,sys)
    
def hyperparameter_tuning(path,X_train,X_test,y_train,y_test):
  
    try:
        #mlflow.sklearn.autolog() 

        mlflow_tracking_uri = "https://dagshub.com/ChetanFernandes/Mlflow.mlflow"
        mlflow_username = os.environ.get("MLFLOW_TRACKING_USERNAME", "default_username")
        mlflow_password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "default_password")


        print(f"MLFlow URI: {mlflow_tracking_uri}")
        print(f"MLFlow Username: {mlflow_username}")
        print(f"MLFlow Password: {mlflow_password}")


        mlflow.set_tracking_uri(mlflow_tracking_uri)

        Hyper_tuning_model_list = []
        Hyper_tuning_report = []

        
        Hyper_models = { 
                    "RFC" : RandomForestClassifier(), 
                    "GBC" : GradientBoostingClassifier()
                  
                 }
        for model_name, model_obj in Hyper_models.items():
                logging.info(f"Starting hyperparameter tuning for: {model_name}")
                model_param_grid = read_yaml_file(path)["model_selection"]["model"][model_name]["search_param_grid"]
                logging.info("Starting Grid search CV")
                grid_search = GridSearchCV(model_obj, param_grid = model_param_grid, cv=4, n_jobs=4, verbose=5)
                grid_search.fit(X_train,y_train)
                logging.info("Grid search CV ended")

    
                logging.info("Fetch the best model and evaluate")
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                test_accuracy = best_model.score(X_test, y_test)
                y_pred = best_model.predict(X_test)
                auc_score = roc_auc_score(y_test, y_pred)

                metrics = {"auc_score": auc_score, "test_accuracy": test_accuracy}
                Hyper_tuning_model_list.append(best_model)
                Hyper_tuning_report.append(test_accuracy)
                logging.info(f"Metrics for {model_name}: {metrics}")


            # Log the model with a unique artifact path
                artifact_path = model_name
                try:
                    with mlflow.start_run(run_name = model_name):
                        # Log the error metrics that were calculated during validation
                        mlflow.log_metrics(metrics)
                        #mlflow.log_params(best_params)
                        mlflow.log_params(best_params, "value 🚀".encode("ascii", "ignore").decode())

                        # Log an instance of the trained model for later use
                        #signature = infer_signature(X_train, best_model.predict(X_train))
                        
                        # Log an instance of the trained model for later use
                        mlflow.sklearn.log_model(sk_model= best_model, input_example= X_train, artifact_path = artifact_path)
                    
                except Exception as e:
                    logging.error(f"MLFlow error for {model_name}: {str(e)}")
        

         # Identify and log the best model
        best_index = np.argmax(Hyper_tuning_report)
        best_model = Hyper_tuning_model_list[best_index]
        logging.info(f" Best model is - {best_model} with accuracy -> {Hyper_tuning_report[best_index]}")
        try:
            with mlflow.start_run(run_name="Best_Model_Run"):
                mlflow.log_params(best_params, "value 🚀".encode("ascii", "ignore").decode())
                mlflow.sklearn.log_model(sk_model=best_model, input_example=X_train, artifact_path = best_model)
        except Exception as e:
                    logging.error(f"MLFlow error for {model_name}: {str(e)}")
        return best_model

    except Exception as e:
        raise CustomException(e, sys)


def read_yaml_file(path):
    try:
        with open(path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
            
    except Exception as e:
        raise CustomException(e, sys)
    
# Load DVC YAML
def load_dvc_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
        



     


        


     
