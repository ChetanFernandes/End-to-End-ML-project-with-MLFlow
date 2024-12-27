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
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
import dagshub, mlflow
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')
import yaml
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler 
from src.constants.constants import URL


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

     
def save_metrics_json(base_path, metric, model):
    try:
        logging.info("Inside JSON function")
        
        # Construct the directory and file path
        dir_path = os.path.dirname(base_path)
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"{model}_metrics.json")
        
        # Save metrics to JSON
        with open(file_path, "w") as file:
            json.dump(metric, file, indent=4)
        
        logging.info(f"Metrics logged successfully at {file_path}")
    
    except Exception as e:
        raise CustomException(f"Error in saving metrics JSON: {e}", sys)


#def integrate_ml_flow():
        dagshub.init(repo_owner='chetanfernandes', repo_name='End-to-End-ML-project-with-MLFlow', mlflow=True)
    
def configure_mlflow():
    dagshub_repo = "https://dagshub.com/ChetanFernandes/End-to-End-ML-project-with-MLFlow.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "ChetanFernandes"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")  
    mlflow.set_tracking_uri(dagshub_repo)

     
def modeltraining(X_train,X_test,y_train,y_test):
      
    try:
        '''
        try:
            a = integrate_ml_flow()
            logging.info(f"{a}")
        except Exception as e:
            logging.info(f" Error - {e,sys}") 
        '''

        models = { "LR" : LogisticRegressionCV(),
            "SVC" : SVC(),
            "LSVC" : LinearSVC(),
            "RFC" : RandomForestClassifier(),
            "ABC" : AdaBoostClassifier(),
            "GBC" : GradientBoostingClassifier(),
            "DTC" : DecisionTreeClassifier(),
            "GNB" : GaussianNB()
                }
            
        logging.info("Intializing ML flow")
        mlflow.set_experiment("Training_model_1.1")
        
        model_list = []
        report = []

        for i in range(len(models)):
            model = (list(models.values())[i])
            logging.info(f" Model name -> {model}")

            logging.info("Cross Validation")
            scores = cross_validate(model, X_train, y_train, cv = 5, scoring = 'accuracy', verbose=5)
            logging.info(f" Scores , {scores['test_score']}")
            mean_scores = np.mean(scores['test_score'])
            logging.info(f"Model {model}: CV Mean Accuracy: {mean_scores * 100:.2f}%")

           
            logging.info("Fit and Predict")
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)


            logging.info("Evaluation Metrics")
            try:
                auc_score = roc_auc_score(y_test,y_pred)
            except ValueError:
                auc_score = None

            accuracy = accuracy_score(y_test,y_pred)
            metrics = {"auc_score": auc_score, "Accuracy Score": accuracy}

            logging.info(f"classification report - {classification_report(y_test,y_pred)}")
            logging.info("Append scores")
            report.append(accuracy * 100)
            model_list.append(list(models.keys())[i])
            logging.info(f"\n{report},\n{model_list}")

        
            run_name = model_list[i]
            artifact_path = model_list[i]
            try:
                with mlflow.start_run(run_name = run_name):
                
                    mlflow.log_metrics(metrics)
                    mlflow.log_params(model.get_params())


                    signature = infer_signature(X_train, model.predict(X_train))
                    
                
                    # Log an instance of the trained model for later use
                    mlflow.sklearn.log_model(sk_model= model, input_example=X_train, artifact_path= artifact_path, signature = signature)
            
            except Exception as e:
                 logging.error(f"MLFlow error: {str(e)}")
                 #return jsonify({'error': 'Pipeline execution failed'}), 500


            '''
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr,thresholds, marker='.', label=f'ROC Curve (AUC = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray') # Diagonal line for random guessing
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            plt.grid()
            plt.show()
        
            '''
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

        logging.info("Creating a new ML flow experiment")
        #mlflow.set_experiment("Hyper_Training_1.6")

        Hyper_tuning_model_list = []
        Hyper_tuning_report = []

        
        Hyper_models = { 
                    "SVC" : SVC(),
                    "LSVC" : LinearSVC(),
                    "RFC" : RandomForestClassifier(), 
                    "GBC" : GradientBoostingClassifier()
                  
                 }
        for model_name, model_obj in Hyper_models.items():
            logging.info(f"Starting hyperparameter tuning for: {model_name}")
            model_param_grid = read_yaml_file(path)["model_selection"]["model"][model_name]["search_param_grid"]
            logging.info("Starting Grid search CV")
            grid_search = GridSearchCV(model_obj, param_grid = model_param_grid, cv=4, n_jobs=1, verbose=5)
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
                    signature = infer_signature(X_train, best_model.predict(X_train))
                    
                    # Log an instance of the trained model for later use
                    mlflow.sklearn.log_model(sk_model= best_model, input_example= X_train, artifact_path = artifact_path, signature = signature)
                    '''-
                    if hasattr(best_model, "feature_importances_"):
                        feature_importances = pd.Series(best_model.feature_importances_)
                        feature_importances.plot(kind="bar", title=f"{model_name} Feature Importances")
                        plt.tight_layout()
                        plt.savefig(f"{model_name}_feature_importances.png")
                        mlflow.log_artifact(f"{model_name}_feature_importances.png")
                    '''
            except Exception as e:
                logging.error(f"MLFlow error for {model_name}: {str(e)}")


         # Identify and log the best model
        best_index = np.argmax(Hyper_tuning_report)
        best_model = Hyper_tuning_model_list[best_index]
        logging.info(f"Best model: {list(Hyper_models.keys())[best_index]} with accuracy: {Hyper_tuning_report[best_index]:.2f}%")

        with mlflow.start_run(run_name="Best_Model_Run"):
            mlflow.sklearn.log_model(sk_model=best_model, input_example=X_train, artifact_path="best_model")

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
        



     


        


     
