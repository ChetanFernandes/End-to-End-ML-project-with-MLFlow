from src.logging import logging
from src.exception_handling import CustomException
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os, sys
from src.utilis import *
import boto3
from src.cloud_operations import *

@dataclass
class prediction_config:
    pred_file_input_dir:str = "prediction"
    predicted_file:str = os.path.join(pred_file_input_dir,"predicted_file.csv")
    transformed_prediction_file:str = os.path.join(pred_file_input_dir,"transformed_file.csv")
    LOCAL_container_PROCESSOR_PATH:str = os.path.join("s3_artifacts","Processor.pkl")
    LOCAL_container_MODEL_PATH:str = os.path.join("s3_artifacts","Model.pkl")
    # File paths in S3
    PROCESSOR_S3_PATH = "artifacts/files/md5/86/f6bad14dc9737fb23c3c23bd557f12"
    MODEL_S3_PATH = "artifacts/files/md5/96/f385843e4653b1ac737b3e882fa36f"
  

class prediction_pipeline:
        def __init__(self,request):
         self.request = request
         self.prediction_configuration = prediction_config()


        def save_input_file(self):
             try:
                    os.makedirs(self.prediction_configuration.pred_file_input_dir, exist_ok = True)
                    Input_csv_file = self.request.files['file']
                    prediction_file_path = os.path.join(self.prediction_configuration.pred_file_input_dir,Input_csv_file.filename)
                    Input_csv_file.save(prediction_file_path)
                    logging.info(f"{prediction_file_path}")
                    return prediction_file_path
             except Exception as e:
                  raise CustomException(e,sys)

        def read_prediction_file(self,prediction_file_path):
             try:
             
                df = pd.read_csv(prediction_file_path)
                logging.info(f"Read prediction_file - \n {df.head(5), df.shape}")

          
                logging.info("Check for null vales")
                for col in df.columns:
                    if df[col].isnull().sum() > 0:
                        if df[col].dtype == "object":
                            df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                            df[col].fillna(df[col].median(), inplace=True)
                    
                df.to_csv(self.prediction_configuration.transformed_prediction_file, index = False)
                logging.info(f" Predicted file path - {self.prediction_configuration.transformed_prediction_file}")
                return self.prediction_configuration.transformed_prediction_file
             
             except Exception as e:
                  raise CustomException(e,sys)
             

        def get_s3_client(self):
          try: 
               self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
               self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
               self.AWS_BUCKET_NAME = 'phishingartifacts' # Name of your S3 bucket
               self.AWS_REGION = os.getenv("AWS_REGION")  # Default region
               logging.info("Reading credentials for s3 bucket")

               if not self.AWS_ACCESS_KEY_ID or not self.AWS_SECRET_ACCESS_KEY:
                raise ValueError("AWS credentials are not set in environment variables")
               s3 = boto3.client(
                         "s3",
                         aws_access_key_id = self.AWS_ACCESS_KEY_ID,
                         aws_secret_access_key = self.AWS_SECRET_ACCESS_KEY,
                         region_name = self.AWS_REGION,
                    )
               logging.info("Credentials successfully read")
               return s3
          except Exception as e:
               raise CustomException(e,sys)
          

        def download_from_s3(self,s3_path,container_local_path):
          logging.info("Inside Down_load function")
          try:
               # Local paths
               s3 = self.get_s3_client()
               try: 
                    s3.head_object(Bucket=self.AWS_BUCKET_NAME, Key=s3_path)  # Check if the file exists
                    logging.info(f"File {s3_path} exists on S3")
               except Exception as e:
                    raise CustomException(e,sys)
          
               if s3:
                    file_path = ensure_directorty_exists_conatiner(container_local_path)
                    logging.info(f"{file_path}")
                    logging.info(f"Downloading {s3_path} from S3...")
                    s3.download_file(self.AWS_BUCKET_NAME, s3_path, file_path)
                    logging.info(f"File downloaded to {file_path}")
                    logging.info(f"File downloaded to {os.path.abspath(file_path)}")
                    return file_path 
          except Exception as e:
              raise CustomException(e,sys)
                    
        def predict(self,df,processor_path, model_path):
             try:
                logging.info("Loading  Model obj")
                model_obj = load_model_obj(model_path)
                logging.info("Model object loaded successfully")

                logging.info("Loading  processor  obj")
                processor_obj = load_processor_obj(processor_path)
                logging.info("Processor object loaded successfully")

                logging.info(f"Processor: {type(processor_obj)}, Fitted: {hasattr(processor_obj, 'transform')}")
                logging.info(f"Model: {type(model_obj)}, Fitted: {hasattr(model_obj, 'predict')}")

                logging.info("Apply processor to predicted file")
                df = processor_obj.transform(df)
               
                logging.info("Applying prediction")
                logging.info(f"{model_obj}")
                preds = model_obj.predict(df)
                return preds
             
             except Exception as e:
                  raise CustomException(e,sys)
             
        def cleaned_predicted_file(self,cleaned_file_path,LOCAL_container_PROCESSOR_PATH,LOCAL_container_Model_PATH):
             try:
                df:pd.DataFrame = pd.read_csv(cleaned_file_path)
                print(cleaned_file_path)
                preds = self.predict(df,LOCAL_container_PROCESSOR_PATH,LOCAL_container_Model_PATH )
                y_test_path = os.path.join("artifacts", "y_test.csv")
                y_test_result:pd.DataFrame = pd.read_csv(y_test_path)

                df:pd.DataFrame = pd.concat([df,y_test_result],ignore_index=False, axis = 1)

                df["Model_Result"] = [pred for pred in preds]
                df["Categorical_result"] = [pred for pred in preds]
                df["Categorical_result"].replace({-1:"Phising", 1: "Safe"}, inplace = True)
                

                dir_name = os.path.dirname(self.prediction_configuration.predicted_file)
                os.makedirs(dir_name, exist_ok=True)
                df.to_csv(self.prediction_configuration.predicted_file, index = False, header = True)
                logging.info(f"predictions completed {self.prediction_configuration.predicted_file}")
                return self.prediction_configuration.predicted_file

             except Exception as e:
                  raise CustomException(e,sys)
             
        def run_pred_pipeline(self):
             try:
                  prediction_file_path = self.save_input_file()
                  cleaned_file_path = self.read_prediction_file(prediction_file_path)
                  LOCAL_container_PROCESSOR_PATH = self.download_from_s3(self.prediction_configuration.PROCESSOR_S3_PATH, self.prediction_configuration.LOCAL_container_PROCESSOR_PATH)
                  LOCAL_container_Model_PATH = self.download_from_s3(self.prediction_configuration.MODEL_S3_PATH, self.prediction_configuration.LOCAL_container_MODEL_PATH)
                  predicted_file_path = self.cleaned_predicted_file(cleaned_file_path, LOCAL_container_PROCESSOR_PATH,LOCAL_container_Model_PATH)
                  return predicted_file_path
             
             except Exception as e:
                 raise CustomException(e,sys)
             



                 
                

             



