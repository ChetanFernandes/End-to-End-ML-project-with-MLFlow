from src.exception_handling import CustomException
from src.logging import logging
import os,sys
import numpy as np
import pandas as pd
from pymongo import MongoClient
import pickle
import subprocess
import logging


def ensure_directorty_exists_conatiner(file_path):
     try: 
          logging.info("Creating s3 directory to store artifacts")
          directory = os.path.dirname(file_path)
          os.makedirs(directory, exist_ok=True)
          logging.info(f"{file_path}")
          return file_path
     except Exception as e:
          return CustomException(e,sys)
     
class dvc:
    def __init__(self):
        self.model_file = "artifacts/Model.pkl"
        self.processor_file = "artifacts/Processor.pkl"
        self.X_test_file = "artifacts/X_test.csv"
        self.y_test_file = "artifacts/y_test.csv"
        
    def initialize_dvc_s3(self):
        try:
            # Fetch environment variables
            logging.info("Fetching environment Variables")
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            s3_bucket = "phishingartifacts"
            s3_remote_url = f"s3://phishingartifacts/artifacts/".rstrip("/")
            remote_name = "s3_remote"

            if not aws_access_key or not aws_secret_key or not s3_bucket:
                raise ValueError("AWS credentials or S3 bucket not configured in environment variables")

            if os.path.exists(".dvc"):
                logging.info("DVC is already initialized.")
            else:
                logging.info("Initializing DVC")
                subprocess.run(["dvc", "init"], check=True)
         
            remote_list = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True, check=True)
            logging.info(f" Remote list -> {remote_list.stdout}")

            if remote_name not in remote_list.stdout:
                subprocess.run(["dvc", "remote", "add", "-d", remote_name, s3_remote_url], check=True)
            else:
                subprocess.run(["dvc", "remote", "add", "-d", remote_name, s3_remote_url, "-f"], check=True)
                logging.info(f"Remote already exists")

            subprocess.run(["dvc", "remote", "modify", "s3_remote", "access_key_id", aws_access_key], check=True)
            subprocess.run(["dvc", "remote", "modify", "s3_remote", "secret_access_key", aws_secret_key], check=True)

            logging.info("DVC initialized and S3 remote configured")
            return "DVC initialized and S3 remote configured"
      
        except Exception as e:
            logging.error(f"Error initializing DVC with S3: {e}")
            raise CustomException (e,sys)
        
    def dvc_pull(self):
        try:
            logging.info("Fetching artifacts using DVC...")
            result = subprocess.run(["dvc", "pull"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info(result.stdout.decode())
            logging.info("Artifacts pulled successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error during DVC pull: {e.stderr.decode()}")
            raise CustomException(e, sys)
        
    def load_model_and_processor(self):
            try:
                self.dvc_pull()
                # Check if files exist locally after pull
                if not os.path.exists(self.model_file):
                    raise FileNotFoundError(f"Model file not found: {self.model_file}")
                if not os.path.exists(self.processor_file):
                    raise FileNotFoundError(f"Processor file not found: {self.processor_file}")
                if not os.path.exists(self.X_test_file):
                    raise FileNotFoundError(f"Test file not found: {self.X_test_file}")
                if not os.path.exists(self.y_test_file):
                    raise FileNotFoundError(f"Test file not found: {self.y_test_file}")


                logging.info(f"Model file located at: {self.model_file}")
                logging.info(f"Processor file located at: {self.processor_file}")
                logging.info(f"X_Testing file located at: {self.X_test_file}")
                logging.info(f"Y_test file located at: {self.y_test_file}")

            except Exception as e:
                raise CustomException(e, sys)

    









