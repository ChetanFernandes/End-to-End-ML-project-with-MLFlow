from flask import render_template, jsonify, Flask, request, send_file
from src.exception_handling import CustomException
from src.logging import logging
from src.pipeline.Prediction_pipeline import prediction_pipeline
import sys,os,io
import subprocess
from src.utilis import *
import requests
import mlflow
import mlflow.pyfunc
import dagshub
import yaml
import boto3
from botocore.exceptions import ClientError
from src.constants.constants import *
from src.cloud_operations import *

app = Flask (__name__)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

 

@app.route("/")
def route():
    return "Welcome to my application"

@app.route ("/upload")
def data_upload_to_db():
    message = upload_data_db()
    return "Data successfully uploaded to MongoDB"

@app.route('/folder_structure')
def folder_structure():
    folder_structure = {}
    for root, dirs, files in os.walk("/"):  # Start from the root directory
        folder_structure[root] = {"dirs": dirs, "files": files}
        return jsonify(folder_structure)


@app.route('/train')
def trigger_pipeline():
        try:
            DVC = dvc()
            logging.info("Iinitialize_dvc_s3")
            DVC.initialize_dvc_s3()
            try:
                logging.info("Running the training pipeline")
                subprocess.run(["dvc", "repro"], check=True)

                # Step 2: Push artifacts to S3
                logging.info("Pushing tracked artifacts to S3")
                subprocess.run(["dvc", "push"], check=True)

                logging.info("Pipeline executed successfully, and artifacts pushed to S3")
                return "Training Successfull"
        
            except subprocess.CalledProcessError as e:
                logging.error(f"Error in DVC pipeline execution: {e}")
  
        except Exception as e:
            raise CustomException (e,sys)
      

# Predict using model
@app.route("/predict", methods = ["GET","POST"])
def predict():
    try:
        if request.method == 'POST':
            DVC = dvc()
            DVC.initialize_dvc_s3()
            DVC.load_model_and_processor()

            pred_pipe = prediction_pipeline(request)
            predicted_file_path = pred_pipe.run_pred_pipeline()
            return send_file(predicted_file_path, download_name = "predicted_file.csv", as_attachment = True)
        else:
            return render_template('index.html')
            
    except Exception as e:
        raise CustomException(e,sys)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug = True)



'''
def get_secret():

    secret_name = "DAGSHUB_TOKEN"
    region_name = "ap-south-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response['SecretString']
        return secret
    except ClientError as e:
        raise e

def initialize_dagshub_connection():
    dagshub_token = DAGSHUB_TOKEN
    os.environ["DAGSHUB_TOKEN"] = DAGSHUB_TOKEN
    dagshub.init(repo_owner='chetanfernandes',repo_name='End-to-End-ML-project-with-MLFlow',mlflow=True)
    if not dagshub_token:
        raise Exception("DagsHub token not found in AWS Secrets Manager.")
    

@app.route("/connect", methods=["GET"])
def connect_dagshub():
    try:
        initialize_dagshub_connection()
        return jsonify({"message": "Successfully connected to DagsHub."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    


@app.route("/configure")
def configure_s3():
    try:
        # Ensure AWS credentials are set as environment variables
        aws_access_key = AWS_ACCESS_KEY_ID
        aws_secret_key = AWS_SECRET_ACCESS_KEY
        aws_region = AWS_REGION

        if not all([aws_access_key, aws_secret_key, aws_region]):
            raise EnvironmentError("AWS credentials are not set in environment variables.")

        # Add S3 remote to DVC
        s3_remote_url = "https://phishingartifacts.s3.ap-south-1.amazonaws.com/artifacts/"
        subprocess.run(["dvc", "remote", "add", "-d", "s3remote", s3_remote_url], check=True)

        # Configure S3 credentials for DVC
        subprocess.run(["dvc", "remote", "modify", "s3remote", "access_key_id", aws_access_key], check=True)
        subprocess.run(["dvc", "remote", "modify", "s3remote", "secret_access_key", aws_secret_key], check=True)
        subprocess.run(["dvc", "remote", "modify", "s3remote", "region", aws_region], check=True)

        return ("DVC S3 remote successfully configured.")

    except subprocess.CalledProcessError as e:
        return(f"Error while configuring DVC remote: {e}")
    except Exception as e:
        return(f"Unexpected error: {e}")        
'''
    
