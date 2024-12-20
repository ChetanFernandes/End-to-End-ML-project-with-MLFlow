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
#from src.constants.constants import URL,MONGODB,Collection,DAGSHUB_TOKEN



app = Flask (__name__)


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')



# Initialize Dagshub with MLflow
dagshub.init(
    repo_owner='chetanfernandes',
    repo_name='End-to-End-ML-project-with-MLFlow',
    mlflow=True
)

import logging
logging.basicConfig(level=logging.DEBUG)

# Debug the environment token
print(f"DAGSHUB_TOKEN: {os.environ.get('DAGSHUB_TOKEN')}")

# Debug repository details
print(f"Repository: chetanfernandes/End-to-End-ML-project-with-MLFlow")
#dagshub.init(repo_owner='chetanfernandes', repo_name='End-to-End-ML-project-with-MLFlow', mlflow=True, auth = auth)

# Load DVC YAML
def load_dvc_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
    

@app.route("/")
def route():
    return "Welcome to my application"


@app.route('/train', methods = ["POST", "GET"])
def trigger_pipeline():
        try:
                if request.method == 'GET':
                     return jsonify({'message': 'Send a POST request with the stages to trigger the pipeline'}), 200

                data = request.get_json()
                print(data)
                if not data or 'stage' not in data or not isinstance(data['stage'], list):
                    return jsonify({'error': 'Invalid payload. Expected a list of stages.'}), 400
                
                results = []
                for stage in data['stage']:
                        try:
                            dvc_file_path = 'dvc.yaml'
                            dvc_pipeline = load_dvc_yaml(dvc_file_path)

                            if not stage or stage not in dvc_pipeline.get('stages', {}):
                                results.append({'stage': stage, 'status': 'failed', 'error': f"Invalid or missing stage: {stage}"})
                                continue

                            command = ["dvc", "repro", stage]
                            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
                            results.append({'stage': stage, 'status': 'success', 'output': result.stdout})
                        
                        except subprocess.CalledProcessError as e:
                            results.append({'stage': stage, 'status': 'failed', 'error': e.stderr})
                        except Exception as e:
                            results.append({'stage': stage, 'status': 'failed', 'error': str(e)})
            
                return jsonify({'results': results}), 200
              
        except Exception as e:
            return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

       
@app.route("/predict", methods = ["GET","POST"])
def predict():
    try:

        if request.method == 'POST':

            pred_pipe = prediction_pipeline(request)
            predicted_file_path = pred_pipe.run_pred_pipeline()
            return send_file(predicted_file_path, download_name = "predicted_file.csv", as_attachment = True)
        
        else:
            return render_template('index.html')
        
    except Exception as e:
        raise CustomException(e,sys)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug = True)
