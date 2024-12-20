from src.logging import logging
from src.exception_handling import CustomException
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os, sys
from src.utilis import *

@dataclass
class prediction_config:
    pred_file_input_dir:str = "prediction"
    predicted_file:str = os.path.join(pred_file_input_dir,"predicted_file.csv")
    transformed_prediction_file:str = os.path.join(pred_file_input_dir,"transformed_file.csv")

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
                    print(prediction_file_path)
                
                    return prediction_file_path

          
             except Exception as e:
                  raise CustomException(e,sys)

        def read_prediction_file(self,prediction_file_path):
             try:
             
                df = pd.read_csv(prediction_file_path)
                logging.info(f"Read prediction_file - \n {df.head(5), df.shape}")

                logging.info("Dropping the target_column")
                #df.drop([df.columns[-1]], axis = 1, inplace=True)

                logging.info("Check for null vales")
                for col in df.columns:
                    if df[col].isnull().sum() > 0:
                        df[col].fillna(df[col].median(), axis = 1 , inplace = True)
                    else:
                        pass
                    
                df.to_csv(self.prediction_configuration.transformed_prediction_file, index = False)
                logging.info(f" Predicted file path - {self.prediction_configuration.transformed_prediction_file}")
                print(self.prediction_configuration.transformed_prediction_file)
                return self.prediction_configuration.transformed_prediction_file
             
             except Exception as e:
                  raise CustomException(e,sys)
             
        def predict(self,df):
             try:
    
                model_file_path = os.path.join("artifacts", "Model.pkl")
                processor_file_path = os.path.join("artifacts", "Processor.pkl")

                logging.info("Loading  Model obj")
                model_obj = load_model_obj(model_file_path)
                logging.info("Model object loaded successfully")

                logging.info("Loading  processor  obj")
                processor_obj = load_processor_obj(processor_file_path)
                logging.info("Processor object loaded successfully")

                logging.info("Apply processor to predicted file")
                transformed_df = processor_obj.transform(df)

                logging.info("Applying prediction")
                logging.info(f"{model_obj}")
                preds = model_obj.predict(df)
                return preds
             
             except Exception as e:
                  raise CustomException(e,sys)
             
        def cleaned_predicted_file(self,cleaned_file_path):
             try:
                df:pd.DataFrame = pd.read_csv(cleaned_file_path)
                preds = self.predict(df)
                y_test_path = os.path.join("artifacts", "y_test.csv")
                y_test_result:pd.DataFrame = pd.read_csv(y_test_path)

                df:pd.DataFrame = pd.concat([df,y_test_result],ignore_index=False, axis = 1)

                df["Model_Result"] = [pred for pred in preds]
                df["Categorical_result"] = [pred for pred in preds]
                df["Categorical_result"].replace({0:"Phising", 1: "Safe"}, inplace = True)
                

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
                  predicted_file_path = self.cleaned_predicted_file(cleaned_file_path)
                  return predicted_file_path
             
             except Exception as e:
                 raise CustomException(e,sys)
             



                 
                

             



