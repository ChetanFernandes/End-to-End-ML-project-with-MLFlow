from src.exception_handling import CustomException
from src.logging import logging
import os,sys
import numpy as np
import pandas as pd
from pymongo import MongoClient
import pickle

def upload_data_db(url):
        logging.info("Uploading Data to MongoDB")
        try:
            client = MongoClient(url)
            db = client['phishing']
            collection = db['phishing_collection']
            df = pd.read_csv("notebook\data\Phishing.csv")
            data = df.to_dict('records')
            collection.insert_many(data)
            logging.info("CSV uploaded successfully")
            logging.info(f" DB created successfully. Existing DB's are - {client.list_database_names()}")
            all_documents = collection.find()
            return "Data successfully uploaded to MongoDB"

        except Exception as e:
            raise Exception(e,sys)
        
def save_obj(path,obj):
     try:

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"wb") as processor_obj:
            pickle.dump(obj,processor_obj)
        return path   
          
     except Exception as e:
          raise Exception(e,sys)
    