from src.exception_handling import CustomException
from src.logging import logging
import os,sys
import numpy as np
import pandas as pd
from pymongo import MongoClient
import pickle


def ensure_directorty_exists_conatiner(file_path):
     try: 
          logging.info("Creating s3 directory to store artifacts")
          directory = os.path.dirname(file_path)
          os.makedirs(directory, exist_ok=True)
          logging.info(f"{file_path}")
          return file_path
     except Exception as e:
          return CustomException(e,sys)
