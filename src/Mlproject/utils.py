import sys
import os
import pandas as pd

from src.Mlproject.logger import logging
from src.Mlproject.exception import CustomException
from dotenv import load_dotenv
import pymysql

import pickle
import numpy as np


load_dotenv()

host =os.getenv("host")
users=os.getenv("user")
password=os.getenv("password")
db=os.getenv('db')

# def  read_sql_data():
#     logging.info("Reading SQL database started")
#     try:
        
#         mydb=pymysql.connect(
#             host ="localhost"
#             users="root"
#             password="12345"
#             db="college"
            
            
#         )
        
#         logging.info("connection Establish",mydb)
    
        
    # except Exception as ex:
    #     raise CustomException(ex)
    
def save_object(file_path,obj):
    try:
        dir_path =os.path.dirname(file_path)
            
        os.makedirs(dir_path,exist_ok=True)
            
        with open (file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
    
    
    except Exception as e:
        raise CustomException(e,sys)
                
                
                