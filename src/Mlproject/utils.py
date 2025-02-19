import sys
import os
import pandas as pd

from src.Mlproject.logger import logging
from src.Mlproject.exception import CustomException
from dotenv import load_dotenv
import pymysql


load_dotenv()

host =os.getenv("host")
users=os.getenv("user")
password=os.getenv("password")
db=os.getenv('db')

def  read_sql_data():
    logging.info("Reading SQL database started")
    try:
        
        mydb=pymysql.connect(
            host ="localhost"
            users="root"
            password="12345"
            db="college"
            
            
        )
        
        logging.info("connection Establish",mydb)
        
    except Exception as ex:
        raise CustomException(ex)
    