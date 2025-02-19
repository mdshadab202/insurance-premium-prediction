from src.Mlproject.logger import logging
from src.Mlproject.components.data_ingestion import DataIngestion
from src.Mlproject.components.data_ingestion import DataIngestionConfig
import sys






if __name__=="__main__":
    logging.info("The excution has started")
    
    
    try:
        
        # data_ingestion_Config = DataIngestionConfig()
        data_ingestion= DataIngestion()
        data_ingestion.initiate_data_ingestion()
        
        
        
    except Exception as e:
        logging.info("Custom_Exception")
        raise CustomException(e,sys)