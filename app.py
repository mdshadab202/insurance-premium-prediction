# from src.Mlproject.logger import logging
# from src.Mlproject.components.data_ingestion import DataIngestion
# from src.Mlproject.components.data_ingestion import DataIngestionConfig
# from src.Mlproject.exception import CustomException
# from src.Mlproject.components.data_transformation import DataTransformationConfig
# from src.Mlproject.components.data_transformation import DataTransformation
# import sys




# #

# if __name__=="__main__":
#     logging.info("The excution has started")
    
    
#     try:
        
#         # data_ingestion_Config = DataIngestionConfig()
#         data_ingestion= DataIngestion()
#         train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        
#         # Data_transformation_config = DataTransformationConfig()
#         data_transformation = DataTransformation() #Data
#         # Data_transformation.initiate_data_transformatioin( train_data_path,test_data_path)
#         data_transformation.initiate_transformation(train_data_path, test_data_path)#Data

        
#     except Exception as e:
#         logging.info("Custom_Exception")
#         raise CustomException(e,sys)

import sys
from src.Mlproject.logger import logging
from src.Mlproject.components.data_ingestion import DataIngestion
from src.Mlproject.components.data_transformation import DataTransformation
from src.Mlproject.exception import CustomException

if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_transformation(
            train_data_path, test_data_path
        )

        logging.info("Data transformation completed successfully.")

    except Exception as e:
        logging.error("CustomException occurred", exc_info=True)
        raise CustomException(e, sys)
