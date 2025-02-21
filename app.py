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

# import sys#
# from src.Mlproject.logger import logging
# from src.Mlproject.components.data_ingestion import DataIngestion
# from src.Mlproject.components.data_transformation import DataTransformation
# from src.Mlproject.exception import CustomException
# from src.Mlproject.components.model_trainer import ModelTrainerConfig
# from src.Mlproject.components.model_trainer import ModelTrainer
# from src.Mlproject.utils import save_object, evaluate_models


# if __name__ == "__main__":
#     logging.info("Execution has started")

#     try:
#         # Data Ingestion
#         data_ingestion = DataIngestion()
#         train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

#         # Data Transformation
#         data_transformation = DataTransformation()
#         train_arr, test_arr, preprocessor_path = data_transformation.initiate_transformation(
#             train_data_path, test_data_path)
        
#         #model Training
#         model_trainer=ModelTrainer()
#         print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        

#         logging.info("Data transformation completed successfully.")

#     except Exception as e:
#         logging.error("CustomException occurred", exc_info=True)
#         raise CustomException(e, sys)
from src.Mlproject.logger import logging
from src.Mlproject.exception import CustomException
from src.Mlproject.components.data_ingestion import DataIngestion
from src.Mlproject.components.data_transformation import DataTransformation
from src.Mlproject.components.model_trainer import ModelTrainer
from src.Mlproject.components.model_trainer import ModelTrainerConfig

import sys

if __name__ == "__main__":
    logging.info("Execution has started")

    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_transformation(train_data_path, test_data_path)

        # Model Training
        model_trainer = ModelTrainer()
        model_trainer.train_model(train_arr, test_arr)

    except Exception as e:
        logging.error("Custom_Exception", exc_info=True)
        raise CustomException(e, sys)
