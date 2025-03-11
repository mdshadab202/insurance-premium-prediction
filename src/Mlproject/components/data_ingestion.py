import sys
import os
import pandas as pd
from src.Mlproject.logger import logging
from src.Mlproject.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
# from src.Mlproject.components.data_transformation import DataTransformation
# from src.Mlproject.components.model_trainer import ModelTrainer

# from src.Mlproject.utils import 

# Creating a DataClass
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        # Initialize DataClass
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion has started.')

        try:
            # Check if the dataset exists
            data_path = os.path.join('notebook/data','insurance.csv')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at {data_path}")

            # Read CSV as DataFrame
            # df = pd.read_csv(os.path.join('notebook/data',"insurance.csv"))
            df = pd.read_csv(data_path)
            logging.info('Data successfully read from CSV.')

            # Create directories for saving data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Raw data saved successfully.")
            
            # Split data into train and test sets
            logging.info('Performing Train-Test Split...')
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            # logging.info('Data ingestion completed successfully.')
            logging.info(f"Data split complete. Train shape: {train_set.shape}, Test shape: {test_set.shape}")


            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f'Error occurred during data ingestion: {e}')
            raise CustomException(e, sys)

# Uncomment if you want to test this as a script

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    
    # data_transformation = DataTransformation()
    # train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    # model_trainer = ModelTrainer()
    # model_trainer.initiate_model_trainer(train_arr, test_arr)

