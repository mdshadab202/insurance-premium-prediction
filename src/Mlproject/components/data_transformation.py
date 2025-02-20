# import sys
# from dataclasses import dataclass

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns 
# #modlling
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder,StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from src.Mlproject.exception import CustomException
# from src.Mlproject.logger import logging
# from src.Mlproject.utils import save_object
# import os


# @dataclass

# class DataTransformationConfig:
#     prerpocessor_obj_file_path =os.path.join('artifacts','preprocessor.pkl')
#     class DataTransformation:
#         def __init__(self):
#             self.data_transformation_config=DataTransformationConfig()
#         def get_data_tarnsformation_object(self):#x
        
        
#             '''this function is responsible for data transformation'''
            
#             try:
#                 num_features = X.select_dtypes(exclude="object").columns
#                 cat_features = X.select_dtypes(include="object").columns
            
            
#                 num_pipeline = Pipeline(steps=[
#                     ("imputer",SimpleImputer(strategy='median')),
#                     ('scaler',StandardScaler())
#                 ])
#                 cat_pipeline=Pipeline(steps=[
#                     ("imputer",SimpleIMputer(strategy='most_frequent')),
#                     ("one_hot_Encoder",OneHotEncoder()),
#                     ("scaler",StandardScaler(with_mean=False))
                
                
#                 ])
            
#                 logging.info(f"categorical columns:{cat_features}")
#                 logging.info(f"Numerical Columns:{num_features}")
            
#                 preprocessor = ColumnTransformer(
#                     [
#                         ("num_pipeline",num_pipeline,num_features),
#                         ("cat_pipeline",cat_pipeline ,cat_features)
                    
#                     ]
#                 )
            
#                 return preprocessor
            
            
#             except Exception as e:
#                 raise CustomException(e.sys)
        
#         def initiate_transformation(self,train_path, test_path):
#             try:
#                 train_df =pd.read_csv(train_path)
#                 test_df =pd.read_csv(test_path)
                
#                 logging.info("Reading The train and test file")
                
#                 preprocessing_obj =self.get_data_tarnsformation_object() #train_4f
                
#                 target_column_name= "expenses"
#                 numerical_columns = ["num_feature"]
                
#                 #divide the train dataset to dependent and independent feature
                
#                 input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
#                 target_feature_train_df= train_df[target_column_name]
                
#                 #divide the test dataset to dependent and independent feature
#                 input_feature_test_df =test_df.drop(columns=[target_column_name],axis=1)
#                 target_feature_test_df=test_df[target_column_name]
                
#                 logging.info("Applying preprocessing on training and test dataframe")
                
#                 # preprocessing_obj.fit_tarnsform(input_features_train_df)

#                 input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
#                 input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
                
#                 train_arr = np.c_[
#                     input_feature_train_arr, np.array(target_feature_train_df)
                    
#                 ]
                
#                 test_arr = np.c_[
#                     input_feature_test_arr, np.array(target_feature_test_df)
                    
#                 ]
                
#                 logging.info(f"Saved prerocessing object")   
                
#                 save_object(
#                     file_path = self.data_transformation_config.prerpocessor_obj_file_path
#                     # obj=preprocessing_obj
                    

#                 )
                
#                 return (
#                     train_arr,
#                     test_arr,
#                     self.data_transformation_config.prerpocessor_obj_file_path
#                 )
#             except Exception as e:
#                 raise CustomException(sys,e)
              
              
                
                
                
import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.Mlproject.exception import CustomException
from src.Mlproject.logger import logging
from src.Mlproject.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''This function is responsible for data transformation'''
        try:
            # Define categorical and numerical features
            num_features = ['age', 'bmi', 'children']  # Replace with actual numerical features
            cat_features = ['sex', 'smoker', 'region']  # Replace with actual categorical features

            # Define Pipelines
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"Numerical columns: {num_features}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_features),
                ("cat_pipeline", cat_pipeline, cat_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test datasets.")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "expenses"
            numerical_columns = ['age', 'bmi', 'children']  # Replace with actual numerical columns

            # Splitting features and target variable
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on training and test datasets.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
               
           